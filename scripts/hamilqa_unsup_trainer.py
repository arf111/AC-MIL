from tqdm import tqdm
from config import CONFIG, device
import torch
import torch.optim as optim

from monai.data import ThreadDataLoader
from torch.utils.data import WeightedRandomSampler
from torchmetrics.functional.classification import multiclass_confusion_matrix

from sklearn.metrics import cohen_kappa_score

from scripts import base_dir
from scripts.corn_utils import corn_label_from_logits, corn_probas
from scripts.losses import corn_loss
from scipy.stats import kendalltau
from util.metrics import ScottsPiQuadratic, accuracy_off1_macro
from imblearn.metrics import macro_averaged_mean_absolute_error as amae
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def build_mil_optimizer(tier_1_model, tier_2_model, lr):
    """Build optimizer with a separate LR multiplier for the Tier-1 encoder."""
    tier1_lr_mult = float(CONFIG.get("tier1_lr_mult", 0.1))

    param_groups = []
    tier1_trainable = [p for p in tier_1_model.parameters() if p.requires_grad]
    if tier1_trainable:
        param_groups.append({"params": tier1_trainable, "lr": lr * tier1_lr_mult})

    tier2_trainable = [p for p in tier_2_model.parameters() if p.requires_grad]
    if tier2_trainable:
        param_groups.append({"params": tier2_trainable, "lr": lr})

    if len(param_groups) == 0:
        raise ValueError("No trainable parameters found for optimizer setup.")

    return optim.AdamW(
        param_groups,
        weight_decay=CONFIG['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-6,
    )

class AFibQCAttentionMILPsuedoBagsUnsupTrainer:
    def __init__(self, tier_1_model, tier_2_model, train_dataset, val_dataset, test_dataset, batch_size, epochs, lr,
                 experiment=None, test_dataset_for_overlaying_patches=None):
        self.tier_1_model = tier_1_model
        self.tier_2_model = tier_2_model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.test_dataset_for_overlaying_patches = test_dataset_for_overlaying_patches
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.experiment = experiment
        self.device = device

        self.tier_1_model.to(self.device)
        self.tier_2_model.to(self.device)

        self.optimizer = build_mil_optimizer(
            tier_1_model=self.tier_1_model,
            tier_2_model=self.tier_2_model,
            lr=self.lr,
        )
        
        self.concept_dim = self.experiment.get_parameter("concept_dim")
        self.lambda_concepts = self.experiment.get_parameter("lambda_concepts")
        self.lambda_adv = self.experiment.get_parameter("lambda_adv")
        self.lambda_div = self.experiment.get_parameter("lambda_div")

        self.use_amp = self.experiment.get_parameter("use_amp")
        # GradScaler is needed to prevent underflow in float16 gradients
        self.scaler = torch.amp.GradScaler(device=self.device, enabled=self.use_amp)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)
        
    def train_one_supervised_epoch(self, epoch_number):
        self.tier_1_model.train()
        self.tier_2_model.train()

        # --- 1. DANN Schedule (Ganin et al.) ---
        # Update alpha based on epoch progress (0.0 -> 1.0)
        # We force the schedule to hit its peak at Epoch 250
        ramp_up_epochs = 150.0

        progress = min(1.0, epoch_number / ramp_up_epochs)

        gamma = 3.0  # Change 10 to a lower number (like 5 or 3) to slow down the attack!
        dann_schedule = 2.0 / (1.0 + np.exp(-gamma * progress)) - 1.0
        
        # Apply weighting if defined in experiment, else default to 1.0 * schedule
        adv_weight = self.experiment.get_parameter("lambda_adv") if self.experiment else 1.0
        current_alpha = adv_weight * dann_schedule

        # CRITICAL: Set alpha on Tier 1 (where the GRL is), handling DataParallel
        if hasattr(self.tier_1_model, 'module'):
            self.tier_1_model.module.grl.alpha = current_alpha
        else:
            self.tier_1_model.grl.alpha = current_alpha

        # lambda_sad = self.get_sad_weight(current_epoch=epoch_number, max_lambda=self.lambda_div)

        # --- 2. Trackers ---
        loss_tracker = {
            "total": 0.0, "task": 0.0, "concept": 0.0, 
            "adv": 0.0, "ortho": 0.0, "div": 0.0
        }
        
        # NEW: Track adversarial logits for entropy computation
        adv_logits_tracker = {'sharpness': [], 'nulling': [], 'aorta': []}
        adv_sample_ids_tracker = {'sharpness': [], 'nulling': [], 'aorta': []}
        adv_true_labels_tracker = {'sharpness': [], 'nulling': [], 'aorta': []}
        
        predicted_logits_all = []
        true_labels_all = []
        concept_predicted_logits_all = {'sharpness': [], 'nulling': [], 'aorta': []}
        concept_true_labels_all = {'sharpness': [], 'nulling': [], 'aorta': []}
        
        # Helper for handling the number of classes for CORN
        # Assuming 4 classes (0,1,2,3), logits should be shaped for that.
        N_CLASSES = 4 
        iterat = 0
        epoch_iterator = tqdm(self.train_loader, desc=f'Epoch {epoch_number+1}/{self.epochs} [Train] (α={current_alpha:.2f})', 
                              total=len(self.train_loader), unit='batch', dynamic_ncols=True)

        sample_offset = 0
        for batch in epoch_iterator:
            inputs = batch['pseudo_bags'].to(self.device, non_blocking=True) # Shape: (B, Bags, Patches, C, H, W)
            labels = batch['labels']
            
            # --- 3. Labels ---
            # Main Task Label
            label_quality = labels['quality_for_fibrosis_assessment'].to(self.device, non_blocking=True).long()
            
            B, Bags = inputs.shape[0], inputs.shape[1]

            def expand_to_bags(lbl):
                # (Batch) -> (Batch, 1) -> (Batch, Bags) -> (Batch * Bags)
                return lbl.view(B, 1).expand(B, Bags).reshape(-1)

            lbl_sharp = labels['sharpness'].to(self.device, non_blocking=True).long()
            lbl_null = labels['myocardium_nulling'].to(self.device, non_blocking=True).long()
            lbl_aorta = labels['enhancement_of_aorta_and_valves'].to(self.device, non_blocking=True).long()

            self.optimizer.zero_grad()

            # Automatically casts operations to float16 where safe
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp, dtype=torch.float16):
                
                # Forward Pass
                results_tier1 = self.tier_1_model(inputs)
                logits_task, _ = self.tier_2_model(results_tier1["fused_features"])
    
                # Loss Calculation
                prediction_loss = corn_loss(logits=logits_task, y_train=label_quality, num_classes=N_CLASSES)

                # Concept Supervision
                B, Bags = inputs.shape[0], inputs.shape[1]
                
                def expand_to_bags(lbl): 
                    return lbl.view(B, 1).expand(B, Bags).reshape(-1)
                
                l_sharp = corn_loss(logits=results_tier1["concept_logits"]["sharpness"], y_train=expand_to_bags(lbl_sharp), num_classes=N_CLASSES)
                l_null = corn_loss(logits=results_tier1["concept_logits"]["nulling"], y_train=expand_to_bags(lbl_null), num_classes=N_CLASSES)
                l_aorta = corn_loss(logits=results_tier1["concept_logits"]["aorta"], y_train=expand_to_bags(lbl_aorta), num_classes=N_CLASSES)
                concept_loss = (l_sharp + l_null + l_aorta) / 3.0
                
                # Adversarial Loss
                total_instances = B * Bags * inputs.shape[2]
                def expand_lbl_patch(l): 
                    return l.view(B, 1, 1).expand(B, Bags, inputs.shape[2]).reshape(total_instances)
                
                l_adv_sharp = corn_loss(logits=results_tier1["adv_logits"]["sharpness"], y_train=expand_to_bags(lbl_sharp), num_classes=N_CLASSES)
                l_adv_null = corn_loss(logits=results_tier1["adv_logits"]["nulling"], y_train=expand_to_bags(lbl_null), num_classes=N_CLASSES)
                l_adv_aorta = corn_loss(logits=results_tier1["adv_logits"]["aorta"], y_train=expand_to_bags(lbl_aorta), num_classes=N_CLASSES)
                adversarial_loss = (l_adv_sharp + l_adv_null + l_adv_aorta) / 3.0
                
                # NEW: Collect adversarial logits for entropy computation
                adv_logits_tracker['sharpness'].append(results_tier1["adv_logits"]["sharpness"].detach())
                adv_logits_tracker['nulling'].append(results_tier1["adv_logits"]["nulling"].detach())
                adv_logits_tracker['aorta'].append(results_tier1["adv_logits"]["aorta"].detach())
                for concept_name in ['sharpness', 'nulling', 'aorta']:
                    n_rows = int(results_tier1["adv_logits"][concept_name].shape[0])
                    if B > 0 and n_rows % B == 0:
                        repeats = n_rows // B
                        sample_ids = (
                            torch.arange(sample_offset, sample_offset + B, device=self.device, dtype=torch.long)
                            .repeat_interleave(repeats)
                        )
                        if concept_name == 'sharpness':
                            true_adv_labels = lbl_sharp.view(-1).long().repeat_interleave(repeats)
                        elif concept_name == 'nulling':
                            true_adv_labels = lbl_null.view(-1).long().repeat_interleave(repeats)
                        else:
                            true_adv_labels = lbl_aorta.view(-1).long().repeat_interleave(repeats)
                    else:
                        sample_ids = torch.full((n_rows,), sample_offset, device=self.device, dtype=torch.long)
                        if concept_name == 'sharpness':
                            fallback_lbl = int(lbl_sharp.view(-1)[0].item())
                        elif concept_name == 'nulling':
                            fallback_lbl = int(lbl_null.view(-1)[0].item())
                        else:
                            fallback_lbl = int(lbl_aorta.view(-1)[0].item())
                        true_adv_labels = torch.full((n_rows,), fallback_lbl, device=self.device, dtype=torch.long)
                    adv_sample_ids_tracker[concept_name].append(sample_ids.detach())
                    adv_true_labels_tracker[concept_name].append(true_adv_labels.detach())

                diversity_loss = results_tier1.get("aux_loss", torch.tensor(0.0, device=self.device))
                # Weighted Sum
                total_loss = prediction_loss + self.lambda_concepts * concept_loss + adversarial_loss + self.lambda_div * diversity_loss

            self.scaler.scale(total_loss).backward()
                
            # 2. Step the optimizer (skipped if NaNs found)
            self.scaler.step(self.optimizer)
            
            # 3. Update the scaler factor
            self.scaler.update()

            # --- 7. Logging Helpers ---
            bs = inputs.size(0)
            loss_tracker["total"] += total_loss.detach() * bs
            loss_tracker["task"] += prediction_loss.detach() * bs
            loss_tracker["concept"] += concept_loss.detach() * bs
            loss_tracker["adv"] += adversarial_loss.detach() * bs
            loss_tracker["div"] += diversity_loss.detach() * bs
            predicted_logits_all.append(logits_task.detach())
            true_labels_all.append(label_quality.detach())
            concept_predicted_logits_all['sharpness'].append(results_tier1["concept_logits"]["sharpness"].detach())
            concept_predicted_logits_all['nulling'].append(results_tier1["concept_logits"]["nulling"].detach())
            concept_predicted_logits_all['aorta'].append(results_tier1["concept_logits"]["aorta"].detach())
            concept_true_labels_all['sharpness'].append(expand_to_bags(lbl_sharp).detach())
            concept_true_labels_all['nulling'].append(expand_to_bags(lbl_null).detach())
            concept_true_labels_all['aorta'].append(expand_to_bags(lbl_aorta).detach())
            sample_offset += B

            # iterat += 1

            # if iterat == 3:
            #     break

        self.scheduler.step()
        
        # --- 8. Epoch Metrics ---
        # Concatenate all batches
        predicted_logits_all = torch.cat(predicted_logits_all)
        true_labels_all = torch.cat(true_labels_all)

        # Convert logits to labels using CORN utility
        # Assuming corn_label_from_logits exists in your utils
        predicted_labels = corn_label_from_logits(predicted_logits_all) 
        
        # Calculate Ordinal Metrics (QWK, etc.)
        # Assuming self.calculate_ordinal_metrics exists
        ordinal_metrics = self.calculate_ordinal_metrics(predicted_labels, true_labels_all)
        concept_ordinal_metrics = {}
        for concept_name in ['sharpness', 'nulling', 'aorta']:
            concept_pred_labels = corn_label_from_logits(torch.cat(concept_predicted_logits_all[concept_name]))
            concept_true_labels = torch.cat(concept_true_labels_all[concept_name]).long()
            concept_ordinal_metrics[concept_name] = self.calculate_ordinal_metrics(
                concept_pred_labels, concept_true_labels
            )
        
        # NEW: Compute Adversarial Entropy Metrics 
        # Concatenate adversarial logits from all batches
        concatenated_adv_logits = {}
        concatenated_adv_sample_ids = {}
        concatenated_adv_true_labels = {}
        for concept_name in ['sharpness', 'nulling', 'aorta']:
            concatenated_adv_logits[concept_name] = torch.cat(adv_logits_tracker[concept_name], dim=0)
            concatenated_adv_sample_ids[concept_name] = torch.cat(adv_sample_ids_tracker[concept_name], dim=0)
            concatenated_adv_true_labels[concept_name] = torch.cat(adv_true_labels_tracker[concept_name], dim=0).long()
        
        # Compute rank distribution metrics for adversarial logits
        adv_entropy_metrics = self.compute_adversarial_rank_distribution_metrics(
            concatenated_adv_logits, num_classes=N_CLASSES, sample_ids=concatenated_adv_sample_ids
        )
        adv_ordinal_metrics = {}
        for concept_name in ['sharpness', 'nulling', 'aorta']:
            adv_pred_labels = corn_label_from_logits(concatenated_adv_logits[concept_name])
            adv_ordinal_metrics[concept_name] = self.calculate_ordinal_metrics(
                adv_pred_labels, concatenated_adv_true_labels[concept_name]
            )
        
        # Normalize losses
        N = len(self.train_loader.dataset)
        
        for k in loss_tracker: 
            if isinstance(loss_tracker[k], torch.Tensor):
                loss_tracker[k] = loss_tracker[k].item() / N
            else:
                loss_tracker[k] = loss_tracker[k] / N
        
        # Log to Comet
        if self.experiment:
            log_data = {
                "Train/Loss": loss_tracker["total"],
                "Train/Prediction_Loss": loss_tracker["task"],
                "Train/Concept_Loss": loss_tracker["concept"],
                "Train/Diversity_Loss": loss_tracker["div"],
                "Train/Adversarial_Loss": loss_tracker["adv"],
                "Train/QWK": ordinal_metrics.get('qwk', 0),
                "Train/AMAE": ordinal_metrics.get('amae', 0),
                "Train/GRL_Alpha": current_alpha,
            }

            for concept_name, metrics in concept_ordinal_metrics.items():
                log_data.update({
                    f"Train/Concept_QWK/{concept_name}": metrics.get('qwk', 0),
                    f"Train/Concept_AMAE/{concept_name}": metrics.get('amae', 0),
                })
            
            # NEW: Add adversarial entropy metrics
            for concept_name, metrics in adv_entropy_metrics.items():
                log_data.update({
                    f"Train/Adv_Entropy/{concept_name}": metrics['entropy'],
                    f"Train/Adv_Entropy_Ratio/{concept_name}": metrics['entropy_ratio'],
                })
            for concept_name, metrics in adv_ordinal_metrics.items():
                log_data.update({
                    f"Train/Adv_QWK/{concept_name}": metrics.get('qwk', 0),
                })
            
            self.experiment.log_metrics(log_data, step=epoch_number)

        print(f'Train Loss: {loss_tracker["total"]:.4f}, QWK: {ordinal_metrics.get("qwk", 0):.4f}')
        
        return loss_tracker["total"]

    def test_one_supervised_epoch(self, test_loader, epoch_number, category='val', log_to_comet=True):
        self.tier_1_model.eval()
        self.tier_2_model.eval()

        running_loss = 0.0
        running_prediction_loss = 0.0
        running_concept_loss = 0.0
        unsup_scale_values = []
        concept_dim = int(getattr(self.tier_1_model, "concept_dim", self.concept_dim))
        
        # NEW: Track adversarial logits for entropy computation
        adv_logits_tracker = {'sharpness': [], 'nulling': [], 'aorta': []}
        adv_sample_ids_tracker = {'sharpness': [], 'nulling': [], 'aorta': []}
        adv_true_labels_tracker = {'sharpness': [], 'nulling': [], 'aorta': []}
        
        predicted_logits_all = []
        true_labels_all = []
        concept_predicted_logits_all = {'sharpness': [], 'nulling': [], 'aorta': []}
        concept_true_labels_all = {'sharpness': [], 'nulling': [], 'aorta': []}
        
        # For test category: store GPU tensors, convert to CPU at end to avoid per-batch sync
        patient_prediction_data = []  # Store (pred_logits, true_labels, patient_ids, batch_idx)
        
        N_CLASSES = 4 
        iterat = 0

        epoch_iterator = tqdm(test_loader, desc=f'Phase: {category}', total=len(test_loader), unit='batch', dynamic_ncols=True)

        sample_offset = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(epoch_iterator):
                inputs_volume = batch['pseudo_bags']
                labels = batch['labels']
                
                # Handle Variable Input
                if isinstance(inputs_volume, torch.Tensor):
                    vol_data = inputs_volume.squeeze(0)
                elif isinstance(inputs_volume, list):
                    vol_data = inputs_volume[0]
                else:
                    vol_data = inputs_volume
                
                label_quality = labels['quality_for_fibrosis_assessment'].to(self.device, non_blocking=True).long()
                lbl_sharp = labels['sharpness'].to(self.device, non_blocking=True).long()
                lbl_null = labels['myocardium_nulling'].to(self.device, non_blocking=True).long()
                lbl_aorta = labels['enhancement_of_aorta_and_valves'].to(self.device, non_blocking=True).long()

                # --- FIXED: Autocast with torch.amp ---
                with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=torch.float16):
                    
                    # 1. Loop Over PseudoBags
                    fused_bag_features_list = []
                    tier1_logits_storage = {'sharpness': [], 'nulling': [], 'aorta': []}

                    for bag_idx in range(len(vol_data)):
                        bag_tensor = vol_data[bag_idx]
                        if not isinstance(bag_tensor, torch.Tensor):
                            bag_tensor = torch.tensor(bag_tensor)
                        bag_tensor = bag_tensor.to(self.device, non_blocking=True)
                        # (1, 1, Patches, C, H, W)
                        bag_input = bag_tensor.unsqueeze(0).unsqueeze(0)

                        # Run Tier 1
                        results_bag = self.tier_1_model(bag_input)
                        
                        fused_bag_features_list.append(results_bag["fused_features"])
                        
                        tier1_logits_storage['sharpness'].append(results_bag["concept_logits"]["sharpness"])
                        tier1_logits_storage['nulling'].append(results_bag["concept_logits"]["nulling"])
                        tier1_logits_storage['aorta'].append(results_bag["concept_logits"]["aorta"])
                        
                        # NEW: Collect adversarial logits for entropy computation
                        if "adv_logits" in results_bag:
                            adv_logits_tracker['sharpness'].append(results_bag["adv_logits"]["sharpness"].detach())
                            adv_logits_tracker['nulling'].append(results_bag["adv_logits"]["nulling"].detach())
                            adv_logits_tracker['aorta'].append(results_bag["adv_logits"]["aorta"].detach())
                            current_batch_size = int(label_quality.view(-1).numel())
                            label_map = {
                                'sharpness': lbl_sharp.view(-1).long(),
                                'nulling': lbl_null.view(-1).long(),
                                'aorta': lbl_aorta.view(-1).long(),
                            }
                            for concept_name in ['sharpness', 'nulling', 'aorta']:
                                n_rows = int(results_bag["adv_logits"][concept_name].shape[0])
                                if current_batch_size > 0 and n_rows % current_batch_size == 0:
                                    repeats = n_rows // current_batch_size
                                    sample_ids = (
                                        torch.arange(
                                            sample_offset,
                                            sample_offset + current_batch_size,
                                            device=self.device,
                                            dtype=torch.long,
                                        ).repeat_interleave(repeats)
                                    )
                                    true_adv_labels = label_map[concept_name].repeat_interleave(repeats)
                                else:
                                    sample_ids = torch.full((n_rows,), sample_offset, device=self.device, dtype=torch.long)
                                    true_adv_labels = torch.full(
                                        (n_rows,),
                                        int(label_map[concept_name][0].item()),
                                        device=self.device,
                                        dtype=torch.long,
                                    )
                                adv_sample_ids_tracker[concept_name].append(sample_ids.detach())
                                adv_true_labels_tracker[concept_name].append(true_adv_labels.detach())

                    # 2. Aggregate
                    volume_features = torch.cat(fused_bag_features_list, dim=1)
                    if volume_features.size(-1) >= 4 * concept_dim:
                        z_unsup = volume_features[:, :, 3 * concept_dim: 4 * concept_dim]
                        unsup_scale_values.append(z_unsup.detach().float().std(unbiased=False).item())

                    # 3. Tier 2
                    logits_task, _ = self.tier_2_model(volume_features)
                    
                    # 4. Loss
                    prediction_loss = corn_loss(logits=logits_task, y_train=label_quality.view(-1), num_classes=N_CLASSES)
                    
                    num_bags = len(vol_data)
                    lbl_sharp_bags = lbl_sharp.expand(num_bags)
                    lbl_null_bags = lbl_null.expand(num_bags)
                    lbl_aorta_bags = lbl_aorta.expand(num_bags)
                    
                    cat_logits_sharp = torch.cat(tier1_logits_storage['sharpness'], dim=0)
                    cat_logits_null = torch.cat(tier1_logits_storage['nulling'], dim=0)
                    cat_logits_aorta = torch.cat(tier1_logits_storage['aorta'], dim=0)
                    
                    l_sharp = corn_loss(logits=cat_logits_sharp, y_train=lbl_sharp_bags, num_classes=N_CLASSES)
                    l_null = corn_loss(logits=cat_logits_null, y_train=lbl_null_bags, num_classes=N_CLASSES)
                    l_aorta = corn_loss(logits=cat_logits_aorta, y_train=lbl_aorta_bags, num_classes=N_CLASSES)
                    
                    concept_loss = (l_sharp + l_null + l_aorta) / 3.0

                    loss = prediction_loss + self.lambda_concepts * concept_loss

                running_loss += loss.detach()
                running_prediction_loss += prediction_loss.detach()
                running_concept_loss += concept_loss.detach()
                
                # Cast to float32 for metric calculation
                predicted_logits_all.append(logits_task.float())
                true_labels_all.extend(label_quality.view(-1))
                concept_predicted_logits_all['sharpness'].append(cat_logits_sharp.detach().float())
                concept_predicted_logits_all['nulling'].append(cat_logits_null.detach().float())
                concept_predicted_logits_all['aorta'].append(cat_logits_aorta.detach().float())
                concept_true_labels_all['sharpness'].append(lbl_sharp_bags.detach().long())
                concept_true_labels_all['nulling'].append(lbl_null_bags.detach().long())
                concept_true_labels_all['aorta'].append(lbl_aorta_bags.detach().long())

                # Store test data on GPU, defer CPU transfer to avoid per-batch sync
                if category == 'test':
                    patient_ids = batch.get('p_id', None)
                    batch_size_eval = label_quality.view(-1).numel()
                    
                    if isinstance(patient_ids, (str, int)):
                        patient_ids = [patient_ids]
                    if patient_ids is None or (hasattr(patient_ids, "__len__") and len(patient_ids) != batch_size_eval):
                        patient_ids = [f"{category}_patient_{batch_idx}_{i}" for i in range(batch_size_eval)]
                    
                    patient_prediction_data.append({
                        'pred_logits': logits_task.detach(),
                        'true_labels': label_quality.view(-1).detach(),
                        'patient_ids': patient_ids,
                        'batch_idx': batch_idx
                    })

                # iterat += 1
                # if iterat == 3:
                #     break
                sample_offset += int(label_quality.view(-1).numel())

        # Metrics
        predicted_labels = corn_label_from_logits(torch.cat(predicted_logits_all))
        true_labels_all = torch.stack(true_labels_all).long()
        
        ordinal_metrics = self.calculate_ordinal_metrics(predicted_labels, true_labels_all)
        concept_ordinal_metrics = {}
        for concept_name in ['sharpness', 'nulling', 'aorta']:
            concept_pred_labels = corn_label_from_logits(torch.cat(concept_predicted_logits_all[concept_name]))
            concept_true_labels = torch.cat(concept_true_labels_all[concept_name]).long()
            concept_ordinal_metrics[concept_name] = self.calculate_ordinal_metrics(
                concept_pred_labels, concept_true_labels
            )
        
        # Process test patient data - NOW transfer to CPU once for all batches
        patient_label_rows = []
        if category == 'test' and patient_prediction_data:
            for data in patient_prediction_data:
                batch_pred_labels = corn_label_from_logits(data['pred_logits'].float()).view(-1).cpu().numpy()
                batch_true_labels = data['true_labels'].cpu().numpy()
                patient_ids = data['patient_ids']
                batch_idx = data['batch_idx']
                
                for i in range(len(batch_true_labels)):
                    patient_label_rows.append([
                        f"{patient_ids[i]}",
                        int(batch_true_labels[i]),
                        int(batch_pred_labels[i]),
                        int(epoch_number),
                        category,
                    ])

        # NEW: Compute Adversarial Entropy Metrics for validation 
        adv_entropy_metrics = {}
        adv_ordinal_metrics = {}
        if all(len(adv_logits_tracker[concept]) > 0 for concept in ['sharpness', 'nulling', 'aorta']):
            # Concatenate adversarial logits from all batches
            concatenated_adv_logits = {}
            concatenated_adv_sample_ids = {}
            concatenated_adv_true_labels = {}
            for concept_name in ['sharpness', 'nulling', 'aorta']:
                concatenated_adv_logits[concept_name] = torch.cat(adv_logits_tracker[concept_name], dim=0)
                concatenated_adv_sample_ids[concept_name] = torch.cat(adv_sample_ids_tracker[concept_name], dim=0)
                concatenated_adv_true_labels[concept_name] = torch.cat(adv_true_labels_tracker[concept_name], dim=0).long()
            
            # Compute rank distribution metrics for adversarial logits
            adv_entropy_metrics = self.compute_adversarial_rank_distribution_metrics(
                concatenated_adv_logits, num_classes=N_CLASSES, sample_ids=concatenated_adv_sample_ids
            )
            for concept_name in ['sharpness', 'nulling', 'aorta']:
                adv_pred_labels = corn_label_from_logits(concatenated_adv_logits[concept_name])
                adv_ordinal_metrics[concept_name] = self.calculate_ordinal_metrics(
                    adv_pred_labels, concatenated_adv_true_labels[concept_name]
                )

        if log_to_comet and self.experiment:
            unsup_scale_mean = float(np.mean(unsup_scale_values)) if len(unsup_scale_values) > 0 else 0.0
            log_data = {
                f"{category}/Loss": running_loss.item() / len(test_loader),
                f"{category}/Prediction_Loss": running_prediction_loss.item() / len(test_loader),
                f"{category}/Concept_Loss": running_concept_loss.item() / len(test_loader),
                f"{category}/QWK": ordinal_metrics['qwk'],
                f"{category}/AMAE": ordinal_metrics['amae'],
                f"{category}/Unsup_Feature_Scale": unsup_scale_mean,
            }

            for concept_name, metrics in concept_ordinal_metrics.items():
                log_data.update({
                    f"{category}/Concept_QWK/{concept_name}": metrics.get('qwk', 0),
                    f"{category}/Concept_AMAE/{concept_name}": metrics.get('amae', 0),
                })
            
            # NEW: Add adversarial entropy metrics to validation logs
            for concept_name, metrics in adv_entropy_metrics.items():
                log_data.update({
                    f"{category}/Adv_Entropy/{concept_name}": metrics['entropy'],
                    f"{category}/Adv_Entropy_Ratio/{concept_name}": metrics['entropy_ratio'],
                })
            for concept_name, metrics in adv_ordinal_metrics.items():
                log_data.update({
                    f"{category}/Adv_QWK/{concept_name}": metrics.get('qwk', 0),
                })
                
            self.experiment.log_metrics(log_data, step=epoch_number)

            if category == 'test':
                conf_mat = multiclass_confusion_matrix(
                    preds=predicted_labels.view(-1),
                    target=true_labels_all.view(-1),
                    num_classes=N_CLASSES
                )
                conf_mat_np = conf_mat.detach().cpu().numpy()
                fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
                sns.heatmap(
                    conf_mat_np,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    cbar=False,
                    square=True,
                    ax=ax_cm,
                )
                ax_cm.set_title(f"{category.upper()} Confusion Matrix")
                ax_cm.set_xlabel("Predicted Label")
                ax_cm.set_ylabel("True Label")
                ax_cm.set_xticklabels([str(i + 1) for i in range(N_CLASSES)])
                ax_cm.set_yticklabels([str(i + 1) for i in range(N_CLASSES)], rotation=0)
                plt.tight_layout()
                self.experiment.log_figure(
                    figure_name=f"{category}_Confusion_Matrix",
                    figure=fig_cm,
                )
                plt.close(fig_cm)

            if patient_label_rows and category in ['test', 'test_val']:
                table_data = [
                    ["patient_id", "true_label", "predicted_label", "epoch", "split"],
                    *patient_label_rows
                ]
                self.experiment.log_table(
                    filename=f"{category}_patient_predictions_epoch_{epoch_number}.csv",
                    tabular_data=table_data
                )

        unsup_scale_mean = float(np.mean(unsup_scale_values)) if len(unsup_scale_values) > 0 else 0.0
        print(
            f'{category.upper()} Loss: {running_loss.item()/len(test_loader):.4f}, '
            f'QWK: {ordinal_metrics["qwk"]:.4f}, UnsupScale(std): {unsup_scale_mean:.4f}'
        )
        
        # NEW: Print adversarial entropy metrics for validation monitoring
        if adv_entropy_metrics:
            print(f"{category.upper()} Adversarial Entropy Metrics:")
            for concept_name, metrics in adv_entropy_metrics.items():
                print(f"  {concept_name.capitalize()}: Entropy={metrics['entropy']:.4f} "
                      f"(Ratio={metrics['entropy_ratio']:.3f})")
                
        return ordinal_metrics

    def train(self, early_stopping, train_weights=None, g: torch.Generator = None, seed_worker: callable = None):
        samples_train_weights = torch.from_numpy(train_weights).float()
        sampler = WeightedRandomSampler(weights=samples_train_weights, num_samples=len(samples_train_weights), replacement=True)

        self.train_loader = ThreadDataLoader(dataset=self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False, sampler=sampler, collate_fn=self.custom_collate_fn, pin_memory=True, worker_init_fn=seed_worker, generator=g)
        self.val_loader = ThreadDataLoader(dataset=self.val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, worker_init_fn=seed_worker, generator=g)

        for epoch in range(self.epochs):
            print(f'\n{"="*80}\nEpoch {epoch+1}/{self.epochs}')
            self.train_one_supervised_epoch(epoch)
            val_metrics = self.test_one_supervised_epoch(self.val_loader, epoch, category='val')
            qwk = val_metrics["qwk"]

            if epoch >= early_stopping.start_epoch:
                early_stopping(qwk, [self.tier_1_model, self.tier_2_model])
                if early_stopping.early_stop:
                    print("Early stopping triggered!")
                    break

    def test(self, model_save_path, g: torch.Generator = None, seed_worker: callable = None):
        self.test_loader = ThreadDataLoader(dataset=self.test_dataset, batch_size=1, shuffle=False, pin_memory=True, worker_init_fn=seed_worker, generator=g)

        self.tier_1_model.load_state_dict(torch.load(base_dir + "/" + CONFIG['model_path'] + "/" + model_save_path[0], weights_only=True))
        self.tier_2_model.load_state_dict(torch.load(base_dir + "/" + CONFIG['model_path'] + "/" + model_save_path[1], weights_only=True))

        test_metrics = self.test_one_supervised_epoch(self.test_loader, self.epochs, category='test')

        return test_metrics

    def custom_collate_fn(self, batch):
        # Assumes each sample['pseudo_bags'] is list/tuple of equal-shaped bags
        # or already a tensor of shape (n_bags, patches, C, H, W)
        first_pb = batch[0]["pseudo_bags"]

        if torch.is_tensor(first_pb):
            pseudo_bags = torch.stack([s["pseudo_bags"] for s in batch], dim=0)
        else:
            pseudo_bags = torch.stack(
                [torch.stack([torch.as_tensor(b) for b in s["pseudo_bags"]], dim=0) for s in batch],
                dim=0,
            )

        labels0 = batch[0]["labels"]
        labels = {
            k: torch.stack([torch.as_tensor(s["labels"][k]) for s in batch], dim=0)
            for k in labels0
        }

        out = {"pseudo_bags": pseudo_bags, "labels": labels}

        if "pseudo_bags_coords" in batch[0] and batch[0]["pseudo_bags_coords"] is not None:
            first_pc = batch[0]["pseudo_bags_coords"]
            if torch.is_tensor(first_pc):
                out["pseudo_bags_coords"] = torch.stack([s["pseudo_bags_coords"] for s in batch], dim=0)
            else:
                out["pseudo_bags_coords"] = torch.stack(
                    [torch.stack([torch.as_tensor(c) for c in s["pseudo_bags_coords"]], dim=0) for s in batch],
                    dim=0,
                )

        return out


    def compute_adversarial_rank_distribution_metrics(self, adv_logits, num_classes=4, sample_ids=None):
        """
        Compute rank distribution metrics for adversarial logits to evaluate entropy maximization.
        
        Args:
            adv_logits: Dictionary with keys ['sharpness', 'nulling', 'aorta'],
                       each containing CORN logits of shape (instances, num_classes-1)
            num_classes: Number of ordinal classes (default 4 for 0,1,2,3)
            sample_ids: Optional dictionary mapping concept -> tensor of shape (instances,)
                        containing sample ids for per-sample aggregation.
            
        Returns:
            Dictionary mapping concept names to their rank distribution metrics:
            - entropy: Current entropy value 
            - entropy_ratio: Ratio to maximum possible entropy (should approach 1.0)
        """
        rank_metrics_per_concept = {}
        max_entropy = np.log(num_classes)  # log(4) ≈ 1.386 for 4 classes
        
        concept_names = ['sharpness', 'nulling', 'aorta']
        
        for concept_name in concept_names:
            if concept_name not in adv_logits:
                continue
                
            # Extract logits for this concept: (instances, num_classes-1)
            concept_logits = adv_logits[concept_name]
            
            with torch.no_grad():
                # Convert CORN logits to cumulative probabilities q_k=P(y>k)
                _, cum_probs = corn_probas(concept_logits)

                # Convert cumulative probabilities to class probabilities.
                eps = 1e-8
                class_probs = torch.zeros(
                    (cum_probs.size(0), num_classes),
                    device=cum_probs.device,
                    dtype=cum_probs.dtype,
                )
                class_probs[:, 0] = 1.0 - cum_probs[:, 0]
                for m in range(1, num_classes - 1):
                    class_probs[:, m] = cum_probs[:, m - 1] - cum_probs[:, m]
                class_probs[:, num_classes - 1] = cum_probs[:, num_classes - 2]
                class_probs = class_probs / (class_probs.sum(dim=1, keepdim=True) + eps)

                # Aggregate by sample first, then compute per-sample entropy and average.
                if sample_ids is not None and concept_name in sample_ids:
                    sample_ids_concept = sample_ids[concept_name].to(class_probs.device).long()
                    if sample_ids_concept.numel() != class_probs.size(0):
                        raise ValueError(
                            f"sample_ids size mismatch for {concept_name}: "
                            f"{sample_ids_concept.numel()} vs {class_probs.size(0)}"
                        )
                    unique_ids, inverse = torch.unique(sample_ids_concept, return_inverse=True)
                    summed = torch.zeros(
                        (unique_ids.numel(), num_classes),
                        device=class_probs.device,
                        dtype=class_probs.dtype,
                    )
                    counts = torch.zeros((unique_ids.numel(), 1), device=class_probs.device, dtype=class_probs.dtype)
                    summed.index_add_(0, inverse, class_probs)
                    counts.index_add_(
                        0,
                        inverse,
                        torch.ones((class_probs.size(0), 1), device=class_probs.device, dtype=class_probs.dtype),
                    )
                    sample_rank_probs = summed / (counts + eps)
                    sample_rank_probs = sample_rank_probs / (sample_rank_probs.sum(dim=1, keepdim=True) + eps)
                else:
                    sample_rank_probs = class_probs

                sample_entropies = -torch.sum(sample_rank_probs * torch.log(sample_rank_probs + eps), dim=1)
                entropy = sample_entropies.mean()
                
                rank_metrics_per_concept[concept_name] = {
                    'entropy': entropy.item(),
                    'entropy_ratio': entropy.item() / max_entropy,  # Should approach 1.0
                    'entropy_std': sample_entropies.std(unbiased=False).item(),
                    'n_samples': int(sample_entropies.numel()),
                }
        
        return rank_metrics_per_concept

    def calculate_ordinal_metrics(self, predicted_labels, true_labels, is_torch=True):
        """Calculate ordinal-aware metrics"""
        if is_torch:
            predicted_labels_np = predicted_labels.detach().cpu().numpy()
            true_labels_np = true_labels.detach().cpu().numpy()
        else:
            predicted_labels_np = predicted_labels
            true_labels_np = true_labels
        
        predicted_labels_np = np.asarray(predicted_labels_np).reshape(-1).astype(np.int64, copy=False)
        true_labels_np = np.asarray(true_labels_np).reshape(-1).astype(np.int64, copy=False)

        scotts_pi = ScottsPiQuadratic()
        
        # Agreement metrics
        qwk = cohen_kappa_score(true_labels_np, predicted_labels_np, weights='quadratic')
        linear_kappa = cohen_kappa_score(true_labels_np, predicted_labels_np, weights='linear')
        kendall_tau, _ = kendalltau(true_labels_np, predicted_labels_np)
        
        # Standard metrics
        acc_off1 = accuracy_off1_macro(true_labels_np, predicted_labels_np)
        try:
            # imblearn's macro-averaged MAE averages over the union of true and
            # predicted classes; if a class is predicted but never occurs in
            # true_labels_np (common on tiny/imbalanced batches), it hits
            # mean_absolute_error on an empty array and raises ValueError.
            amae_score = amae(y_true=true_labels_np, y_pred=predicted_labels_np)
        except ValueError:
            amae_score = float('nan')
        
        return {
            'one_off_accuracy': acc_off1,
            'amae': amae_score,
            'qwk': qwk,
            'linear_weighted_kappa': linear_kappa,
            'kendall_tau': kendall_tau,
            'scotts_pi': scotts_pi.fit_score(true_labels_np, predicted_labels_np) if len(np.unique(true_labels_np)) > 1 else 0.0
        }

