import argparse
import datetime
import os
import socket
import traceback

from comet_ml import Experiment
from config import CONFIG, MODEL_NAME, config, disable_comet, machine
import torch
import numpy as np
import random
from model import AttentionMILPseudoBagTier1Unsup, AttentionMILPseudoBagTier2Unsup
from scripts.hamilqa_unsup_trainer import AFibQCAttentionMILPsuedoBagsUnsupTrainer
from scripts.qc_dataset import get_patient_records_monai

from util.early_stopping import EarlyStopping
from util.mil_utils import get_overlay_test_transform, get_transforms
os.environ['COMET_DISABLE_ANNOUNCEMENT'] = "1"
import glob
import json
from pathlib import Path

from monai.data import CacheDataset, Dataset
from sklearn.model_selection import StratifiedKFold, train_test_split

# Load repository .env (if present) so COMET_API_KEY is available without manual export.
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)
except Exception:
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        for raw_line in env_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)

CONFIG = config[MODEL_NAME]

def _notification_value(value):
    if isinstance(value, dict):
        return {str(k): _notification_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_notification_value(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _build_notification_data(
    config_dict,
    fold=None,
    start_time_utc=None,
    fold_saved_model_name=None,
    test_results=None,
):
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    runtime_seconds = None
    if start_time_utc is not None:
        runtime_seconds = int((now_utc - start_time_utc).total_seconds())

    data = {
        "hostname": socket.gethostname(),
        "cwd": os.getcwd(),
        "machine": machine,
        "model_name": MODEL_NAME,
        "fold": fold,
        "start_time_utc": start_time_utc.isoformat() if start_time_utc is not None else None,
        "event_time_utc": now_utc.isoformat(),
        "runtime_seconds": runtime_seconds,
        "batch_size": config_dict.get("batch_size"),
        "epochs": config_dict.get("epochs"),
        "learning_rate": config_dict.get("learning_rate"),
        "classification": config_dict.get("classification"),
        "data_path": config_dict.get("data_path"),
        "model_path": config_dict.get("model_path"),
        "n_patches": config_dict.get("n_patches"),
        "no_of_pseudo_bags": config_dict.get("no_of_pseudo_bags"),
        "tier1_encoder_name": config_dict.get("tier1_encoder_name"),
        "saved_model_name": fold_saved_model_name,
    }
    if test_results is not None:
        data["test_results"] = _notification_value(test_results)
    return data


def _safe_send_comet_notification(experiment, title, status, additional_data):
    try:
        experiment.send_notification(
            title=title,
            status=status,
            additional_data=additional_data,
        )
    except Exception as notify_error:
        print(f"Warning: failed to send Comet notification '{title}': {notify_error}")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def parse_arguments():
    """Parse command-line arguments for HamilQA unsupervised model"""
    parser = argparse.ArgumentParser(description="Run HamilQA Unsupervised Model Training")
    parser.add_argument("--beta", type=float, default=1e-3, help="Override beta value")
    parser.add_argument("--learning_rate", type=float, default=CONFIG['learning_rate'], help="Override learning rate")
    parser.add_argument("--no_of_pseudo_bags", type=int, default=CONFIG['no_of_pseudo_bags'], help="Override no_of_pseudo_bags value")
    parser.add_argument("--n_patches", type=int, default=CONFIG['n_patches'], help="No of patches to extract from each image")
    parser.add_argument("--seed", type=int, default=CONFIG['seed'], help="Random seed")
    parser.add_argument(
        "--min_la_slices",
        type=int,
        default=CONFIG.get("min_la_slices", 0),
        help=(
            "Keep only scans whose stored post-Spacingd LA-containing axial "
            "slice count is strictly greater than this value."
        ),
    )
    return parser.parse_args()

def update_config(args):
    """Update CONFIG based on command-line arguments"""
    params_to_update = [
        "learning_rate",
        "no_of_pseudo_bags",
        "n_patches",
        "seed",
        "min_la_slices",
    ]

    for param in params_to_update:
        arg_value = getattr(args, param)
        if arg_value is not None and param in CONFIG:
            CONFIG[param] = arg_value
            print(f"Overriding {param} to {arg_value}")

    return CONFIG


def set_seed(seed):
    """Set random seed for reproducibility"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_qc_scores(
    data_files,
    qc_dict_json_path,
    require_concept_labels=True,
    min_la_slices=0,
):
    """
    Get the quality scores for the data files.
    For AFibQCAttentionMILPsuedoBagsUnsupervisedNet2D, concept labels are required.

    Args:
        data_files (list): List of data file paths.
        qc_dict_json_path (str): Path to the JSON file containing quality scores.
        require_concept_labels (bool): If True, only include files with concept labels.
        min_la_slices (int): Keep only scans with a stored post-Spacingd
            LA-containing axial-slice count strictly greater than this value.

    Returns:
        tuple: (qc_scores, labeled_data_files, qc_dict_scores, unlabeled_data_files)
    """
    with open(qc_dict_json_path, 'r') as f:
        qc_dict = json.load(f)

    qc_scores = []
    qc_dict_scores = {}
    labeled_data_files = []
    unlabeled_data_files = []
    excluded_by_la_slice_count = 0

    if min_la_slices < 0:
        raise ValueError("min_la_slices must be non-negative")

    for data_file in data_files:
        data_file_name = data_file.split("/")[-1]

        # Skip files with wrong segmentation
        if data_file_name in qc_dict and "segmented_region_indices" in qc_dict[data_file_name]:
            if qc_dict[data_file_name]["segmented_region_indices"] == "wrong":
                continue

        if data_file_name in qc_dict and qc_dict[data_file_name]["label"] != 0:
            la_slice_count = qc_dict[data_file_name].get(
                "la_axial_slice_count_after_spacing"
            )
            if la_slice_count is None or la_slice_count <= min_la_slices:
                excluded_by_la_slice_count += 1
                continue

            # Check for concept labels (required for unsupervised model)
            if require_concept_labels:
                has_sharpness = "sharpness" in qc_dict[data_file_name]["label"].keys()
                has_myocardium_nulling = "myocardium_nulling" in qc_dict[data_file_name]["label"].keys()
                has_enhancement = "enhancement_of_aorta_and_valves" in qc_dict[data_file_name]["label"].keys()

                if not (has_sharpness and has_myocardium_nulling and has_enhancement):
                    continue

                if "sharpness" in qc_dict[data_file_name]["label"]:
                    if qc_dict[data_file_name]["label"]["sharpness"] == 5:
                        qc_dict[data_file_name]["label"]["sharpness"] = 4
                if "myocardium_nulling" in qc_dict[data_file_name]["label"]:
                    if qc_dict[data_file_name]["label"]["myocardium_nulling"] == 5:
                        qc_dict[data_file_name]["label"]["myocardium_nulling"] = 4
                if "enhancement_of_aorta_and_valves" in qc_dict[data_file_name]["label"]:
                    if qc_dict[data_file_name]["label"]["enhancement_of_aorta_and_valves"] == 5:
                        qc_dict[data_file_name]["label"]["enhancement_of_aorta_and_valves"] = 4

            labeled_data_files.append(data_file)

            # Normalize quality score from 5 to 4
            if qc_dict[data_file_name]["label"]["quality_for_fibrosis_assessment"] == 5:
                qc_dict[data_file_name]["label"]["quality_for_fibrosis_assessment"] = 4

            qc_scores.append(qc_dict[data_file_name]["label"]["quality_for_fibrosis_assessment"])

            if data_file_name not in qc_dict_scores:
                qc_dict_scores[data_file_name] = {"label": {}}

            qc_dict_scores[data_file_name]["label"]["quality_for_fibrosis_assessment"] = qc_dict[data_file_name]["label"]["quality_for_fibrosis_assessment"]

            if require_concept_labels:
                if "sharpness" in qc_dict[data_file_name]["label"]:
                    qc_dict_scores[data_file_name]["label"]["sharpness"] = qc_dict[data_file_name]["label"]["sharpness"]
                if "myocardium_nulling" in qc_dict[data_file_name]["label"]:
                    qc_dict_scores[data_file_name]["label"]["myocardium_nulling"] = qc_dict[data_file_name]["label"]["myocardium_nulling"]
                if "enhancement_of_aorta_and_valves" in qc_dict[data_file_name]["label"]:
                    qc_dict_scores[data_file_name]["label"]["enhancement_of_aorta_and_valves"] = qc_dict[data_file_name]["label"]["enhancement_of_aorta_and_valves"]
        else:
            unlabeled_data_files.append(data_file)

    print(
        "LA slice-count filter: "
        f"kept scans with la_axial_slice_count_after_spacing > {min_la_slices}; "
        f"excluded {excluded_by_la_slice_count} labeled scans."
    )
    return qc_scores, labeled_data_files, qc_dict_scores, unlabeled_data_files

def main():
    """Main function to train AFibQCAttentionMILPsuedoBagsUnsupervisedNet2D"""
    args = parse_arguments()
    CONFIG = update_config(args)

    # Set seed for reproducibility
    set_seed(CONFIG['seed'])

    g = torch.Generator()
    g.manual_seed(CONFIG['seed'])

    saved_model_name = CONFIG['saved_model_name']

    # Load data files
    data_files = glob.glob(CONFIG['data_path'] + '/*')
    data_files.sort()

    # Get labeled data (requires concept labels for this model)
    qc_scores, labeled_data_files, qc_dict_scores, unlabeled_data_files = get_qc_scores(
        data_files,
        CONFIG['qc_label_dict'],
        require_concept_labels=True,
        min_la_slices=CONFIG['min_la_slices'],
    )

    print(f"Found {len(labeled_data_files)} labeled cases with concept labels")
    print(f"Found {len(unlabeled_data_files)} unlabeled cases")

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=CONFIG.get('k_folds', 10), shuffle=True, random_state=CONFIG['seed'])

    # Store results for each fold
    fold_results = []

    # Get transforms for this model
    train_transform, val_transform = get_transforms(CONFIG)

    for fold, (train_val_indices, test_indices) in enumerate(skf.split(labeled_data_files, qc_scores)):
        print(f"\n{'='*50}")
        print(f"Starting Fold {fold + 1}/{CONFIG.get('k_folds', 10)}")
        print(f"{'='*50}")

        # Split data based on indices
        train_val_files = [labeled_data_files[i] for i in train_val_indices]
        test_files = [labeled_data_files[i] for i in test_indices]

        # Get QC scores for train_val split
        train_val_qc_scores = [qc_scores[i] for i in train_val_indices]

        # Further split train_val into train and validation
        train_files, val_files = train_test_split(
            train_val_files, test_size=0.1, random_state=CONFIG['seed'], stratify=train_val_qc_scores
        )

        # Calculate class weights for this fold
        train_file_labels = []
        for train_file in train_files:
            train_file_name = train_file.split("/")[-1]
            # Convert to integer labels (0, 1, 2, 3 for quality scores 1, 2, 3, 4)
            label = int(qc_dict_scores[train_file_name]["label"]["quality_for_fibrosis_assessment"] - 1)
            train_file_labels.append(label)

        train_file_labels = np.array(train_file_labels, dtype=int)

        class_sample_counts = np.bincount(train_file_labels)

        # Check for division by zero - raise error if any class has no samples
        if np.any(class_sample_counts == 0):
            zero_classes = np.where(class_sample_counts == 0)[0]
            raise ValueError(f"Division by zero: Classes {zero_classes} have no samples in training set. "
                           f"Class sample counts: {class_sample_counts}")

        train_class_weight = 1.0 / class_sample_counts

        samples_train_weights = np.array([train_class_weight[label] for label in train_file_labels])

        print(f"Fold {fold + 1} - Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
        print(f"Class sample counts: {class_sample_counts}")

        # Update saved model name for this fold
        if isinstance(saved_model_name, list):
            fold_saved_model_name = [name.replace('.pth', f'_fold{fold+1}.pth') for name in saved_model_name]
        else:
            fold_saved_model_name = saved_model_name.replace('.pth', f'_fold{fold+1}.pth')

        # Get patient records with concept labels
        train_patient_records = get_patient_records_monai(
            train_files, data_category='train', qc_dict_json=qc_dict_scores,
            require_concept_labels=True
        )
        val_patient_records = get_patient_records_monai(
            val_files, data_category='val', qc_dict_json=qc_dict_scores,
            require_concept_labels=True
        )
        test_patient_records = get_patient_records_monai(
            test_files, data_category='test', qc_dict_json=qc_dict_scores,
            require_concept_labels=True
        )

        # Create datasets
        AFibQCDataset_train = CacheDataset(data=train_patient_records, transform=train_transform, cache_rate=1.0, num_workers=8, copy_cache=False)
        AFibQCDataset_val = CacheDataset(data=val_patient_records, transform=val_transform, cache_rate=1.0, num_workers=4, copy_cache=False)
        AFibQCDataset_test = CacheDataset(data=test_patient_records, transform=val_transform, cache_rate=1.0, num_workers=4, copy_cache=False)
        # AFibQCDataset_train = Dataset(data=train_patient_records, transform=train_transform)
        # AFibQCDataset_val = Dataset(data=val_patient_records, transform=val_transform)
        # AFibQCDataset_test = Dataset(data=test_patient_records, transform=val_transform)

        overlay_test_transform = get_overlay_test_transform(
            CONFIG=CONFIG,
            no_edges=True,
            bagging="redistribute",
        )
        # AFibQCDataset_test_for_overlaying_patches = CacheDataset(data=test_patient_records, transform=overlay_test_transform, cache_rate=1.0, num_workers=4, copy_cache=False)
        AFibQCDataset_test_for_overlaying_patches = Dataset(data=test_patient_records, transform=overlay_test_transform)

        print(f"Datasets created for {MODEL_NAME}")

        encoder_name = CONFIG.get("tier1_encoder_name", "resnet")
        tier1_kwargs = {
            "encoder_name": encoder_name,
            "n_input_channels": 1,
            "spatial_dims": 2,
        }
        print(f"Tier-1 encoder: {encoder_name}")

        # Initialize models - Tier 1 and Tier 2 for the unsupervised pseudo-bag approach
        tier1_model = AttentionMILPseudoBagTier1Unsup(
            num_classes=4,
            concept_dim=CONFIG.get("concept_dim", 64),
            **tier1_kwargs,
        )
        tier2_model = AttentionMILPseudoBagTier2Unsup(num_classes=4, fused_dim=256)

        print("AFibQCAttentionMILPsuedoBagsUnsupervisedNet2D models created")

        # Set up early stopping
        early_stopping = EarlyStopping(
            patience=CONFIG['training_patience'],
            verbose=False,
            delta=0.0001,
            path=[
                CONFIG['model_path'] + f'/{fold_saved_model_name[0]}',
                CONFIG['model_path'] + f'/{fold_saved_model_name[1]}'
            ],
            score_name='auroc',
            start_epoch=0
        )

        fold_start_time_utc = datetime.datetime.now(datetime.timezone.utc)

        # Initialize Comet experiment
        experiment = Experiment(
            api_key=os.environ.get("COMET_API_KEY"),
            project_name="afib-quality-assessment",
            workspace="arefeen111",
            log_code=True,
            disabled=disable_comet,
        )
        experiment.set_name(f"{MODEL_NAME}_fold_{fold+1}")
        experiment.log_parameter("fold", fold+1)

        # Log code files
        experiment.log_code(file_name="model.py")
        experiment.log_code(file_name="scripts/hamilqa_unsup_trainer.py")
        experiment.log_code(file_name="util/mil_utils.py")
        experiment.log_code(file_name="util/data_utils.py")
        experiment.log_code(file_name="scripts/qc_dataset.py")
        experiment.log_code(file_name="config.py")

        # Log hyperparameters
        hyper_params = {
            "seed": CONFIG['seed'],
            "patch_size": CONFIG['patch_size'],
            "n_patches": CONFIG['n_patches'],
            "stride": CONFIG['stride'],
            "enlarge_xy": CONFIG['enlarge_xy'],
            "batch_size": CONFIG['batch_size'],
            "num_epochs": CONFIG['epochs'],
            "spacing": CONFIG['spacing'],
            "learning_rate": CONFIG['learning_rate'],
            "weight_decay": CONFIG['weight_decay'],
            "no_of_pseudo_bags": CONFIG['no_of_pseudo_bags'],
        }

        experiment.log_parameters(hyper_params)
        experiment.log_parameters(CONFIG)

        run_started_data = _build_notification_data(
            config_dict=CONFIG,
            fold=fold + 1,
            start_time_utc=fold_start_time_utc,
            fold_saved_model_name=fold_saved_model_name,
        )
        experiment.log_other("run_status", "started")
        experiment.log_other("run_start_time_utc", run_started_data["start_time_utc"])
        _safe_send_comet_notification(
            experiment=experiment,
            title=f"[HAMILQA] Started: {MODEL_NAME} fold {fold + 1} on {run_started_data['hostname']}",
            status="started",
            additional_data=run_started_data,
        )

        try:
            # Initialize trainer for the unsupervised pseudo-bag approach
            afib_trainer = AFibQCAttentionMILPsuedoBagsUnsupTrainer(
                tier_1_model=tier1_model,
                tier_2_model=tier2_model,
                train_dataset=AFibQCDataset_train,
                val_dataset=AFibQCDataset_val,
                test_dataset=AFibQCDataset_test,
                test_dataset_for_overlaying_patches=AFibQCDataset_test_for_overlaying_patches,
                batch_size=CONFIG['batch_size'],
                epochs=CONFIG['epochs'],
                lr=CONFIG['learning_rate'],
                experiment=experiment
            )
            print(f"AFibQCAttentionMILPsuedoBagsUnsupTrainer created")

            print("Training started")

            afib_trainer.train(
                early_stopping=early_stopping,
                train_weights=samples_train_weights,
                g=g,
                seed_worker=seed_worker
            )

            print(f"Loading the saved models:")
            print(f"  Tier 1: {CONFIG['model_path']}/{fold_saved_model_name[0]}")
            print(f"  Tier 2: {CONFIG['model_path']}/{fold_saved_model_name[1]}")
            test_results = afib_trainer.test(
                model_save_path=fold_saved_model_name,
                g=g,
                seed_worker=seed_worker
            )

            fold_results.append({
                'fold': fold + 1,
                'test_results': test_results
            })

            success_data = _build_notification_data(
                config_dict=CONFIG,
                fold=fold + 1,
                start_time_utc=fold_start_time_utc,
                fold_saved_model_name=fold_saved_model_name,
                test_results=test_results,
            )
            experiment.log_other("run_status", "completed")
            experiment.log_other("run_end_time_utc", success_data["event_time_utc"])
            experiment.log_other("runtime_seconds", success_data["runtime_seconds"])
            _safe_send_comet_notification(
                experiment=experiment,
                title=f"[HAMILQA] Completed: {MODEL_NAME} fold {fold + 1} on {success_data['hostname']}",
                status="completed successfully",
                additional_data=success_data,
            )
            print("Testing finished")
        except Exception as exc:
            failure_data = _build_notification_data(
                config_dict=CONFIG,
                fold=fold + 1,
                start_time_utc=fold_start_time_utc,
                fold_saved_model_name=fold_saved_model_name,
            )
            failure_data["error_type"] = type(exc).__name__
            failure_data["error_message"] = str(exc)
            failure_data["traceback"] = traceback.format_exc()
            experiment.log_other("run_status", "failed")
            experiment.log_other("failure_type", failure_data["error_type"])
            experiment.log_other("failure_message", failure_data["error_message"])
            experiment.log_other("run_end_time_utc", failure_data["event_time_utc"])
            experiment.log_other("runtime_seconds", failure_data["runtime_seconds"])
            _safe_send_comet_notification(
                experiment=experiment,
                title=f"[HAMILQA] FAILED: {MODEL_NAME} fold {fold + 1} on {failure_data['hostname']}",
                status="failed",
                additional_data=failure_data,
            )
            raise
        finally:
            experiment.end()
        print(f"Fold {fold + 1} completed")

        # break  # Remove this break statement to run all folds

    # Calculate and log average results across all folds
    if fold_results:
        print(f"\n{'='*50}")
        print("K-Fold Cross-Validation Results Summary")
        print(f"{'='*50}")

        # Calculate average metrics
        avg_metrics = {}
        for key in fold_results[0]['test_results'].keys():
            if isinstance(fold_results[0]['test_results'][key], (int, float)):
                avg_metrics[f'avg_{key}'] = np.mean([result['test_results'][key] for result in fold_results])
                avg_metrics[f'std_{key}'] = np.std([result['test_results'][key] for result in fold_results])

        print("Average Results:", avg_metrics)

        # Log summary to a final experiment
        summary_experiment = Experiment(
            api_key=os.environ.get("COMET_API_KEY"),
            project_name="afib-quality-assessment",
            workspace="arefeen111",
            disabled=disable_comet,
        )
        summary_experiment.set_name(f"{MODEL_NAME}_kfold_summary")
        summary_experiment.log_parameters(avg_metrics)
        summary_experiment.end()

if __name__ == '__main__':
    main()
