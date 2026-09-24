import torch

gpu_id = 0
machine = 'ebe'
seed = 0

n_patches = 80
enlarge_xy = 100
n_epochs = 350
no_of_pseudo_bags = 8  # 18 (now 9!!) is the lowest number of Axial slices in the dataset
patch_size_3d = (64, 64, 7)  # (64, 64, 7), (8, 8, 3)
patch_size_2d = (patch_size_3d[0], patch_size_3d[1])  # (64, 64), (8, 8)

def calculate_stride(patch_size, overlap_percentage):
    return tuple(int(size * (1 - overlap_percentage / 100)) for size in patch_size)

overlap_percentage = 50
stride_2d = calculate_stride(patch_size_2d, overlap_percentage)

disable_comet = False

# Edit this to point at your local copy of the dataset (not distributed with
# this repo). See README.md > Data preparation for the expected layout.
_DATASET_ROOT = "/path/to/your/dataset"

# Model configuration dict.
# Only AFibQCAttentionMILPsuedoBagsUnsupervisedNet2D (the unsupervised+adversarial
# pseudo-bag variant) is run by run_hamilqa_unsup.py; other model families from the
# source repo were intentionally left out of this copy.
config = {
    "AFibQCAttentionMILPsuedoBagsUnsupervisedNet2D": {
        'patch_size': patch_size_2d,
        'stride': stride_2d,
        'enlarge_xy': enlarge_xy,
        'n_patches': n_patches,
        'tier1_encoder_name': 'resnet',
        'tier1_lr_mult': 0.8,
        'batch_size': 32,
        'lambda_concepts': 1.0,
        'lambda_adv': 0.5,
        'lambda_div': 0.1,
        'use_amp': False,
        'concept_dim': 64,
        'classification': 'multiclass',
        'no_of_pseudo_bags': no_of_pseudo_bags,
        'epochs': n_epochs,
        # Keep a scan only when its stored post-Spacingd LA slice count is
        # strictly greater than this value. Override with --min_la_slices.
        'min_la_slices': 8,
        'training_patience': 100,
        'seed': seed,
        'learning_rate': 1e-4,
        'spacing': [0.625, 0.625, 2.5],
        'weight_decay': 1e-3,
        'data_path': f'{_DATASET_ROOT}/afib_db',
        'qc_label_dict': f'{_DATASET_ROOT}/new_surface_area_with_ratings.json',
        'model_path': 'model/saved_models',
    },
}


device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

MODEL_NAME = 'AFibQCAttentionMILPsuedoBagsUnsupervisedNet2D'
CONFIG = config[MODEL_NAME]

CONFIG['tier1_saved_model_name'] = f'afib_qc_attn_pseudo_bags_tier1_unsupervised_adversarial_2d_{no_of_pseudo_bags}_{n_patches}_div_{CONFIG["lambda_div"]:2f}_adv_{CONFIG["lambda_adv"]:2f}_{gpu_id}_{machine}.pth'
CONFIG['tier2_saved_model_name'] = f'afib_qc_attn_pseudo_bags_tier2_unsupervised_adversarial_2d_{no_of_pseudo_bags}_{n_patches}_div_{CONFIG["lambda_div"]:2f}_adv_{CONFIG["lambda_adv"]:2f}_{gpu_id}_{machine}.pth'
CONFIG['saved_model_name'] = [CONFIG['tier1_saved_model_name'], CONFIG['tier2_saved_model_name']]
