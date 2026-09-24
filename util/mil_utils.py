from monai.transforms import Compose, Spacingd, ToTensord
import torch

from util.data_utils import (
    LoadNrrd, LoadNrrdWithSpacing, LoadAxialViewLA, ComputeEdgesForAxialView,
    LoadRandom2DPatchesPseudoBags, Load2DPatchesPseudoBags, Load2DPatchesPseudoBagsOverlay,
)
from util.volume_preprocessing import PrepareVolume


def get_overlay_test_transform(CONFIG, no_edges=False, bagging="redistribute"):
    """
    Build test-time transform for overlay visualization.

    Output includes:
    - pseudo_bags
    - overlay_bag_meta
    - overlay_axial_base
    """
    t = [
        LoadNrrd(keys=["image", "la_label"]),
        Spacingd(
            keys=["image", "la_label"],
            pixdim=(CONFIG['spacing'][0], CONFIG['spacing'][1], CONFIG['spacing'][2]),
            mode=("trilinear", "nearest"),
        ),
        PrepareVolume(image_key="image", label_key="la_label"),
        LoadAxialViewLA(keys=["image", "la_label"]),
    ]

    if not no_edges:
        t.append(ComputeEdgesForAxialView(keys=["axial_image"], add_as_channel=True))

    t.extend([
        Load2DPatchesPseudoBagsOverlay(
            keys=["axial_image", "la_label"],
            stride=CONFIG["stride"],
            patch_size=CONFIG["patch_size"],
            no_of_pseudo_bags=CONFIG["no_of_pseudo_bags"],
            bagging=bagging,
        ),
    ])

    return Compose(t)


def get_transforms(CONFIG):
    # Normalize the full volume before axial slice extraction. This matches
    # AnatomyAwareACMIL and preserves intensity relationships between slices;
    # per-slice normalization would otherwise remove that across-volume context.
    volume_load_transforms = [LoadNrrdWithSpacing(keys=["image", "la_label"])]

    train_transform = Compose(
        [
            *volume_load_transforms,
            Spacingd(
                keys=["image", "la_label"],
                pixdim=(CONFIG['spacing'][0], CONFIG['spacing'][1], CONFIG['spacing'][2]),
                mode=("trilinear", "nearest"),
            ),
            PrepareVolume(image_key="image", label_key="la_label"),
            LoadAxialViewLA(keys=["image", "la_label"]),
            LoadRandom2DPatchesPseudoBags(
                keys=["axial_image", "la_label"],
                n_patches=CONFIG['n_patches'],
                patch_size=CONFIG['patch_size'],
                enlarge_xy=CONFIG['enlarge_xy'],
                no_of_pseudo_bags=CONFIG['no_of_pseudo_bags'],
                return_positions=True,
            ),
            ToTensord(keys=["labels"], dtype=torch.float32)
        ]
    )

    val_transform = Compose(
        [
            *volume_load_transforms,
            Spacingd(
                keys=["image", "la_label"],
                pixdim=(CONFIG['spacing'][0], CONFIG['spacing'][1], CONFIG['spacing'][2]),
                mode=("trilinear", "nearest"),
            ),
            PrepareVolume(image_key="image", label_key="la_label"),
            LoadAxialViewLA(keys=["image", "la_label"]),
            Load2DPatchesPseudoBags(
                keys=["axial_image", "la_label"],
                stride=CONFIG['stride'],
                patch_size=CONFIG['patch_size'],
                return_positions=True,
            ),
            ToTensord(keys=["labels"], dtype=torch.float32)
        ]
    )

    return train_transform, val_transform
