import cv2
import numpy as np
import nrrd

import torch
from typing import Dict, List
from monai.transforms.transform import MapTransform
from monai.transforms import RandomizableTransform
from collections.abc import Hashable, Mapping

from monai.config import KeysCollection
from config import CONFIG

def load_nrrd_data(file_path):
    """Load the NRRD file and return the data array."""
    data, _ = nrrd.read(file_path)
    data = np.expand_dims(data, axis=0)  # Add channel dimension

    return data

def find_bounding_box_2d(label_data):
    """Find the bounding box for the blood pool regions."""
    if torch.is_tensor(label_data):
        label_data = label_data.detach().cpu().numpy()
    else:
        label_data = np.asarray(label_data)

    while label_data.ndim > 2 and label_data.shape[0] == 1:
        label_data = label_data[0]

    if label_data.ndim != 2:
        raise ValueError(f"find_bounding_box_2d expects 2D label data, got shape {label_data.shape}")
    if not np.any(label_data == 1):
        raise ValueError("find_bounding_box_2d received label data with no foreground")

    xs, ys = np.where(label_data == 1)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return x_min, x_max, y_min, y_max

def get_sampling_bbox_2d(mri_data, label_data, enlarge_xy):
    """Return an enlarged LA bbox, or the full slice when no LA is present."""
    mri_shape = mri_data.shape
    if len(mri_shape) == 3:
        h, w = mri_shape[1], mri_shape[2]
    elif len(mri_shape) == 2:
        h, w = mri_shape[0], mri_shape[1]
    else:
        raise ValueError(f"2D patch extraction expects 2D or channel-first 3D image data, got shape {mri_shape}")

    if torch.is_tensor(label_data):
        label_arr = label_data.detach().cpu().numpy()
    else:
        label_arr = np.asarray(label_data)

    while label_arr.ndim > 2 and label_arr.shape[0] == 1:
        label_arr = label_arr[0]

    if np.any(label_arr == 1):
        x_min, x_max, y_min, y_max = find_bounding_box_2d(label_arr)
        x_min = max(0, x_min - enlarge_xy) # up
        x_max = min(h - 1, x_max + enlarge_xy // 2) # down
        y_min = max(0, y_min - enlarge_xy // 3) # left
        y_max = min(w - 1, y_max + enlarge_xy // 2) # right
    else:
        x_min, x_max = 0, h - 1
        y_min, y_max = 0, w - 1

    return x_min, x_max, y_min, y_max

def crop_patches_2d(volume, patch_size, stride):
    """Crop overlapping patches from the volume."""
    patches = []

    patches_position = []

    if len(volume.shape) == 2:
        h, w = volume.shape
    else:
        ch, h, w = volume.shape

    for x in range(0, h - patch_size[0] + 1, stride[0]):  # 0, 64, 32
        for y in range(0, w - patch_size[1] + 1, stride[1]):
            if len(volume.shape) == 3:
                patch = volume[:, x:x + patch_size[0], y:y + patch_size[1]]
            else:
                patch = volume[x:x + patch_size[0], y:y + patch_size[1]]
                patch = np.expand_dims(patch, axis=0) # Add channel dimension
        
            patches_position.append((x, y)) # (n_patches, 2)
            patches.append(patch)

    if len(patches) == 0:
        raise ValueError("No patches were generated")

    patches_position = np.stack(patches_position, axis=0) # (n_patches, 2)

    patches = np.stack(patches, axis=0)  # (n_patches, 64, 64, 7)

    return patches, patches_position

def crop_random_2d_patches(volume, patch_size, n_patches, random_state=np.random):
    """Crop random patches from the volume."""
    patches = []
    patches_position = []
    
    if len(volume.shape) == 2:
        assert volume.shape[0] >= patch_size[0], f"Patch size cannot be larger than the volume size, volume shape: {volume.shape}, patch size: {patch_size}"
        assert volume.shape[1] >= patch_size[1], f"Patch size cannot be larger than the volume size, volume shape: {volume.shape}, patch size: {patch_size}"
    else:
        assert volume.shape[1] >= patch_size[0], f"Patch size cannot be larger than the volume size, volume shape: {volume.shape}, patch size: {patch_size}"
        assert volume.shape[2] >= patch_size[1], f"Patch size cannot be larger than the volume size, volume shape: {volume.shape}, patch size: {patch_size}"

    for _ in range(n_patches):

        if len(volume.shape) == 2:
            # volume.shape[0] = 256, patch_size[0] = 64 => 0, 192
            x = random_state.randint(0, volume.shape[0] - patch_size[0] + 1)
            # volume.shape[1] = 256, patch_size[1] = 64 => 0, 192
            y = random_state.randint(0, volume.shape[1] - patch_size[1] + 1)

            patch = volume[x:x + patch_size[0], y:y + patch_size[1]]
            patch = np.expand_dims(patch, axis=0)  # Add channel dimension

        else:
            x = random_state.randint(0, volume.shape[1] - patch_size[0] + 1)
            y = random_state.randint(0, volume.shape[2] - patch_size[1] + 1)

            patch = volume[:, x:x + patch_size[0], y:y + patch_size[1]]

        patches.append(patch)
        patches_position.append((x, y))

    if len(patches) == 0:
        raise ValueError("No patches were generated")

    patches_position = np.stack(patches_position, axis=0)  # (n_patches, 2)
    patches = np.stack(patches, axis=0)  # (n_patches, 64, 64, 7)

    return patches, patches_position

def extract_and_crop_patches_2d(
    mri_data,
    label_data,
    patch_size=(64, 64),
    stride=(32, 32),
    enlarge_xy=20,
    return_positions=False,
):
    squeezed_mri_data = mri_data
    x_min, x_max, y_min, y_max = get_sampling_bbox_2d(mri_data, label_data, enlarge_xy)

    # 3. Extract ROI (Sub-Volume)
    if len(mri_data.shape) == 3:
        sub_mri_data = squeezed_mri_data[:, x_min:x_max +1, y_min:y_max+1] 
        crop_h, crop_w = sub_mri_data.shape[1], sub_mri_data.shape[2]
    else:
        sub_mri_data = squeezed_mri_data[x_min:x_max +1, y_min:y_max+1]
        crop_h, crop_w = sub_mri_data.shape[0], sub_mri_data.shape[1]

    # 4. Crop Patches (Sliding Window)
    # patches_position is relative to Top-Left of sub_mri_data (0,0)
    patches, patches_position = crop_patches_2d(sub_mri_data, patch_size, stride=stride)

    if not return_positions:
        return patches, sub_mri_data, patches_position

    # 5. ROI-Relative Normalization Logic
    
    # A. Shift to Center of Patch (Critical)
    patch_center_offset = np.array([patch_size[0] // 2, patch_size[1] // 2])
    centers_local = patches_position + patch_center_offset

    # B. Normalize to [0, 1] relative to the CROP size (not full image)
    roi_dims = np.array([max(crop_h, 1), max(crop_w, 1)], dtype=np.float32)
    positions_norm_01 = centers_local / roi_dims

    # C. Scale to [-1, 1] for Neural Network
    # (-1,-1)=Top-Left of ROI, (0,0)=Center of Heart
    positions_norm_signed = (positions_norm_01 * 2) - 1

    return patches, sub_mri_data, patches_position, centers_local, positions_norm_signed

def extract_and_crop_random_2d_patches(
    mri_data,
    label_data,
    patch_size=(64, 64),
    enlarge_xy=20,
    n_patches=60,
    random_state=np.random,
    return_positions=False,
):
    squeezed_mri_data = mri_data
    x_min, x_max, y_min, y_max = get_sampling_bbox_2d(mri_data, label_data, enlarge_xy)

    # 3. Extract ROI (Sub-Volume)
    if len(mri_data.shape) == 3:
        sub_mri_data = squeezed_mri_data[:, x_min:x_max +1, y_min:y_max+1] 
        crop_h, crop_w = sub_mri_data.shape[1], sub_mri_data.shape[2]
    else:
        sub_mri_data = squeezed_mri_data[x_min:x_max +1, y_min:y_max+1] 
        crop_h, crop_w = sub_mri_data.shape[0], sub_mri_data.shape[1]

    # 4. Crop Random Patches
    # patches_topleft is relative to Top-Left of sub_mri_data (0,0)
    patches, patches_topleft = crop_random_2d_patches(sub_mri_data, patch_size, n_patches, random_state=random_state)

    if not return_positions:
        return patches, sub_mri_data

    # 5. ROI-Relative Normalization Logic
    
    # A. Shift to Center of Patch (Critical)
    patch_center_offset = np.array([patch_size[0] // 2, patch_size[1] // 2])
    centers_local = patches_topleft + patch_center_offset
    
    # B. Normalize to [0, 1] relative to the CROP size
    roi_dims = np.array([max(crop_h, 1), max(crop_w, 1)], dtype=np.float32)
    positions_norm_01 = centers_local / roi_dims
    
    # C. Scale to [-1, 1]
    # (-1,-1)=Top-Left of ROI, (0,0)=Center of Heart
    positions_norm_signed = (positions_norm_01 * 2) - 1

    return patches, sub_mri_data, patches_topleft, centers_local, positions_norm_signed

class LoadNrrd(MapTransform):

    def __init__(
        self, keys: KeysCollection, strict_check: bool = True, allow_missing_keys: bool = False, channel_dim=None
    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:

        d = dict(data)

        for keys in self.keys:
            d[keys] = load_nrrd_data(d[keys])

        return d


class LoadNrrdWithSpacing(MapTransform):
    """
    Like LoadNrrd, but reads the NRRD header's native voxel spacing and
    attaches it as MetaTensor affine metadata, so that a downstream Spacingd
    resamples relative to the file's actual native spacing instead of
    assuming an identity (1mm isotropic) affine.

    Uses pynrrd (nrrd.read), not MONAI's ITKReader: this dataset's NRRD files
    store spacing under the simple "spacings" header field rather than a full
    "space directions" matrix, which MONAI's NrrdReader/ITKReader fallback
    chain cannot parse (raises KeyError: 'space directions'), and ITKReader
    has shown intermittent failures under CacheDataset's multi-threaded
    loading that silently fall back to that broken path.
    """

    def __init__(
        self, keys: KeysCollection, allow_missing_keys: bool = False
    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        from monai.data import MetaTensor

        d = dict(data)

        for key in self.key_iterator(d):
            path = d[key]
            array, header = nrrd.read(path)
            array = np.expand_dims(array, axis=0)  # channel-first: (1, H, W, D)

            spacing = header.get("spacings")
            if spacing is None:
                direction = header.get("space directions")
                if direction is None:
                    raise ValueError(f"No spacing metadata ('spacings' or 'space directions') found in {path}")
                spacing = np.linalg.norm(np.asarray(direction, dtype=float), axis=1)
            spacing = np.asarray(spacing, dtype=float)

            affine = np.eye(4)
            for i in range(min(3, len(spacing))):
                affine[i, i] = float(spacing[i])

            d[key] = MetaTensor(
                torch.as_tensor(array.astype(np.float32)),
                affine=torch.as_tensor(affine, dtype=torch.float64),
            )

        return d

class Load2DPatchesPseudoBags(MapTransform):
    def __init__(
        self, keys: KeysCollection, strict_check: bool = True, allow_missing_keys: bool = False, channel_dim=None, stride=None, patch_size=(64, 64), 
        return_positions=False
    ) -> None:
        super().__init__(keys, allow_missing_keys)

        self.stride = stride
        self.patch_size = patch_size
        self.return_positions = return_positions

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        
        d = dict(data)
        
        axial_images = d['axial_image'] # shape: (30, 256, 190), (slices, height, width)
        axial_labels = d['la_label'] # shape: (30, 256, 190), (slices, height, width)

        # original_axial_images = d['original_axial_images'] # shape: (30, 256, 190), (slices, height, width)
        
        pseudo_bags_with_original, pseudo_bags, pseud_bags_original_sub_mri_data = [], [], []
        original_patches_position_list = []

        for i in range(axial_images.shape[0]):
            # original_patches, original_sub_mri_data, original_patches_position = extract_and_crop_patches_2d(mri_data=original_axial_images[i], label_data=axial_labels[i],
            #                                     patch_size=self.patch_size, stride=self.stride, enlarge_xy=CONFIG['enlarge_xy'])
            patch, _, _, _, patch_positions_norm = extract_and_crop_patches_2d(
                mri_data=axial_images[i],
                label_data=axial_labels[i],
                patch_size=self.patch_size,
                stride=self.stride,
                enlarge_xy=CONFIG['enlarge_xy'],
                return_positions=self.return_positions,
            )
            # patches.extend(patch)

            pseudo_bags.append(torch.tensor(patch, dtype=torch.float32))
            original_patches_position_list.append(torch.tensor(patch_positions_norm, dtype=torch.float32))
            # pseudo_bags_with_original.append(torch.tensor(original_patches, dtype=torch.float32))
            # pseud_bags_original_sub_mri_data.append(torch.tensor(original_sub_mri_data, dtype=torch.float32))
            # original_patches_position_list.append(torch.tensor(original_patches_position, dtype=torch.float32))

        # Combine all patches into a single list
        all_patches = [patch for bag in pseudo_bags for patch in bag]
        # all_patches_with_original = [patch for bag in pseudo_bags_with_original for patch in bag]
        # all_sub_mri_data = [data for bag in pseud_bags_original_sub_mri_data for data in bag]
        all_patches_positions = [pos for bag in original_patches_position_list for pos in bag]

        # Calculate the number of patches per pseudo-bag
        n_patches = len(all_patches)
        patches_per_bag = n_patches // CONFIG['no_of_pseudo_bags']
        remainder = n_patches % CONFIG['no_of_pseudo_bags']

        # Distribute patches into pseudo-bags
        pseudo_bags = []
        # pseudo_bags_with_original = []
        # pseud_bags_original_sub_mri_data = []
        original_patches_position_list = []

        start = 0
        for i in range(CONFIG['no_of_pseudo_bags']):
            end = start + patches_per_bag
            if i < remainder:
                end += 1

            pseudo_bags.append(torch.stack(all_patches[start:end]))
            # pseudo_bags_with_original.append(torch.stack(all_patches_with_original[start:end]))
            # pseud_bags_original_sub_mri_data.append(torch.stack(all_sub_mri_data[start:end]))
            original_patches_position_list.append(torch.stack(all_patches_positions[start:end]))

            start = end

        d['pseudo_bags'] = pseudo_bags
        d['pseudo_bags_coords'] = original_patches_position_list

        del d['la_label']
        del d['axial_image']

        return d

class Load2DPatchesPseudoBagsOverlay(MapTransform):
    """
    Build deterministic pseudo-bags for test-time overlays and keep patch metadata.

    Output fields:
    - pseudo_bags: list[Tensor], each Tensor shape (n_patches_bag, C, H, W)
    - overlay_bag_meta: list[list[dict]], same structure as pseudo_bags with:
        {"slice_idx": int, "pos": [x, y], "sub_img_bbox": [x_min, x_max, y_min, y_max]}
    - overlay_axial_base: (N, H, W) numpy array (base grayscale slice data)
    """

    def __init__(
        self,
        keys: KeysCollection,
        strict_check: bool = True,
        allow_missing_keys: bool = False,
        channel_dim=None,
        stride=None,
        patch_size=(64, 64),
        no_of_pseudo_bags=5,
        bagging: str = "redistribute",
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.stride = stride
        self.patch_size = patch_size
        self.no_of_pseudo_bags = no_of_pseudo_bags
        self.bagging = bagging

    @staticmethod
    def _extract_patches_with_positions(slice_img, slice_label, patch_size, stride, enlarge_xy):
        x_min, x_max, y_min, y_max = find_bounding_box_2d(slice_label)

        if slice_img.ndim == 3:
            _, h, w = slice_img.shape
        else:
            h, w = slice_img.shape

        x_min = max(0, x_min - enlarge_xy)
        x_max = min(h - 1, x_max + enlarge_xy // 2)
        y_min = max(0, y_min - enlarge_xy // 3)
        y_max = min(w - 1, y_max + enlarge_xy // 2)

        if slice_img.ndim == 3:
            sub_img = slice_img[:, x_min:x_max + 1, y_min:y_max + 1]
        else:
            sub_img = slice_img[x_min:x_max + 1, y_min:y_max + 1]

        patches, positions = crop_patches_2d(sub_img, patch_size, stride=stride)
        positions_abs = positions + np.array([x_min, y_min])

        return patches, positions_abs, (x_min, x_max, y_min, y_max)

    def _build_per_slice_bags(self, axial_images, axial_labels):
        bags: List[torch.Tensor] = []
        meta: List[list] = []

        for slice_idx in range(axial_images.shape[0]):
            slice_img = axial_images[slice_idx]
            slice_label = axial_labels[slice_idx]

            if slice_label.sum() == 0:
                continue

            patches, positions, sub_img_bbox = self._extract_patches_with_positions(
                slice_img=slice_img,
                slice_label=slice_label,
                patch_size=self.patch_size,
                stride=self.stride,
                enlarge_xy=CONFIG["enlarge_xy"],
            )

            bag_meta = [
                {
                    "slice_idx": int(slice_idx),
                    "pos": [int(positions[i][0]), int(positions[i][1])],
                    "sub_img_bbox": [
                        int(sub_img_bbox[0]),
                        int(sub_img_bbox[1]),
                        int(sub_img_bbox[2]),
                        int(sub_img_bbox[3]),
                    ],
                }
                for i in range(len(patches))
            ]

            bags.append(torch.tensor(np.stack(patches), dtype=torch.float32))
            meta.append(bag_meta)

        if not bags:
            raise ValueError("No per-slice bags were created. Check label coverage.")

        min_len = min(bag.shape[0] for bag in bags)
        if min_len == 0:
            raise ValueError("At least one bag has zero patches after extraction.")

        if any(bag.shape[0] != min_len for bag in bags):
            bags = [bag[:min_len] for bag in bags]
            meta = [bag_meta[:min_len] for bag_meta in meta]

        return bags, meta

    def _build_redistributed_bags(self, axial_images, axial_labels):
        all_patches = []
        all_positions = []
        all_sub_img_bbox = []
        all_slice_idx = []

        for slice_idx in range(axial_images.shape[0]):
            slice_img = axial_images[slice_idx]
            slice_label = axial_labels[slice_idx]

            if slice_label.sum() == 0:
                continue

            patches, positions, sub_img_bbox = self._extract_patches_with_positions(
                slice_img=slice_img,
                slice_label=slice_label,
                patch_size=self.patch_size,
                stride=self.stride,
                enlarge_xy=CONFIG["enlarge_xy"],
            )

            all_patches.extend(list(patches))
            all_positions.extend(list(positions))
            all_sub_img_bbox.extend([sub_img_bbox] * len(patches))
            all_slice_idx.extend([slice_idx] * len(patches))

        if len(all_patches) == 0:
            raise ValueError("No patches extracted. Check label coverage.")

        n_patches = len(all_patches)
        if n_patches < self.no_of_pseudo_bags:
            raise ValueError(
                f"Number of extracted patches ({n_patches}) is less than no_of_pseudo_bags ({self.no_of_pseudo_bags})."
            )
        patches_per_bag = n_patches // self.no_of_pseudo_bags
        remainder = n_patches % self.no_of_pseudo_bags

        bags = []
        meta = []
        start = 0

        for i in range(self.no_of_pseudo_bags):
            end = start + patches_per_bag
            if i < remainder:
                end += 1

            bag_patches = [all_patches[j] for j in range(start, end)]
            bag_meta = [
                {
                    "slice_idx": int(all_slice_idx[j]),
                    "pos": [int(all_positions[j][0]), int(all_positions[j][1])],
                    "sub_img_bbox": [
                        int(all_sub_img_bbox[j][0]),
                        int(all_sub_img_bbox[j][1]),
                        int(all_sub_img_bbox[j][2]),
                        int(all_sub_img_bbox[j][3]),
                    ],
                }
                for j in range(start, end)
            ]

            bags.append(torch.tensor(np.stack(bag_patches), dtype=torch.float32))
            meta.append(bag_meta)
            start = end

        min_len = min(bag.shape[0] for bag in bags)
        if min_len == 0:
            raise ValueError("At least one bag has zero patches after extraction.")

        if any(bag.shape[0] != min_len for bag in bags):
            bags = [bag[:min_len] for bag in bags]
            meta = [bag_meta[:min_len] for bag_meta in meta]

        return bags, meta

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)

        axial_images = d["axial_image"]  # (N, C, H, W)
        axial_labels = d["la_label"]  # (N, H, W)

        # Keep base grayscale for overlays before axial image is removed.
        if isinstance(axial_images, torch.Tensor):
            axial_np = axial_images.detach().cpu().numpy()
        else:
            axial_np = axial_images
        if isinstance(axial_labels, torch.Tensor):
            label_np = axial_labels.detach().cpu().numpy()
        else:
            label_np = axial_labels
        d["overlay_axial_base"] = axial_np[:, 0].copy()

        if self.bagging == "per-slice":
            bags, meta = self._build_per_slice_bags(axial_np, label_np)
        else:
            bags, meta = self._build_redistributed_bags(axial_np, label_np)

        d["pseudo_bags"] = bags
        d["overlay_bag_meta"] = meta

        del d["la_label"]
        del d["axial_image"]

        return d

class LoadAxialViewLA(MapTransform):
    def __init__(
        self, keys: KeysCollection, strict_check: bool = True, allow_missing_keys: bool = False, channel_dim=None,
        require_label_foreground: bool = True,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.require_label_foreground = require_label_foreground

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        
        d = dict(data)
        
        mri_data = d['image'] # (1, 254, 190, 36)
        label_data = d['la_label']
        
        axial_images, axial_labels = [], []

        for i in range(mri_data.shape[3]):
            has_label = label_data[0, :, :, i].sum() > 0
            if self.require_label_foreground and not has_label:
                continue

            axial_images.append(mri_data[0, :, :, i].transpose(1, 0))
            axial_labels.append(label_data[0, :, :, i].transpose(1, 0))

        if len(axial_images) == 0:
            raise ValueError("No valid axial slices found with non-zero labels")

        axial_images = np.stack(axial_images, axis=0) # (N, 256, 190)
        axial_labels = np.stack(axial_labels, axis=0) # (N, 256, 190)

        axial_images = np.expand_dims(axial_images, axis=1)  # Add channel dimension: (N, 1, 256, 190)
        
        d['axial_image'] = axial_images# (N, 1, 256, 190)
        d['la_label'] = axial_labels # (N, 256, 190)

        del d['image']
        
        return d

class ComputeEdgesForAxialView(MapTransform):
    def __init__(
        self, 
        keys: KeysCollection, 
        strict_check: bool = True, 
        allow_missing_keys: bool = False,
        low_threshold: int = 100,
        high_threshold: int = 200,
        add_as_channel: bool = True
    ) -> None:
        """
        Compute Canny edge detection for axial view images from LoadAxialViewLA output.
        
        Args:
            keys: Keys to process
            low_threshold: Lower threshold for Canny edge detection
            high_threshold: Upper threshold for Canny edge detection
            add_as_channel: If True, concatenate edges as additional channel to images.
                          If False, store edges in separate key 'axial_edges'
        """
        super().__init__(keys, allow_missing_keys)
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.add_as_channel = add_as_channel

    def _compute_canny_edges(self, img2d):
        """
        Compute Canny edge detection for a single slice.
        
        Args:
            img2d: (H, W) image slice
            
        Returns:
            edge_image: (H, W) Canny edge map
        """
        # Convert to numpy if needed
        if isinstance(img2d, torch.Tensor):
            img = img2d.cpu().numpy()
        else:
            img = img2d
        
        # Normalize image to 0-255 range
        normalized_image = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        
        # Apply Canny edge detection
        edge_image = cv2.Canny(normalized_image.astype(np.uint8), self.low_threshold, self.high_threshold)
        
        return edge_image.astype(np.float32)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        
        # Get axial images and labels from LoadAxialViewLA output
        if 'axial_image' not in d:
            raise KeyError("'axial_image' key not found. This transform expects output from LoadAxialViewLA.")
        
        axial_images = d['axial_image'].squeeze(1)  # Shape: (N, H, W)
        
        # Handle case where no slices are available
        if axial_images.shape[0] == 0:
            print("Warning: No axial slices available. Skipping edge computation.")
            return d
        
        # List to store edge maps
        edge_maps = []
        
        # Process each slice
        for i in range(axial_images.shape[0]):
            img_slice = axial_images[i]  # (H, W)
            edge_map = self._compute_canny_edges(img_slice)
            edge_maps.append(edge_map)
        
        # Stack edge maps
        edge_maps_array = np.stack(edge_maps, axis=0)  # (N, H, W)
        
        if self.add_as_channel:
            # Add edges as an additional channel to axial_images
            # axial_images: (N, H, W) -> (N, 1, H, W)
            # edge_maps: (N, H, W) -> (N, 1, H, W)
            # Result: (N, 2, H, W)
            images_with_channel = np.expand_dims(axial_images, axis=1)  # (N, 1, H, W)
            edges_with_channel = np.expand_dims(edge_maps_array, axis=1)  # (N, 1, H, W)
            combined = np.concatenate([images_with_channel, edges_with_channel], axis=1)  # (N, 2, H, W)
            d['axial_image'] = combined
        else:
            # Store edges separately
            d['axial_edges'] = edge_maps_array
        
        return d

class LoadRandom2DPatchesPseudoBags(RandomizableTransform, MapTransform):
    def __init__(
        self, 
        keys: KeysCollection, 
        strict_check: bool = True, 
        allow_missing_keys: bool = False, 
        channel_dim=None, 
        n_patches=60, 
        patch_size=(64, 64), 
        enlarge_xy=20,
        no_of_pseudo_bags=10,
        return_positions=False,
        prob: float = 1.0, # Added prob to control execution probability (usually 1.0 for this type of transform)
    ) -> None:
        # Initialize both parent classes
        RandomizableTransform.__init__(self, prob=prob)
        MapTransform.__init__(self, keys, allow_missing_keys)

        self.n_patches = n_patches
        self.patch_size = patch_size
        self.enlarge_xy = enlarge_xy
        self.no_of_pseudo_bags = no_of_pseudo_bags
        self.return_positions = return_positions
        
        # Placeholders for random parameters
        self.bag_indices = None
        self.remainder_indices = None

    def randomize(self, data: Mapping[Hashable, torch.Tensor]) -> None:
        """
        Calculate random parameters here using self.R (MONAI's random state).
        This method is called automatically by MONAI before __call__.
        """
        d = dict(data)
        axial_images = d['axial_image']
        num_slices = axial_images.shape[0]

        # Use self.R instead of np.random
        self.bag_indices = self.R.choice(
            a=num_slices, 
            size=self.no_of_pseudo_bags, 
            replace=False
        )
        
        remainder = self.n_patches % self.no_of_pseudo_bags
        if remainder > 0:
            self.remainder_indices = self.R.choice(
                a=self.no_of_pseudo_bags, 
                size=remainder, 
                replace=False
            )
        else:
            self.remainder_indices = None

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        
        # If randomize hasn't been called (e.g. used standalone outside a Compose), call it.
        if self.bag_indices is None:
            self.randomize(data)

        axial_images = d['axial_image']
        axial_labels = d['la_label']

        pseudo_bags = []
        pseudo_bags_coords = []
        no_of_patches = 0

        assert axial_images.shape[0] >= self.no_of_pseudo_bags, \
            f"Number of slices {axial_images.shape[0]} < pseudo-bags {self.no_of_pseudo_bags}"

        # Use the pre-calculated random indices
        pseudo_bags_slices = axial_images[self.bag_indices] 
        pseudo_bags_slices_labels = axial_labels[self.bag_indices]

        n_instances_per_bag = self.n_patches // self.no_of_pseudo_bags
        
        # 1. Fill main bags
        for i in range(self.no_of_pseudo_bags):
            patches_per_bag, _, _, _, patches_positions_norm = extract_and_crop_random_2d_patches(
                mri_data=pseudo_bags_slices[i], 
                label_data=pseudo_bags_slices_labels[i],
                patch_size=self.patch_size, 
                enlarge_xy=self.enlarge_xy,
                n_patches=n_instances_per_bag,
                random_state=self.R,  # <--- CRITICAL CHANGE
                return_positions=self.return_positions,
            )
            no_of_patches += patches_per_bag.shape[0]
            pseudo_bags.append(patches_per_bag) 
            pseudo_bags_coords.append(patches_positions_norm)

        # 2. Handle remainder using pre-calculated remainder indices
        if self.remainder_indices is not None:
            pseudo_bags_slices_remainder = pseudo_bags_slices[self.remainder_indices]
            pseudo_bags_slices_labels_remainder = pseudo_bags_slices_labels[self.remainder_indices]

            for i in range(len(self.remainder_indices)):
                patch_per_bag, _, _, _, patch_positions_norm = extract_and_crop_random_2d_patches(
                    mri_data=pseudo_bags_slices_remainder[i], 
                    label_data=pseudo_bags_slices_labels_remainder[i],
                    patch_size=self.patch_size, 
                    enlarge_xy=self.enlarge_xy,
                    n_patches=1,
                    random_state=self.R,  # <--- CRITICAL CHANGE
                    return_positions=self.return_positions,
                )
                no_of_patches += patch_per_bag.shape[0]
                
                # Append to the specific bag selected by random_indices_remainder
                target_bag_idx = self.remainder_indices[i]
                pseudo_bags[target_bag_idx] = np.concatenate(
                    (pseudo_bags[target_bag_idx], patch_per_bag), axis=0
                )
                pseudo_bags_coords[target_bag_idx] = np.concatenate(
                    (pseudo_bags_coords[target_bag_idx], patch_positions_norm), axis=0
                )

        assert no_of_patches == self.n_patches, \
            f"Number of patches is {no_of_patches} instead of {self.n_patches}"

        d['pseudo_bags'] = pseudo_bags 
        d['pseudo_bags_coords'] = [torch.tensor(coords, dtype=torch.float32) for coords in pseudo_bags_coords]

        del d['axial_image']
        del d['la_label']
        
        # Reset indices to ensure fresh randomness next time if object is reused
        self.bag_indices = None
        self.remainder_indices = None

        return d
