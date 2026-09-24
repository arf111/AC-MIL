import torch
from monai.transforms import Compose, NormalizeIntensityd
from monai.transforms.transform import MapTransform


class PrepareVolume(MapTransform):
    """
    Deterministic preprocessing stage for `PersistentDataset`.

    This stage is safe to cache because it contains no random sampling:
    - convert loaded arrays to channel-first tensors
    - apply MONAI MRI intensity normalization to the image volume
    """

    def __init__(self, image_key="image", label_key="la_label"):
        super().__init__([image_key, label_key])
        self.image_key = image_key
        self.label_key = label_key
        self.base_transforms = Compose(
            [
                NormalizeIntensityd(keys=[image_key], nonzero=True, channel_wise=True),
            ]
        )

    def _ensure_channel_first(self, tensor_like):
        if torch.is_tensor(tensor_like):
            tensor = tensor_like
        else:
            tensor = torch.as_tensor(tensor_like)

        if tensor.ndim == 2:
            return tensor.unsqueeze(0)

        if tensor.ndim == 3:
            if tensor.shape[0] == 1:
                return tensor
            return tensor.unsqueeze(0)

        if tensor.ndim == 4 and tensor.shape[0] == 1:
            return tensor

        raise ValueError(f"Unsupported tensor shape for channel-first conversion: {tuple(tensor.shape)}")

    def __call__(self, data):
        d = dict(data)
        d[self.image_key] = self._ensure_channel_first(d[self.image_key])
        d[self.label_key] = self._ensure_channel_first(d[self.label_key])
        return self.base_transforms(d)
