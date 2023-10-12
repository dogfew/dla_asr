from torch_audiomentations import AddColoredNoise, AddBackgroundNoise
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class BackGroundNoise(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = AddBackgroundNoise(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)


class ColoredNoise(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = AddColoredNoise(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
