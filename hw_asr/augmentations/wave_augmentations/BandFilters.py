import torch_audiomentations
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class BandPassFilter(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = torch_audiomentations.BandPassFilter(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)


class BandStopFilter(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = torch_audiomentations.BandStopFilter(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
