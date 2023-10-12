import logging
from collections import defaultdict
import torch
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: list[dict]) -> dict:
    """
    Collate and pad fields in dataset items
    """
    result_batch = dict()
    for key in dataset_items[0].keys():
        match key:
            case "text_encoded" | "spectrogram":
                result_batch[f"{key}_length"] = torch.tensor(
                    data=[item.get(key).shape[-1] for item in dataset_items]
                )
                result_batch[key] = pad_sequence(
                    sequences=[
                        torch.squeeze(item.get(key), dim=0).t()
                        for item in dataset_items
                    ],
                    batch_first=True,
                )
            case _:
                result_batch[key] = [item.get(key) for item in dataset_items]
    result_batch["spectrogram"] = result_batch["spectrogram"].permute(0, 2, 1)
    return result_batch
