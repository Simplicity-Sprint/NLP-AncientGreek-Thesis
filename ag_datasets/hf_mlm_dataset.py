import torch
import pickle
import torch.utils.data

from typing import Dict
from pathlib import Path


class AGHFMLMDataset(torch.utils.data.Dataset):
    """Implements a torch Dataset class for Ancient Greek input data that
        will be used by the HF Trainer API during training."""

    def __init__(
            self,
            