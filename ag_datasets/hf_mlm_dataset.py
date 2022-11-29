import torch
import pickle
import torch.utils.data

from typing import Dict
from pathlib import Path


class AGHFMLMDataset(torch.utils.data.Dataset):
    """Implements a torch Dataset cla