import torch
import pickle

from pathlib import Path
from typing import Dict, List
from torch.utils.data import Dataset
from transformers import RobertaTokenizerFast


class PoSDataset(Dataset):
    """PoS Tagging Dataset."""

    def __init__(
            self,
            tokenizer: RobertaTokenizerFast,
            input_ids_path: Path,
            labels_path: Path,
            le_path: Path,
            maxlen: int
    ):
        """
        :param tokenizer:
            Pre-trained RoBERTa tokenizer for t