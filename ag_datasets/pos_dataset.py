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
            Pre-trained RoBERTa tokenizer for the Ancient Greek language.

        :param input_ids_path:
            Path to the .pkl file containing the input IDs.

        :param labels_path:
            Path to the .pkl file containing the corresponding PoS labels.

        :param le_path:
            Path to the .pkl file containing the pre-trained sklearn
            LabelEncoder object.

        :param maxlen:
            Maximum length of a sequence.
        """
        self.tokenizer = tokenizer
        with open(input_ids_path, 'rb') as fp:
            self.input_ids = pickle.load(fp)
        with open(labels_path, 'rb') as fp:
            self.labels = pickle.load(fp)
        with open(le_path, 'rb') as fp:
            self.le = pickle.load(fp)
        self.maxlen = maxlen

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, item: in