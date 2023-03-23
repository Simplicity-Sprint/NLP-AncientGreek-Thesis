
import torch
import torch.utils.data
import pytorch_lightning as pl

from pathlib import Path
from torch.optim import AdamW
from typing import List, Tuple, Dict, Union, Optional
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import (
    RobertaTokenizerFast,
    RobertaConfig,
    RobertaForMaskedLM
)
from transformers.modeling_outputs import MaskedLMOutput

from ag_datasets.mlm_dataset import AGMLMDataset


def create_roberta_mlm_model(
        roberta_tokenizer: RobertaTokenizerFast,
        hyperparameters: Dict[str, Union[int, float]]
) -> RobertaForMaskedLM:
    """Creates and returns a RoBERTa for MLM model from the given arguments."""
    config = RobertaConfig(
        vocab_size=roberta_tokenizer.vocab_size,
        max_position_embeddings=hyperparameters['max-length'] + 2,
        hidden_size=int(hyperparameters['hidden-size']),
        num_attention_heads=int(hyperparameters['num-attention-heads']),
        num_hidden_layers=int(hyperparameters['num-hidden-layers']),
        type_vocab_size=hyperparameters['type-vocab-size'],
        bos_token_id=roberta_tokenizer.bos_token_id,
        eos_token_id=roberta_tokenizer.eos_token_id,
        pad_token_id=roberta_tokenizer.pad_token_id
    )
    return RobertaForMaskedLM(config).train()


class LitRoBERTaMLM(pl.LightningModule):
    """Wrapper class for a Lightning Module Ancient Greek RoBERTa model."""

    def __init__(
            self,
            tokenizer: RobertaTokenizerFast,
            paths: Tuple[Path, Path, Path],
            hyperparams: Dict[str, Union[int, float]]
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.train_ds, self.val_ds, self.test_ds = None, None, None
        self.train_data_path, self.val_data_path, self.test_data_path = paths