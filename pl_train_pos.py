import torch
import argparse
import pytorch_lightning as pl

from transformers import RobertaTokenizerFast
from pytorch_lightning.loggers import TensorBoardLogger

from pl_models.pos_model import PoSRoBERTa
from utils.cmd_args import parse_pl_pos_input
from utils.plot_utils import plot_pos_metrics
from ag_datasets.pos_dataset import PoSDataset
from utils.fs_utils import force_empty_directory
from utils.run_utils import device_from_str, get_seed, hyperparams_from_config
from data_preparation.processing import (
    TOKENIZER_PATH,
    LABEL_ENCODER_PATH,
    PROCESSED_DATA_PATH
)


def main(args: argparse.Namespace):
    """main() driver function."""

    # fix some ar