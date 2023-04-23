import time
import torch
import logging
import argparse
import warnings
import numpy as np
import pytorch_lightning as pl

from pathlib import Path
from functools import partial
from typing import Tuple, Dict, Union
from transformers import RobertaTokenizerFast
from pytorch_lightning import seed_everything
from hyperopt import hp, fmin, tpe, space_eval
from pytorch_lightning.loggers import TensorBoardLogger

from pl_models.mlm_model import LitRoBERTaMLM
from utils.plot_utils import get_pl_mlm_losses
from utils.cmd_args import parse_tune_mlm_input
from utils.fs_utils import force_empty_directory, delete_file_if_exists
from data_preparation.processing import TOKENIZER_PATH, PROCESSED_DATA_PATH


BEST_VAL_LOSS = float('inf')
BEST_ARGS = None


def create_and_train_model(
        args: Dict[str, Union[float, int]],
        constants: Dict[str, Union[int, float, bool, Tuple[Path, Path, Path],
                                   RobertaTokenizerFast, Path]]
) -> LitRoBERTaMLM:
    """Creates and pre-trains a PL MLM Ancient Greek RoBERTa model."""
    # set the seed
    seed_everything(args['seed'])

    # create PL model
    model = LitRoBERTaMLM(
        tokenizer=constants['tokenizer'],
        paths=constants['data-paths'],
        hyperparams={**args, **constants}
    )

    # handle logging
    logdir = 