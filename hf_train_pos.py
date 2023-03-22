
import glob
import torch
import argparse

from transformers import (
    RobertaTokenizerFast,
    TrainingArguments,
    IntervalStrategy,
    SchedulerType,
    RobertaForTokenClassification,
    Trainer,
    set_seed
)
from typing import Dict, Tuple
from torchmetrics import ConfusionMatrix
from sklearn.metrics import accuracy_score, f1_score
from transformers.training_args import OptimizerNames
from transformers.trainer_utils import EvalPrediction

from utils.cmd_args import parse_hf_pos_input
from ag_datasets.pos_dataset import PoSDataset
from utils.fs_utils import force_empty_directory
from utils.run_utils import hyperparams_from_config, get_seed
from utils.plot_utils import plot_pos_metrics, plot_confusion_matrix
from data_preparation.processing import (
    TOKENIZER_PATH,
    PROCESSED_DATA_PATH,
    LABEL_ENCODER_PATH
)


class CustomMetricsTrainer(Trainer):
    """Overriding the Trainer Class so that custom metrics such as Accuracy
        and F1 score can be logged during training."""