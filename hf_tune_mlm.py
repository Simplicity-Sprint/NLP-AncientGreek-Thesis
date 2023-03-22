
import argparse

from transformers import (
    RobertaTokenizerFast,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    IntervalStrategy,
    SchedulerType,
    RobertaConfig,
    RobertaForMaskedLM,
    Trainer
)
from ray import tune
from pathlib import Path
from ray.tune.trial import Trial
from typing import Dict, Optional, Any
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from transformers.training_args import OptimizerNames
from transformers.trainer_utils import HPSearchBackend

from utils.cmd_args import parse_tune_mlm_input
from ag_datasets.hf_mlm_dataset import AGHFMLMDataset
from utils.fs_utils import force_empty_directory, delete_file_if_exists
from data_preparation.processing import TOKENIZER_PATH, PROCESSED_DATA_PATH


def main(args: argparse.Namespace):
    """main() driver function."""

    # define the constant values of the model
    tokenizer = RobertaTokenizerFast.from_pretrained(TOKENIZER_PATH)
    local_dir = (Path('logs')/'hf-mlm-ray-tune-results').absolute()
    force_empty_directory(local_dir)
    output_dir = Path('objects')/'HF-Tuned-AG-RoBERTa'
    force_empty_directory(output_dir)
    tune_logfile = (Path('logs')/'hf-mlm-hp-tuning-results.txt').absolute()
    delete_file_if_exists(tune_logfile)
    resources_per_trial = {'cpu': 1, 'gpu': 1}
    constants = {
        'max-length': 512,
        'mask-probability': 0.15,
        'type-vocab-size': 1,
        'decay-lr-at-percentage-of-steps': 0.1,
        'train-epochs': 2,
        'tokenizer': tokenizer,
        'local-dir': local_dir,
        'output-dir': output_dir,
        'tune-logfile': tune_logfile,
        'resources-per-trial': resources_per_trial
    }

    def model_init(trial: Trial) -> RobertaForMaskedLM:
        """Initializes and returns a model given a Ray Tune trial object."""
        # trial will be `None` during the creation of Trainer()
        if trial is None:
            trial = {
                'hidden-size': 128,
                'num-attention-heads': 2,
                'num-hidden-layers': 2
            }

        # the hidden size must be a multiple of the number of attention heads
        hidden_size = trial['hidden-size']
        num_attention_heads = trial['num-attention-heads']
        hidden_size = (hidden_size // num_attention_heads) * num_attention_heads

        # create and return the model
        config = RobertaConfig(
            vocab_size=tokenizer.vocab_size,
            max_position_embeddings=constants['max-length'] + 2,
            hidden_size=int(hidden_size),
            num_attention_heads=int(trial['num-attention-heads']),