
import torch
import argparse

from transformers import (
    RobertaTokenizerFast,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    IntervalStrategy,
    SchedulerType,
    RobertaConfig,
    RobertaForMaskedLM,
    Trainer,
    set_seed
)
from transformers.training_args import OptimizerNames

from utils.plot_utils import plot_mlm_losses