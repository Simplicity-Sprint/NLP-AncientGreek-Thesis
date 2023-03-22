
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

    def compute_loss(self, model, inputs, return_outputs=False):
        """Override the compute_loss() function such that it logs the
            accuracy and the f1 score."""
        if self.label_smoother is not None and 'labels' in inputs:
            labels = inputs.pop('labels')
        else:
            labels = None
        outputs = model(**inputs)

        # compute batch accuracy and f1 score for training batches
        #  Small hack: If the logits do not require a gradient, then this
        #    function has been called with torch.no_grad(), which means that
        #    this is an evaluation call, so don't compute the metrics as this
        #    block is meant only for training.
        if 'labels' in inputs and outputs.logits.requires_grad:
            preds = outputs.logits.detach().cpu().argmax(-1).reshape(-1).numpy()
            labels_ = inputs['labels'].detach().cpu().reshape(-1).numpy()

            valid_indices = labels_ != -100
            preds = preds[valid_indices]
            labels_ = labels_[valid_indices]

            acc = accuracy_score(labels_, preds)
            f1 = f1_score(labels_, preds, average='weighted')

            self.log({'accuracy': acc, 'f1': f1})

        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples
            #  instead of ModelOutput.
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


def main(args: argparse.Namespace):
    """main() driver function."""

    # args
    seed = get_seed(args.seed)
    set_seed(seed)

    # empty the tensorboard and model directories
    force_empty_directory(args.logdir)
    force_empty_directory(args.savedir)

    # create the model
    model_dir = glob.glob(f'{args.pre_trained_model}/checkpoint-*')[0]
    model = RobertaForTokenClassification.from_pretrained(
        model_dir,
        num_labels=PoSDataset.num_classes(LABEL_ENCODER_PATH)
    )

    # define the custom hyperparameters for the model here
    custom_hyperparameters = {
        'max-length': 512,
        'batch-size': 4,
        'learning-rate': 1e-4,
        'weight-decay': 1e-2,
        'decay-lr-at-percentage-of-steps': 0.1,
        'train-epochs': 5
    }
