
import torch
import torch.utils.data
import pytorch_lightning as pl

from pathlib import Path
from torch.optim import AdamW
from typing import Tuple, List, Dict, Union, Optional
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers.modeling_outputs import MaskedLMOutput
from torchmetrics import Accuracy, F1Score, ConfusionMatrix
from transformers import RobertaTokenizerFast, RobertaForTokenClassification

from ag_datasets.pos_dataset import PoSDataset
from utils.plot_utils import plot_confusion_matrix


class PoSRoBERTa(pl.LightningModule):
    """Wrapper class for a Lightning Module model."""

    def __init__(
            self,
            mlm_model_path: Path,
            tokenizer: RobertaTokenizerFast,
            paths: Tuple[Tuple[Path, Path], Tuple[Path, Path],
                         Tuple[Path, Path]],
            le_path: Path,
            hyperparams: Dict[str, Union[int, float]],
            num_classes: int,
            test_cm_path: Optional[Path]
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.train_ds, self.val_ds, self.test_ds = None, None, None
        self.train_data_path, self.val_data_path, self.test_data_path = paths
        self.model = RobertaForTokenClassification.from_pretrained(
            mlm_model_path, num_labels=num_classes)
        self.freeze_base()
        self.le_path = le_path
        self.hyperparams = hyperparams
        self.num_classes = num_classes
        self.test_cm_path = test_cm_path
        self.val_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.acc = Accuracy(num_classes=num_classes)
        self.f1 = F1Score(num_classes=num_classes, average='weighted')
        self.cm = ConfusionMatrix(num_classes=num_classes)

    def freeze_base(self) -> None:
        for param in self.model.roberta.parameters():
            param.requires_grad = False
        self.model.roberta.eval()

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_ds = PoSDataset(
            tokenizer=self.tokenizer,
            input_ids_path=self.train_data_path[0],
            labels_path=self.train_data_path[1],
            le_path=self.le_path,
            maxlen=self.hyperparams['max-length']
        )
        self.val_ds = PoSDataset(
            tokenizer=self.tokenizer,
            input_ids_path=self.val_data_path[0],
            labels_path=self.val_data_path[1],
            le_path=self.le_path,
            maxlen=self.hyperparams['max-length']
        )
        self.test_ds = PoSDataset(
            tokenizer=self.tokenizer,
            input_ids_path=self.test_data_path[0],
            labels_path=self.test_data_path[1],
            le_path=self.le_path,
            maxlen=self.hyperparams['max-length']
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: torch.Tensor) -> MaskedLMOutput:
        return self.model(input_ids, attention_mask=attention_mask,