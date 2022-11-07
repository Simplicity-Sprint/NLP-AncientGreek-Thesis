import torch

from transformers import RobertaTokenizerFast


def mlm(
        batch_input_ids: torch.Tens