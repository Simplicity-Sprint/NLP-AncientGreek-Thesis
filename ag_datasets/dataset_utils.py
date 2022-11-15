import torch

from transformers import RobertaTokenizerFast


def mlm(
        batch_input_ids: torch.Tensor,
        tokenizer: RobertaTokenizerFast,
        mask_probability