import torch

from transformers import RobertaTokenizerFast


def mlm(
        batch_input_ids: torch.Tensor,
        tokenizer: RobertaTokenizerFast,
        mask_probability_p: float = 0.15
) -> torch.Tensor:
    """Randomly masks the parts