import torch

from transformers import RobertaTokenizerFast


def mlm(
        batch_input_ids: torch.Tensor,
        tokenizer: RobertaTokenizerFast,
        mask_probability_p: float = 0.15
) -> torch.Tensor:
    """Randomly masks the parts of the given tensor, according to a masking
        probability `p`."""
    # clone the array and fix its shape to [B, maxlen]
    batch_input_ids_ = b