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
    batch_input_ids_ = batch_input_ids.detach().clone()
    if len(batch_input_ids_.shape) == 1:
        batch_input_ids_ = batch_input_ids_.reshape(1, -1)

    # random floats in [0, 1] for each 