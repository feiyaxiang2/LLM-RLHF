import torch
import torch.nn.functional as F
from config import device


def sequence_logprobs(model, input_ids, attention_mask, prompt_lens, normalize_by_length=False):
    """Return summed response log-probs for each sequence in the batch."""
    input_ids      = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    prompt_lens    = prompt_lens.to(device)

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    # Shift by one to score next-token predictions.
    target_labels  = input_ids[:, 1:] # shape (B, T-1)
    target_logits  = logits[:, :-1, :] # shape (B, T-1, V)
    shift_mask     = attention_mask[:, 1:] # shape (B, T-1), 

    log_probs = F.log_softmax(target_logits, dim=-1) # shape (B, T-1, V), convert logits to log-probabilities for numerical stability when summing
    token_log_probs = torch.gather(
        log_probs, dim=2, index=target_labels.unsqueeze(-1) # gather requires the index tensor to have the same number of dimensions as the input tensor
    ).squeeze(-1) # shape (B, T-1), gather the log-probabilities corresponding to the target labels

    B, T = token_log_probs.shape
    response_mask = torch.zeros_like(token_log_probs)
    for i in range(B):
        idx = int(prompt_lens[i].item())
        start = max(idx - 1, 0)
        response_mask[i, start:] = 1

    final_mask = response_mask * shift_mask # first input the whole sequence, then mask out the prompt tokens and the padding tokens, since prob depends on the prompt

    token_sums = (token_log_probs * final_mask).sum(dim=1)

    if normalize_by_length:
        token_counts = final_mask.sum(dim=1).clamp(min=1)
        token_sums = token_sums / token_counts

    return token_sums

def sequence_logprobs_ppo(model, input_ids, attention_mask, prompt_lens):
    """Return summed response log-probs for each sequence in the batch."""
    input_ids      = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    prompt_lens    = prompt_lens.to(device)

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits # shape (B, T, V) - get the logits for all tokens in the input sequence, where B is batch size, T is sequence length, and V is vocabulary size

    # Shift by one to score next-token predictions.
    target_labels  = input_ids[:, 1:] # shape (B, T-1)
    target_logits  = logits[:, :-1, :] # shape (B, T-1, V)
    shift_mask     = attention_mask[:, 1:] # shape (B, T-1), 

    log_probs = F.log_softmax(target_logits, dim=-1) # shape (B, T-1, V), convert logits to log-probabilities for numerical stability when summing
    token_log_probs = torch.gather(
        log_probs, dim=2, index=target_labels.unsqueeze(-1) # gather requires the index tensor to have the same number of dimensions as the input tensor
    ).squeeze(-1) # shape (B, T-1), gather the log-probabilities corresponding to the target labels

    B, T = token_log_probs.shape
    response_mask = torch.zeros_like(token_log_probs)

    for i in range(B):
        prompt_len = int(prompt_lens[i].item())

        # Use the first non-zero position in attention_mask to find left-padding only.
        # generated_ids has both left-pad (from prompt batching) and right-pad (from
        # variable-length responses). Using total pad_len = seq_len - real_len would
        # include right-pad and push the response start too far right, dropping early
        # response tokens from the mask.
        nonzero = (attention_mask[i] == 1).nonzero(as_tuple=True)[0]
        left_pad_len = int(nonzero[0].item()) if len(nonzero) > 0 else attention_mask.size(1)

        # after shifting, token_log_probs[:, k] scores input_ids[:, k+1]
        start = left_pad_len + prompt_len - 1
        start = max(start, 0)

        response_mask[i, start:] = 1

    final_mask = response_mask * shift_mask # first input the whole sequence, then mask out the prompt tokens and the padding tokens, since prob depends on the prompt


    return token_log_probs, final_mask
