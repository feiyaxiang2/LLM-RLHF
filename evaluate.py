import torch
from config import device
from logprobs import sequence_logprobs


def evaluate(model, ref_model, loader):
    model.eval()
    ref_model.eval()

    correct_model = 0
    correct_ref   = 0
    total         = 0
    margins_model = []
    margins_ref   = []

    with torch.no_grad():
        for batch in loader:
            c_ids  = batch["chosen_input_ids"].to(device)
            c_mask = batch["chosen_attention_mask"].to(device)
            r_ids  = batch["rejected_input_ids"].to(device)
            r_mask = batch["rejected_attention_mask"].to(device)
            p_len  = batch["prompt_len"].to(device)

            lp_c     = sequence_logprobs(model,     c_ids, c_mask, p_len, normalize_by_length=True)
            lp_r     = sequence_logprobs(model,     r_ids, r_mask, p_len, normalize_by_length=True)
            lp_c_ref = sequence_logprobs(ref_model, c_ids, c_mask, p_len, normalize_by_length=True)
            lp_r_ref = sequence_logprobs(ref_model, r_ids, r_mask, p_len, normalize_by_length=True)

            correct_model += (lp_c > lp_r).sum().item()
            correct_ref   += (lp_c_ref > lp_r_ref).sum().item()
            total         += lp_c.size(0)

            margins_model.append((lp_c - lp_r).detach())
            margins_ref.append((lp_c_ref - lp_r_ref).detach())

    margins_model = torch.cat(margins_model)
    margins_ref   = torch.cat(margins_ref)

    acc_model = correct_model / total
    acc_ref   = correct_ref   / total

    print("====== Evaluation ======")
    print(f"Model Accuracy: {acc_model:.4f}")
    print(f"Ref Accuracy:   {acc_ref:.4f}")
    print(f"Improvement:    {acc_model - acc_ref:+.4f}")

    print("\n--- Margin Stats ---")
    print(f"Model Δ mean: {margins_model.mean().item():.3f}")
    print(f"Ref Δ mean:   {margins_ref.mean().item():.3f}")
    print(f"Model Δ var:  {margins_model.var().item():.3f}")
    print(f"Ref Δ var:    {margins_ref.var().item():.3f}")

    better = (margins_model > margins_ref).float().mean().item()
    print("\n--- Head-to-head ---")
    print(f"Model better than ref: {better:.4f}")

    return acc_model, acc_ref
