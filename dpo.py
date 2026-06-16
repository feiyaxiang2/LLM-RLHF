import time
import torch
import torch.nn.functional as F
from config import device
from logprobs import sequence_logprobs


def dpo_train_step(model, ref_model, loader, optimizer, beta=0.1):
    model.train()
    batch_losses = []
    count = 0

    for batch in loader:
        c_ids  = batch["chosen_input_ids"].to(device)
        c_mask = batch["chosen_attention_mask"].to(device)
        r_ids  = batch["rejected_input_ids"].to(device)
        r_mask = batch["rejected_attention_mask"].to(device)
        p_len  = batch["prompt_len"].to(device)

        start_time = time.time()

        lp_c = sequence_logprobs(model, c_ids, c_mask, p_len, normalize_by_length=True)
        lp_r = sequence_logprobs(model, r_ids, r_mask, p_len, normalize_by_length=True)

        with torch.no_grad():
            lp_c_ref = sequence_logprobs(ref_model, c_ids, c_mask, p_len, normalize_by_length=True)
            lp_r_ref = sequence_logprobs(ref_model, r_ids, r_mask, p_len, normalize_by_length=True)

        delta     = lp_c - lp_r
        delta_ref = lp_c_ref - lp_r_ref

        # DPO loss: increase the policy's preference gap beyond the reference gap.
        loss = -F.logsigmoid(beta * (delta - delta_ref)).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        end_time = time.time()
        loss_item = loss.item()
        batch_losses.append(loss_item)

        window = 100
        count += 1
        if len(batch_losses) >= window:
            smooth = sum(batch_losses[-window:]) / window
            print(f"Batch {count}/{len(loader)} | Loss: {loss_item:.4f} | Smooth({window}): {smooth:.4f} | Time: {end_time - start_time:.2f}s")
        else:
            print(f"Batch {count}/{len(loader)} | Loss: {loss_item:.4f} | Smooth({window}): n/a ({len(batch_losses)}/{window}) | Time: {end_time - start_time:.2f}s")

    return batch_losses
