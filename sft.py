import torch
import torch.nn.functional as F
from config import device


def sft_train_step(model, loader, optimizer, sft_epochs=1):
    model.train()
    all_losses = []

    for epoch in range(sft_epochs):
        epoch_losses = []
        count = 0

        for batch in loader:
            input_ids      = batch["chosen_input_ids"].to(device)
            attention_mask = batch["chosen_attention_mask"].to(device) # these are all 1s for the actual tokens and will be padded to 0s in the collate_fn, so we can use them to mask out the loss for the padding tokens
            prompt_lens    = batch["prompt_len"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits # shape (B, T, V)

            shift_logits = logits[:, :-1, :] # these logits predict tokens at position 1, ..., T (index starting from 0).
            shift_labels = input_ids[:, 1:]
            shift_mask   = attention_mask[:, 1:] 

            # Mask out prompt tokens — only supervise on response tokens.
            B, T = shift_labels.shape
            response_mask = torch.zeros(B, T, device=device)
            for i in range(B):
                start = max(int(prompt_lens[i].item()) - 1, 0)
                response_mask[i, start:] = 1
            final_mask = response_mask * shift_mask

            loss_per_token = F.cross_entropy( # expects inputs of shape (N, C) and targets of shape (N,) 
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1), # turns to vector of shape (B*T,)
                reduction="none",
            ).reshape(B, T) # reshape back to (B, T) so that we can apply the final_mask to zero out the loss for non-response tokens and padding tokens

            loss = (loss_per_token * final_mask).sum() / final_mask.sum().clamp(min=1) # average loss per response token, clamp to avoid division by zero

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            count += 1
            loss_item = loss.item()
            epoch_losses.append(loss_item)

            window = 100
            if len(epoch_losses) >= window:
                smooth = sum(epoch_losses[-window:]) / window
                print(f"[SFT] Epoch {epoch+1} | Batch {count}/{len(loader)} | Loss: {loss_item:.4f} | Smooth({window}): {smooth:.4f}")
            else:
                print(f"[SFT] Epoch {epoch+1} | Batch {count}/{len(loader)} | Loss: {loss_item:.4f} | Smooth({window}): n/a ({len(epoch_losses)}/{window})")

        mean = sum(epoch_losses) / len(epoch_losses)
        print(f"[SFT] Epoch {epoch+1} mean loss: {mean:.4f}")
        all_losses.extend(epoch_losses)

    return all_losses
