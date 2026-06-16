from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import time
import matplotlib.pyplot as plt

# ------------------ Setup ------------------
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# GPT-2 does not define a padding token, so we reuse EOS for batching.
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)
ref_model = AutoModelForCausalLM.from_pretrained(model_name)

model.config.pad_token_id = tokenizer.pad_token_id # Ensure the model knows the correct padding token ID for loss masking.
ref_model.config.pad_token_id = tokenizer.pad_token_id # No padding yet, just telling the model the ID for the padding token.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
ref_model.to(device) # Move the models to GPU

ref_model.eval() # The ref model is fixed and only used for inference, so we set it to eval mode.
for p in ref_model.parameters():
    p.requires_grad = False # Freeze the reference model's parameters to save memory and computation during training.

# Fresh optimizer with DPO-appropriate lr
optimizer_DPO = torch.optim.AdamW(model.parameters(), lr=5e-6)

# AdamW updates only the policy model; the reference model stays fixed.
optimizer_SFT = torch.optim.AdamW(model.parameters(), lr=5e-5)
PROMPT_MAX_LEN = 128
RESP_MAX_LEN = 256

# ------------------ Dataset ------------------
dataset = load_dataset("Anthropic/hh-rlhf", split="train")
dataset_eval = load_dataset("Anthropic/hh-rlhf", split="test")

'''In this dataset, we should generate ONE pair of prompt and response for each of the "chosen" and "rejected" fields.
There are multiple rounds of conversation in each data sample, but the ONLY DIVERGENCE between chosen and rejected is in the last response.'''

def extract_prompt_and_response(text):
    marker = "\n\nAssistant:"
    last_idx = text.rfind(marker)
    if last_idx == -1: # If the marker is not found, we can try a simpler marker that might be used in some examples.
        marker = "Assistant:"
        last_idx = text.rfind(marker)
    if last_idx == -1: # If neither marker is found, we can treat the entire text as the prompt and have an empty response.
        return text.strip(), ""
    prompt = text[:last_idx].strip()
    response = text[last_idx + len(marker):].strip()
    return prompt, response

def preprocess(example):
    p1, c = extract_prompt_and_response(example["chosen"])
    p2, r = extract_prompt_and_response(example["rejected"])
    if p1 != p2:                                              # ✅ soft filter
        return {"prompt": None, "chosen": None, "rejected": None}
    return {"prompt": p1, "chosen": c, "rejected": r}

dataset = dataset.map(preprocess).filter(lambda x: x["prompt"] is not None)      # ✅
dataset_eval = dataset_eval.map(preprocess).filter(lambda x: x["prompt"] is not None)  # ✅
# print(dataset[0]) # Check the first example to ensure the preprocessing worked correctly.


# ------------------ Tokenization (ONCE) ------------------
# We tokenize the dataset once here to avoid redundant tokenization during training and evaluation. The tokenized results are stored in new columns in the dataset for efficient access later.
def tokenize_example(example):
    prompt_str = example["prompt"] + "\n\nAssistant:"
    chosen_str = " " + example["chosen"]
    rejected_str = " " + example["rejected"]

    # Tokenize each piece independently, no truncation yet.
    prompt_ids   = tokenizer(prompt_str,   add_special_tokens=False)["input_ids"]
    chosen_ids   = tokenizer(chosen_str,   add_special_tokens=False)["input_ids"]
    rejected_ids = tokenizer(rejected_str, add_special_tokens=False)["input_ids"]

    # Cap response length first.
    chosen_ids   = chosen_ids[:RESP_MAX_LEN]
    rejected_ids = rejected_ids[:RESP_MAX_LEN]

    # Left-truncate the prompt so it fits in PROMPT_MAX_LEN.
    # This keeps the most recent turns and the trailing "\n\nAssistant:" cue.
    if len(prompt_ids) > PROMPT_MAX_LEN:
        prompt_ids = prompt_ids[-PROMPT_MAX_LEN:]
    prompt_len = len(prompt_ids)

    c_ids = prompt_ids + chosen_ids
    r_ids = prompt_ids + rejected_ids

    return {
        "chosen_input_ids":        c_ids,
        "chosen_attention_mask":   [1] * len(c_ids),
        "rejected_input_ids":      r_ids,
        "rejected_attention_mask": [1] * len(r_ids),
        "prompt_len":              prompt_len,
    }

dataset = dataset.map(tokenize_example)
dataset_eval = dataset_eval.map(tokenize_example)

# ------------------ Collate ------------------
def collate_fn(batch):
    def pad_ids(key): # Pad the input IDs so that all seqs in the batch have the same length
        return pad_sequence(
            [torch.tensor(b[key]) for b in batch],
            batch_first=True,
            padding_value=tokenizer.pad_token_id # Pad with the tokenizer's pad token ID so that the model can ignore these positions during attention and loss calculation.
        )
    def pad_mask(key):
        return pad_sequence(
            [torch.tensor(b[key]) for b in batch],
            batch_first=True,
            # Padding positions should be ignored by the model.
            padding_value=0 # Attention mask has 1 for real tokens and 0 for padding, so we pad with 0 to ensure the model ignores the padded positions during attention calculations.
        )
    return {
        "chosen_input_ids":        pad_ids("chosen_input_ids"),
        "chosen_attention_mask":   pad_mask("chosen_attention_mask"),
        "rejected_input_ids":      pad_ids("rejected_input_ids"),
        "rejected_attention_mask": pad_mask("rejected_attention_mask"),
        "prompt_len": torch.tensor([b["prompt_len"] for b in batch]),
    }

batch_size = 8
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
loader_eval = DataLoader(dataset_eval, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# ------------------ SFT ------------------
def sft_train_step(model, loader, optimizer, sft_epochs=1):
    model.train()
    all_losses = []
    for epoch in range(sft_epochs):
        epoch_losses = []
        count = 0
        for batch in loader:
            input_ids = batch["chosen_input_ids"].to(device)
            attention_mask = batch["chosen_attention_mask"].to(device)
            prompt_lens = batch["prompt_len"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits # size [B, T, V]

            shift_logits = logits[:, :-1, :] # size [B, T-1, V], we predict the next token at each position
            shift_labels = input_ids[:, 1:] # size [B, T-1], the target labels are the input IDs shifted by one position to align with the next-token prediction task
            shift_mask = attention_mask[:, 1:] # size [B, T-1], we also shift the attention mask to align with the shifted labels, so that we only compute loss on positions where there are real tokens (not padding) in the target labels.

            # Mask out prompt tokens — only supervise on response tokens.
            B, T = shift_labels.shape
            response_mask = torch.zeros(B, T, device=device)
            for i in range(B):
                start = max(int(prompt_lens[i].item()) - 1, 0)
                response_mask[i, start:] = 1
            final_mask = response_mask * shift_mask

            loss_per_token = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                reduction="none",
            ).reshape(B, T)

            loss = (loss_per_token * final_mask).sum() / final_mask.sum().clamp(min=1) # average loss per response token, ignoring padding and prompt tokens

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


# ------------------ Log Prob ------------------
def sequence_logprobs(model, input_ids, attention_mask, prompt_lens):

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    prompt_lens = prompt_lens.to(device)

    # Causal LM logits predict the next token at every position.
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits 
    '''The model needs to see the full prompt+response to assign proper probabilities to the response tokens, so we feed in the entire sequence of input IDs. 
    The attention mask ensures that the model only attends to the real tokens and ignores the padding during its computations.'''

    # Shift logits and labels by one token to score next-token predictions.
    target_labels = input_ids[:, 1:] # size [B, T-1]
    target_logits = logits[:, :-1, :] # size [B, T-1, V]
    shift_mask = attention_mask[:, 1:]

    log_probs = F.log_softmax(target_logits, dim=-1) # size [B, T-1, V]

    token_log_probs = torch.gather(
        log_probs, dim=2, index=target_labels.unsqueeze(-1)
    ).squeeze(-1) # size [B, T-1]

    B, T = token_log_probs.shape
    response_mask = torch.zeros_like(token_log_probs)

    for i in range(B):
        # Only the response tokens after the prompt should contribute.
        idx = int(prompt_lens[i].item())
        if idx - 1 < 0:
            start = 0
        else:
            start = idx - 1
        response_mask[i, start:] = 1

    final_mask = response_mask * shift_mask

    token_sums = (token_log_probs * final_mask).sum(dim=1)

    # token_counts = final_mask.sum(dim=1).clamp(min=1)

    return token_sums


# ------------------ Train ------------------
def train_step(model, ref_model, loader, optimizer):

    model.train()
    # beta controls how strongly the model is pushed toward the chosen reply.
    beta = 0.1

    batch_losses = []
    count = 0
    for batch in loader:

        c_ids = batch["chosen_input_ids"].to(device)
        c_mask = batch["chosen_attention_mask"].to(device)

        r_ids = batch["rejected_input_ids"].to(device)
        r_mask = batch["rejected_attention_mask"].to(device)

        p_len = batch["prompt_len"].to(device)

        start_time = time.time()

        # Policy model scores for the chosen and rejected responses.
        lp_c = sequence_logprobs(model, c_ids, c_mask, p_len)
        lp_r = sequence_logprobs(model, r_ids, r_mask, p_len)

        # Reference model gives the baseline preference signal.
        with torch.no_grad():
            lp_c_ref = sequence_logprobs(ref_model, c_ids, c_mask, p_len)
            lp_r_ref = sequence_logprobs(ref_model, r_ids, r_mask, p_len)

        # length_c = c_mask.sum(dim=1)
        # length_r = r_mask.sum(dim=1)

        # delta = (lp_c / length_c) - (lp_r / length_r)
        # delta_ref = (lp_c_ref / length_c) - (lp_r_ref / length_r)

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

        # Smoothed loss over last 100 batches
        window = 100
        if len(batch_losses) >= window:
            smooth = sum(batch_losses[-window:]) / window
            print(f"Loss: {loss_item:.4f} | Smooth({window}): {smooth:.4f} | Time: {end_time - start_time:.2f}s")
        else:
            print(f"Loss: {loss_item:.4f} | Smooth({window}): n/a ({len(batch_losses)}/{window}) | Time: {end_time - start_time:.2f}s")

        count += 1
        print(f"Batch {count}/{len(loader)} processed")

    return batch_losses

# ------------------ Evaluate ------------------
def evaluate(model, ref_model, loader):

    model.eval()
    ref_model.eval()

    correct_model = 0
    correct_ref = 0
    total = 0

    margins_model = []
    margins_ref = []

    with torch.no_grad():
        for batch in loader:

            c_ids = batch["chosen_input_ids"].to(device)
            c_mask = batch["chosen_attention_mask"].to(device)

            r_ids = batch["rejected_input_ids"].to(device)
            r_mask = batch["rejected_attention_mask"].to(device)

            p_len = batch["prompt_len"].to(device)

                # Compare chosen-vs-rejected log-probability gaps.
            lp_c = sequence_logprobs(model, c_ids, c_mask, p_len)
            lp_r = sequence_logprobs(model, r_ids, r_mask, p_len)

            lp_c_ref = sequence_logprobs(ref_model, c_ids, c_mask, p_len)
            lp_r_ref = sequence_logprobs(ref_model, r_ids, r_mask, p_len)

                # Accuracy means the model assigns a higher score to the chosen reply.
            correct_model += (lp_c > lp_r).sum().item()
            correct_ref += (lp_c_ref > lp_r_ref).sum().item()
            total += lp_c.size(0)

                # Margin tracks how much stronger the chosen reply is than the rejected one.
            margins_model.append((lp_c - lp_r).detach())
            margins_ref.append((lp_c_ref - lp_r_ref).detach())

            # Merge batch-wise margins for summary statistics.
    margins_model = torch.cat(margins_model)
    margins_ref = torch.cat(margins_ref)

            # Report the final comparison numbers.
    acc_model = correct_model / total
    acc_ref = correct_ref / total

    print("====== Evaluation ======")
    print(f"Model Accuracy: {acc_model:.4f}")
    print(f"Ref Accuracy:   {acc_ref:.4f}")
    print(f"Improvement:    {acc_model - acc_ref:+.4f}")

    print("\n--- Margin Stats ---")
    print(f"Model Δ mean: {margins_model.mean().item():.3f}")
    print(f"Ref Δ mean:   {margins_ref.mean().item():.3f}")

    print(f"Model Δ var:  {margins_model.var().item():.3f}")
    print(f"Ref Δ var:    {margins_ref.var().item():.3f}")

    # How often the trained model's margin beats the reference margin.
    better = (margins_model > margins_ref).float().mean().item()

    print("\n--- Head-to-head ---")
    print(f"Model better than ref: {better:.4f}")

    return acc_model, acc_ref

# ------------------ Run ------------------
print("=== SFT Phase ===")
sft_losses = sft_train_step(model, loader, optimizer_SFT, sft_epochs=1)
if sft_losses:
    print(f"[SFT] Final mean loss: {sum(sft_losses)/len(sft_losses):.4f} | Last loss: {sft_losses[-1]:.4f}")

# Quick generation sanity check
model.eval()
test_prompts = [
    "\n\nHuman: How do I make a cup of tea?\n\nAssistant:",
    "\n\nHuman: What's the capital of France?\n\nAssistant:",
]
for p in test_prompts:
    ids = tokenizer(p, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=60, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tokenizer.pad_token_id)
    print("---")
    print(tokenizer.decode(out[0], skip_special_tokens=True))
model.train()

input("SFT complete. Press Enter to continue to DPO...")

# DPO divergence is measured from the SFT policy, not the raw pretrained model.
ref_model.load_state_dict(model.state_dict())
ref_model.eval()

print("=== DPO Phase ===")
all_losses = []
for epoch in range(1):
    print(f"Epoch {epoch+1}")
    batch_losses = train_step(model, ref_model, loader, optimizer_DPO)
    all_losses.extend(batch_losses)

evaluate(model, ref_model, loader_eval)

# Plot training loss so you can see whether optimization is stabilizing.
if len(all_losses) > 0:
    window = 100
    smoothed = [
        sum(all_losses[max(0, i - window):i]) / min(i, window)
        for i in range(1, len(all_losses) + 1)
    ]

    plt.figure()
    plt.plot(all_losses, alpha=0.3, label="Raw loss")
    plt.plot(smoothed, label=f"Smoothed (window={window})")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Iterations")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_vs_iterations.png")