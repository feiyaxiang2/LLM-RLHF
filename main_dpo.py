import torch
import matplotlib.pyplot as plt

from config import device
from model_setup import load_tokenizer, load_models, load_optimizers
from data import get_dataloaders
from sft import sft_train_step
from dpo import dpo_train_step
from evaluate import evaluate

# ── Setup ──────────────────────────────────────────────────────────────────
tokenizer = load_tokenizer()
model, ref_model = load_models(tokenizer)
optimizer_DPO, optimizer_SFT = load_optimizers(model)
loader, loader_eval = get_dataloaders(tokenizer)

# ── SFT Phase ──────────────────────────────────────────────────────────────
print("=== SFT Phase ===")
sft_losses = sft_train_step(model, loader, optimizer_SFT, sft_epochs=0)
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
        out = model.generate(ids, max_new_tokens=60, do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.1, pad_token_id=tokenizer.pad_token_id)
    print("---")
    print(tokenizer.decode(out[0], skip_special_tokens=True))
model.train()

input("SFT complete. Press Enter to continue to DPO...")

# ── DPO Phase ──────────────────────────────────────────────────────────────
# DPO divergence is measured from the SFT policy, not the raw pretrained model.
ref_model.load_state_dict(model.state_dict())
ref_model.eval()

print("=== DPO Phase ===")
all_losses = []
for epoch in range(1):
    print(f"Epoch {epoch+1}")
    batch_losses = dpo_train_step(model, ref_model, loader, optimizer_DPO)
    all_losses.extend(batch_losses)

evaluate(model, ref_model, loader_eval)

# ── Plot ───────────────────────────────────────────────────────────────────
if all_losses:
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
