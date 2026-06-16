import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM

from config import device, model_name
from model_setup import load_tokenizer, load_models
from data_ppo import get_dataloaders_ppo
from data import get_dataloaders
from GRPO import grpo
from reward_model import RewardModel


tokenizer = load_tokenizer(direction="left")
model, _ = load_models(tokenizer)
model = model.to(device)
loader, _ = get_dataloaders_ppo(tokenizer)
_, loader_eval = get_dataloaders(tokenizer)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-6)
_rm_base = AutoModelForCausalLM.from_pretrained(model_name)
reward_model = RewardModel(_rm_base.transformer).to(device)
reward_model.load_state_dict(torch.load("reward_model.pt", map_location=device))
reward_model.eval()
for p in reward_model.parameters():
    p.requires_grad = False
with torch.no_grad():
    _, ref_model = load_models(tokenizer)
    ref_model = ref_model.to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

epochs = 1
grpo_losses, all_accuracies = grpo(model, ref_model, loader, loader_eval, optimizer, tokenizer, reward_model, epochs=epochs, m_train=2, epsilon=0.2, group_size=4, beta_kl=0.02)
for batch_idx, acc_model, acc_ref in all_accuracies:
    print(f"Batch {batch_idx} | Model Accuracy: {acc_model:.4f} | Reference Accuracy: {acc_ref:.4f} | Improvement: {acc_model - acc_ref:+.4f}")
if grpo_losses:
    window = 100
    smoothed = [
        sum(grpo_losses[max(0, i - window):i]) / min(i, window)
        for i in range(1, len(grpo_losses) + 1)
    ]
    plt.figure()
    plt.plot(grpo_losses, label="GRPO Loss")
    plt.plot(smoothed, label=f"Smoothed (window={window})")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("GRPO Training Loss")
    plt.legend()
    plt.show()