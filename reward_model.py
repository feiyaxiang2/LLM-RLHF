import torch
from config import device, PROMPT_MAX_LEN, RESP_MAX_LEN, model_name
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from data import get_dataloaders
from model_setup import load_tokenizer
from sft import sft_train_step
import matplotlib.pyplot as plt

class RewardModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model # outputs hidden states for all tokens in the input sequence, before the language modeling head
        hidden_size = self.base_model.config.hidden_size
        self.reward_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 1),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask) # outputs a HuggingFace BaseModelOutput; .last_hidden_state has shape (B, T, H)
        last_token_indices = attention_mask.size(1) - 1 - attention_mask.flip(dims=[1]).argmax(dim=1)
        '''Using the last token because it attends to all previous tokens and thus can be a good summary'''
        hidden = outputs.last_hidden_state # shape (B, T, H), get the hidden states of all tokens
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device) # shape (B,), create a tensor of batch indices [0, 1, ..., B-1]
        last_hidden = hidden[batch_indices, last_token_indices] # shape (B, H), gather the hidden states of the last non-padding tokens using advanced indexing
        reward = self.reward_head(last_hidden).squeeze(-1) # shape (B,), apply the reward head to get a scalar reward value for each sequence in the batch
        return reward

def eval_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids_chosen = batch["chosen_input_ids"].to(device)
            attention_mask_chosen = batch["chosen_attention_mask"].to(device)
            input_ids_rejected = batch["rejected_input_ids"].to(device)
            attention_mask_rejected = batch["rejected_attention_mask"].to(device)
            rewards_chosen = model(input_ids=input_ids_chosen, attention_mask=attention_mask_chosen)
            rewards_rejected = model(input_ids=input_ids_rejected, attention_mask=attention_mask_rejected)
            correct += (rewards_chosen > rewards_rejected).sum().item()
            total += len(rewards_chosen)
    model.train()
    return correct / total if total > 0 else 0.0


if __name__ == "__main__":
    tokenizer = load_tokenizer(direction="right")
    loader, loader_eval = get_dataloaders(tokenizer)
    epochs = 3

    # SFT phase: fine-tune a CausalLM on chosen responses, then use its backbone
    sft_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    sft_model.config.pad_token_id = tokenizer.pad_token_id
    optimizer_sft = torch.optim.AdamW(sft_model.parameters(), lr=5e-5)
    print("=== SFT Phase ===")
    sft_train_step(sft_model, loader, optimizer_sft, sft_epochs=3)

    base_model = sft_model.transformer  # GPT-2 backbone (outputs last_hidden_state)

    model = RewardModel(base_model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    model.train()
    all_losses = []
    all_accuracies = []
    best_val_acc = -1.0
    eval_every = 1000
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        for batch in loader:
            input_ids_chosen = batch["chosen_input_ids"].to(device)
            attention_mask_chosen = batch["chosen_attention_mask"].to(device)
            rewards_chosen = model(input_ids=input_ids_chosen, attention_mask=attention_mask_chosen) # shape (B,)

            input_ids_rejected = batch["rejected_input_ids"].to(device)
            attention_mask_rejected = batch["rejected_attention_mask"].to(device)
            rewards_rejected = model(input_ids=input_ids_rejected, attention_mask=attention_mask_rejected) # shape (B,)

            loss = -F.logsigmoid(rewards_chosen - rewards_rejected).mean() # loss formula for reward modeling
            acc = (rewards_chosen > rewards_rejected).float().mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_item = loss.item()
            all_losses.append(loss_item)
            all_accuracies.append(acc.item())

            step = len(all_losses)
            print(f"Batch {step}/{len(loader)*epochs} | Loss: {loss_item:.4f} | Batch acc: {acc.item():.4f}", end="\r")

            if step % eval_every == 0:
                val_acc = eval_accuracy(model, loader_eval)
                print(f"\nStep {step} eval accuracy: {val_acc:.4f}")
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), "reward_model.pt")
                    print(f"New best ({best_val_acc:.4f}) — saved to reward_model.pt")

    print(f"\nTraining complete. Best eval accuracy: {best_val_acc:.4f}")
    val_acc = best_val_acc

    # Plotting the loss curve and accuracy curve
    if all_losses:
        window = 100
        smoothed_loss = [
            sum(all_losses[max(0, i - window):i + 1]) / min(i + 1, window)
            for i in range(len(all_losses))
        ]
        smoothed_acc = [
            sum(all_accuracies[max(0, i - window):i + 1]) / min(i + 1, window)
            for i in range(len(all_accuracies))
        ]
        plt.figure()
        plt.plot(all_losses, alpha=0.3, label="Raw loss")
        plt.plot(smoothed_loss, label=f"Smoothed (window={window})")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training Loss vs Iterations")
        plt.legend()
        plt.grid(True)
        plt.savefig("loss_vs_iterations.png")

        plt.figure()
        plt.plot(all_accuracies, alpha=0.3, label="Batch accuracy (raw)")
        plt.plot(smoothed_acc, label=f"Batch accuracy (smoothed, window={window})")
        plt.axhline(y=val_acc, color="red", linestyle="--", label=f"Final eval accuracy: {val_acc:.4f}")
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        plt.title("Accuracy vs Iterations")
        plt.legend()
        plt.grid(True)
        plt.savefig("accuracy_vs_iterations.png")
        plt.show()
