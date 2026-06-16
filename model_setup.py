import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import model_name, device


def load_tokenizer(direction="right"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # GPT-2 does not define a padding token, so we reuse EOS for batching.
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = direction
    return tokenizer


def load_models(tokenizer):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name)

    model.config.pad_token_id = tokenizer.pad_token_id
    ref_model.config.pad_token_id = tokenizer.pad_token_id

    model.to(device)
    ref_model.to(device)

    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False  # Freeze reference model to save memory during training.

    return model, ref_model


def load_optimizers(model):
    optimizer_DPO = torch.optim.AdamW(model.parameters(), lr=5e-6)
    optimizer_SFT = torch.optim.AdamW(model.parameters(), lr=5e-5)
    return optimizer_DPO, optimizer_SFT
