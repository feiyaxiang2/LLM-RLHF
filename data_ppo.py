import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from config import PROMPT_MAX_LEN, batch_size


def extract_prompt(text):
    marker = "\n\nAssistant:"
    last_idx = text.rfind(marker)
    if last_idx == -1:
        marker = "Assistant:"
        last_idx = text.rfind(marker) # data should be cut at the last occurence of the marker since there could be multiple turns
    if last_idx == -1:
        return text.strip(), "" # if the marker is not found, return the whole text as prompt and an empty response
    prompt = text[:last_idx].strip()
    return prompt


def preprocess(example):
    p = extract_prompt(example["chosen"])
    return {"prompt": p} # return the extracted prompt and responses, or None if the prompt is not consistent between chosen and rejected


def _make_tokenize_fn(tokenizer): # create a function that tokenizes the prompt, chosen response, and rejected response for each example in the dataset
    def tokenize_example(example): # defining closure because tokenizer is not global and we want to avoid passing it around explicitly
        prompt_str = example["prompt"] + "\n\nAssistant:"

        prompt_ids   = tokenizer(prompt_str,   add_special_tokens=False)["input_ids"]

        # Left-truncate the prompt to keep the most recent turns and the trailing cue.
        if len(prompt_ids) > PROMPT_MAX_LEN:
            prompt_ids = prompt_ids[-PROMPT_MAX_LEN:]
        prompt_len = len(prompt_ids)

        return {
            "prompt_ids":        prompt_ids,
            "mask":   [1] * len(prompt_ids), # no padding yet, just a mask of 1s for the actual tokens
            "prompt_len":        prompt_len,
        }
    return tokenize_example


def _make_collate_fn(tokenizer): # create a function that pads the tokenized input ids and attention masks for a batch of examples, so that they can be fed into the model
    def collate_fn(batch):
        def left_pad(key, pad_value):
            seqs = [torch.tensor(b[key]) for b in batch]
            max_len = max(s.size(0) for s in seqs)
            out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
            for i, s in enumerate(seqs):
                out[i, max_len - s.size(0):] = s  # right-align = left-pad
            return out

        return {
            "prompt_ids":  left_pad("prompt_ids", tokenizer.pad_token_id),
            "mask":        left_pad("mask", 0),
            "prompt_len":  torch.tensor([b["prompt_len"] for b in batch]),
        }
    return collate_fn


def get_dataloaders_ppo(tokenizer):
    dataset      = load_dataset("Anthropic/hh-rlhf", split="train")
    dataset_eval = load_dataset("Anthropic/hh-rlhf", split="test")

    dataset      = dataset.map(preprocess)
    dataset_eval = dataset_eval.map(preprocess)

    tokenize_example = _make_tokenize_fn(tokenizer)
    dataset      = dataset.map(tokenize_example)
    dataset_eval = dataset_eval.map(tokenize_example)

    collate_fn = _make_collate_fn(tokenizer)
    loader      = DataLoader(dataset,      batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    loader_eval = DataLoader(dataset_eval, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return loader, loader_eval
