import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from config import PROMPT_MAX_LEN, RESP_MAX_LEN, batch_size


def extract_prompt_and_response(text):
    marker = "\n\nAssistant:"
    last_idx = text.rfind(marker)
    if last_idx == -1:
        marker = "Assistant:"
        last_idx = text.rfind(marker) # data should be cut at the last occurence of the marker since there could be multiple turns
    if last_idx == -1:
        return text.strip(), "" # if the marker is not found, return the whole text as prompt and an empty response
    prompt = text[:last_idx].strip()
    response = text[last_idx + len(marker):].strip()
    return prompt, response


def preprocess(example):
    p1, c = extract_prompt_and_response(example["chosen"])
    p2, r = extract_prompt_and_response(example["rejected"])
    if p1 != p2:
        return {"prompt": None, "chosen": None, "rejected": None}
    return {"prompt": p1, "chosen": c, "rejected": r} # return the extracted prompt and response, or None if the prompt is not consistent between chosen and rejected


def _make_tokenize_fn(tokenizer): # create a function that tokenizes the prompt, chosen response, and rejected response for each example in the dataset
    def tokenize_example(example): # defining closure because tokenizer is not global and we want to avoid passing it around explicitly
        prompt_str = example["prompt"] + "\n\nAssistant:"
        chosen_str = " " + example["chosen"]
        rejected_str = " " + example["rejected"]

        prompt_ids   = tokenizer(prompt_str,   add_special_tokens=False)["input_ids"]
        chosen_ids   = tokenizer(chosen_str,   add_special_tokens=False)["input_ids"]
        rejected_ids = tokenizer(rejected_str, add_special_tokens=False)["input_ids"]

        chosen_ids   = chosen_ids[:RESP_MAX_LEN]
        rejected_ids = rejected_ids[:RESP_MAX_LEN]

        # Left-truncate the prompt to keep the most recent turns and the trailing cue.
        if len(prompt_ids) > PROMPT_MAX_LEN:
            prompt_ids = prompt_ids[-PROMPT_MAX_LEN:]
        prompt_len = len(prompt_ids)

        c_ids = prompt_ids + chosen_ids
        r_ids = prompt_ids + rejected_ids

        return {
            "chosen_input_ids":        c_ids,
            "chosen_attention_mask":   [1] * len(c_ids), # no padding yet, just a mask of 1s for the actual tokens
            "rejected_input_ids":      r_ids,
            "rejected_attention_mask": [1] * len(r_ids),
            "prompt_len":              prompt_len,
        }
    return tokenize_example


def _make_collate_fn(tokenizer): # create a function that pads the tokenized input ids and attention masks for a batch of examples, so that they can be fed into the model
    def collate_fn(batch):
        def pad_ids(key):
            return pad_sequence(
                [torch.tensor(b[key]) for b in batch], # pad_sequence expects a list of tensors, so we convert the lists of input ids into tensors first
                batch_first=True,
                padding_value=tokenizer.pad_token_id,
            )
        def pad_mask(key):
            return pad_sequence(
                [torch.tensor(b[key]) for b in batch],
                batch_first=True,
                padding_value=0,
            )
        return {
            "chosen_input_ids":        pad_ids("chosen_input_ids"),
            "chosen_attention_mask":   pad_mask("chosen_attention_mask"), # All padded
            "rejected_input_ids":      pad_ids("rejected_input_ids"),
            "rejected_attention_mask": pad_mask("rejected_attention_mask"),
            "prompt_len": torch.tensor([b["prompt_len"] for b in batch]),
        }
    return collate_fn


def get_dataloaders(tokenizer):
    dataset      = load_dataset("Anthropic/hh-rlhf", split="train")
    dataset_eval = load_dataset("Anthropic/hh-rlhf", split="test")

    dataset      = dataset.map(preprocess).filter(lambda x: x["prompt"] is not None) # filter out examples where the prompt was not consistent
    dataset_eval = dataset_eval.map(preprocess).filter(lambda x: x["prompt"] is not None)

    tokenize_example = _make_tokenize_fn(tokenizer)
    dataset      = dataset.map(tokenize_example)
    dataset_eval = dataset_eval.map(tokenize_example)

    collate_fn = _make_collate_fn(tokenizer)
    loader      = DataLoader(dataset,      batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    loader_eval = DataLoader(dataset_eval, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return loader, loader_eval
