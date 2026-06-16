import torch

model_name = "gpt2"
PROMPT_MAX_LEN = 128
RESP_MAX_LEN = 256
batch_size = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
