# RLHF-DPO

A from-scratch RLHF training project focused on Direct Preference Optimization (DPO), reward modeling, and optimization behavior in preference learning.

This project implements core RLHF components without relying on high-level RLHF libraries, with the goal of understanding the full training pipeline from token-level log-probabilities to preference-based optimization.

---

## Highlights

- Implemented DPO from scratch on Anthropic HH-RLHF
- Built custom sequence log-probability computation with correct prompt/response masking
- Implemented SFT + DPO training pipeline
- Compared raw DPO and length-normalized DPO
- Investigated variance and optimization stability in DPO training
- Added GPU-accelerated batched training and evaluation

---

## Preliminary Results

| Experiment | Preference Accuracy |
|---|---|
| Raw DPO (GPT-2) | ~0.57 |
| Length-Normalized DPO | ~0.63 |

Initial experiments also showed a large reduction in DPO margin variance after length normalization:

- Raw DPO variance: ~20,000+
- Length-normalized DPO variance: ~85

These results are preliminary and will be expanded with larger models and additional experiments.

---

## Code Structure

- `config.py` — experiment configuration
- `data.py` — HH-RLHF preprocessing and dataloaders
- `model_setup.py` — tokenizer/model setup
- `logprobs.py` — sequence log-probability computation
- `sft.py` — supervised fine-tuning
- `dpo.py` — DPO objective and training loop
- `evaluate.py` — preference accuracy and margin evaluation
- `main.py` — training entry point
- `dpo_gathered.py` — DPO implementation in a single file for debugging and experimentation

---

## GRPO Project Quick Start

This project only requires the GRPO pipeline.

### 1. Clone the repository

```bash
git clone https://github.com/fzhu0628/LLM-RLHF.git
cd LLM-RLHF
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

If any package is missing, install it manually with `pip`.

### 3. Train the Reward Model (only once)

Run:

```bash
python reward_model.py
```

This trains a GPT-2 reward model and saves:

```text
reward_model.pt
```

GRPO loads this file automatically.

### 4. Run GRPO

Simply execute:

```bash
python main_grpo.py
```

The script will automatically:

- Load the tokenizer and GPT-2
- Load `reward_model.pt`
- Perform supervised fine-tuning (SFT)
- Copy the SFT model to create the frozen reference model
- Run GRPO training
- Evaluate the final model

No other scripts need to be run manually.

### 5. Files You Will Likely Modify

- `main_grpo.py` — main training pipeline
- `ppo.py` — core GRPO algorithm; modify this file to change the optimization algorithm
- `data_ppo.py` — data loading for GRPO
- `reward_model.py` — reward model training; normally does not need modification unless changing the reward model
- `config.py` — hyperparameters such as learning rate, batch size, epochs, and group size

### 6. Recommended Workflow

1. Verify that the original code runs successfully.
2. Create a copy of `main_grpo.py`, such as `main_my_method.py`.
3. Implement your idea without modifying the original implementation.
4. Compare the results against the baseline.

If you encounter a runtime error, copy the complete error message before making changes.

---

## Current Experiments

- Raw DPO vs length-normalized DPO
- SFT vs no-SFT initialization
- Preference accuracy and margin statistics
- Loss dynamics and variance diagnostics
- Generation behavior under different training setups

---

## Key Concepts Explored

- Sequence-level log-probabilities
- Preference optimization
- Pairwise ranking objectives
- RLHF training pipelines
- Variance behavior in DPO
- Policy-gradient analogies for preference optimization

---

## Next Steps

- Train an explicit reward model
- Implement PPO-style RLHF
- Add LoRA-based larger-model experiments
- Run systematic variance/stability ablations
- Improve generation-quality evaluation
- Explore additional preference optimization methods

---

## Tech Stack

- PyTorch
- HuggingFace Transformers
- HuggingFace Datasets
- CUDA

---

## Status

Active summer project focused on developing deeper understanding of LLM training, RLHF, and preference optimization. The codebase is still evolving and more components will be added over time.
