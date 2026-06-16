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