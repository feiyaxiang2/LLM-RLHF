from config import model_name, device
import torch
from logprobs import sequence_logprobs_ppo
from evaluate import evaluate

def grpo(model, ref_model, loader, loader_eval, optimizer, tokenizer, reward_model, epochs=1, m_train=4, epsilon=0.2, group_size=4, beta_kl=0.01):
    all_losses = []
    all_accuracies = []
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        for batch_idx, batch in enumerate(loader): 
            model.eval()
            input_ids = batch["prompt_ids"].to(device) # shape (B, seq_len)
            attention_mask = batch["mask"].to(device) # shape (B, seq_len)
            prompt_len = batch["prompt_len"].to(device) # shape (B,) - this tells us how many tokens in the input_ids correspond to the prompt, so we can calculate logprobs only for the prompt tokens and not the response tokens
            prompt_len_expanded = prompt_len.repeat_interleave(group_size)  # (group_size * B,)

            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=60,
                    do_sample=True,
                    num_return_sequences=group_size,
                    pad_token_id=tokenizer.pad_token_id,
                ) # shape (B * group_size, seq_len + gen_len)

            generated_attention_mask = (generated_ids != tokenizer.pad_token_id).long() # shape (B * group_size, seq_len + gen_len)
            '''At this point, we have generated multiple responses for each prompt in the batch'''

            with torch.no_grad():
                logprobs_old, _ = sequence_logprobs_ppo(model, generated_ids, generated_attention_mask, prompt_len_expanded) # shape (B * group_size, seq_len + gen_len - 1)
                rewards = reward_model(input_ids=generated_ids, attention_mask=generated_attention_mask).squeeze(-1) # shape (B * group_size,)

                logprobs_ref, _ = sequence_logprobs_ppo(ref_model, generated_ids, generated_attention_mask, prompt_len_expanded) # shape (B * group_size, seq_len + gen_len - 1)
            
            B = input_ids.size(0)

            rewards = rewards.view(B, group_size)  # (B, G)
            mean = rewards.mean(dim=1, keepdim=True)
            std = rewards.std(dim=1, keepdim=True).clamp(min=1e-8) # size (B, 1)
            # print(f"Reward std: {std.mean().item():.4f}")

            advantages = ((rewards - mean) / std).view(-1)  # (B*G,)
            
            for _ in range(m_train):
                logprobs_new, final_mask_new = sequence_logprobs_ppo(model, generated_ids, generated_attention_mask, prompt_len_expanded) # shape (B * group_size, seq_len + gen_len - 1)

                ratio = torch.exp(logprobs_new - logprobs_old) # shape (B * group_size, seq_len + gen_len - 1)
                clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) # clip the ratio to be between 1 - epsilon and 1 + epsilon

                # loss_grpo = -torch.min(ratio * advantages.unsqueeze(1), clipped_ratio * advantages.unsqueeze(1))
                # loss_grpo = (loss_grpo * final_mask_new).sum() / final_mask_new.sum().clamp(min=1)
                # loss_grpo = loss_grpo.mean() # average over the batch

                loss_grpo_token = -torch.min(
                    ratio * advantages.unsqueeze(1),
                    clipped_ratio * advantages.unsqueeze(1)
                )

                # Average over tokens within each completion first
                per_completion_loss_grpo = (
                    loss_grpo_token * final_mask_new
                ).sum(dim=1) / final_mask_new.sum(dim=1).clamp(min=1)

                # Then average over completions equally
                loss_grpo = per_completion_loss_grpo.mean()

                log_ratio = logprobs_ref - logprobs_new

                loss_kl_token = torch.exp(log_ratio) - log_ratio - 1.0

                per_completion_loss_kl = (
                    loss_kl_token * final_mask_new
                ).sum(dim=1) / final_mask_new.sum(dim=1).clamp(min=1)

                loss_kl = per_completion_loss_kl.mean()

                
                # loss_kl = torch.exp(log_ratio)  - log_ratio - 1.0       
                # loss_kl = (loss_kl * final_mask_new).sum() / final_mask_new.sum().clamp(min=1)
                # loss_kl = loss_kl.mean() # average over the batch

                loss = loss_grpo + beta_kl * loss_kl

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                all_losses.append(loss.item())
                print(f"Batch {batch_idx+1} / {len(loader)} | GRPO Loss: {loss.item():.4f} | GRPO Component: {loss_grpo.item():.4f} | KL Component: {loss_kl.item():.4f}")
            
            if (batch_idx + 1) % 2000 == 0:
                print(f"\nEvaluating at batch {batch_idx+1}...")
                acc_model, acc_ref = evaluate(model, ref_model, loader_eval)
                print(f"\nModel Accuracy: {acc_model:.4f} | Reference Accuracy: {acc_ref:.4f}")
                all_accuracies.append((batch_idx+1, acc_model, acc_ref))

                # with torch.no_grad():
                #     # Generate responses from current model and reference model on the same fixed prompts
                #     policy_outputs = model.generate(
                #         input_ids=input_ids,
                #         attention_mask=attention_mask,
                #         max_new_tokens=60,
                #         do_sample=True,
                #         pad_token_id=tokenizer.pad_token_id,
                #     )

                #     ref_outputs = ref_model.generate(
                #         input_ids=input_ids,
                #         attention_mask=attention_mask,
                #         max_new_tokens=60,
                #         do_sample=True,
                #         pad_token_id=tokenizer.pad_token_id,
                #     )

                #     policy_mask = (policy_outputs != tokenizer.pad_token_id).long()
                #     ref_mask = (ref_outputs != tokenizer.pad_token_id).long()

                #     policy_rewards = reward_model(
                #         input_ids=policy_outputs,
                #         attention_mask=policy_mask,
                #     ).squeeze(-1)

                #     ref_rewards = reward_model(
                #         input_ids=ref_outputs,
                #         attention_mask=ref_mask,
                #     ).squeeze(-1)

                #     print("policy reward mean:", policy_rewards.mean().item())
                #     print("ref reward mean:", ref_rewards.mean().item())
                #     print("RM win rate vs ref:", (policy_rewards > ref_rewards).float().mean().item())

    return all_losses, all_accuracies