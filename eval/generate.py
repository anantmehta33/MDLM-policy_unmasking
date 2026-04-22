import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import torch.distributed as dist


def _token_ids_to_digits(token_ids, digit_token_ids):
    digits = torch.zeros_like(token_ids)
    for idx, tok_id in enumerate(digit_token_ids):
        digits[token_ids == tok_id] = idx + 1
    return digits


def _count_duplicates_1d(values):
    counts = torch.bincount(values, minlength=5)
    dup = torch.clamp(counts[1:] - 1, min=0)
    return dup.sum()


def _sudoku_error_score_4x4(board_digits):
    bsz = board_digits.shape[0]
    scores = torch.zeros(bsz, device=board_digits.device, dtype=torch.float32)

    for b in range(bsz):
        grid = board_digits[b]
        score = 0.0
        for r in range(4):
            row = grid[r * 4 : (r + 1) * 4]
            score += _count_duplicates_1d(row).item()
        for c in range(4):
            col = torch.stack([grid[r * 4 + c] for r in range(4)], dim=0)
            score += _count_duplicates_1d(col).item()
        for sr in (0, 2):
            for sc in (0, 2):
                block = torch.stack(
                    [
                        grid[(sr + dr) * 4 + (sc + dc)]
                        for dr in range(2)
                        for dc in range(2)
                    ],
                    dim=0,
                )
                score += _count_duplicates_1d(block).item()
        scores[b] = score

    return scores


def _build_policy_state_features(
    logits,
    token_ids,
    digit_token_ids,
    t,
    total_steps,
    mask_id,
):
    """Build token_state and continuous features expected by Sudoku policy network."""
    # logits/token_ids are for generation segment only: [B, L, V], [B, L]
    digit_ids = torch.tensor(digit_token_ids, device=token_ids.device, dtype=torch.long)

    # token state: 0 for mask/other, 1..4 for recognized digit token ids
    token_state = torch.zeros_like(token_ids)
    for idx, tok_id in enumerate(digit_ids):
        token_state[token_ids == tok_id] = idx + 1
    token_state[token_ids == mask_id] = 0

    digit_logits = logits.index_select(dim=-1, index=digit_ids)  # [B, L, 4]
    probs = F.softmax(digit_logits, dim=-1)
    entropy = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1, keepdim=True)
    t_norm = torch.full_like(entropy, fill_value=float(t) / float(total_steps))
    cont_features = torch.cat([probs, entropy, t_norm], dim=-1)

    return token_state.long(), cont_features


def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    Using float16 for better performance while maintaining reasonable quality.
    """
    if temperature == 0.0:
        return logits  # Skip noise when temperature is 0

    # Use float32 instead of float64 for better performance
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits, dtype=torch.float32)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """
    Precompute the number of tokens to transition at each step.
    Optimized to be more efficient.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps

    # Create tensor once and modify in-place
    num_transfer_tokens = base.expand(-1, steps).clone()

    # Handle remainder more efficiently
    if remainder.sum() > 0:
        indices = torch.arange(steps, device=mask_index.device)
        mask = indices.unsqueeze(0) < remainder
        num_transfer_tokens[mask] += 1

    return num_transfer_tokens.to(torch.int64)


@torch.no_grad()
def generate(
    model,
    prompt,
    tokenizer,
    steps=64,
    gen_length=128,
    block_length=32,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    init_gen_ids=None,
    policy_model=None,
    policy_digit_token_ids=None,
    policy_given_mask=None,
    policy_sample_actions=False,
    policy_reward_guided=False,
    policy_reward_candidates=4,
):
    """
    Optimized version of the generate function.
    """
    # Use mixed precision for faster computation
    with torch.autocast(device_type="cuda"):
        x = torch.full(
            (prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=prompt.device
        )
        x[:, : prompt.shape[1]] = prompt.clone()

        if init_gen_ids is not None:
            if init_gen_ids.shape != (prompt.shape[0], gen_length):
                raise ValueError("init_gen_ids must have shape [batch, gen_length]")
            x[:, prompt.shape[1] : prompt.shape[1] + gen_length] = init_gen_ids.to(prompt.device)

        prompt_index = x != mask_id

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        steps_per_block = max(1, steps // num_blocks)
        for num_block in tqdm(range(num_blocks), disable=(dist.get_rank() != 0)):
            start_idx = prompt.shape[1] + num_block * block_length
            end_idx = prompt.shape[1] + (num_block + 1) * block_length

            block_mask_index = x[:, start_idx:end_idx] == mask_id
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

            for i in range(steps_per_block):
                mask_index = x == mask_id

                # Handle classifier-free guidance more efficiently
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)

                    # Get logits in a single forward pass
                    logits = model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(x).logits

                # Apply Gumbel noise for sampling
                logits_with_noise = add_gumbel_noise(logits, temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                # Policy-guided remasking over the generation segment.
                if remasking == "policy":
                    if policy_model is None or policy_digit_token_ids is None:
                        raise ValueError("policy remasking requires policy_model and policy_digit_token_ids")

                    gen_start = prompt.shape[1]
                    gen_end = prompt.shape[1] + gen_length

                    token_state, cont_features = _build_policy_state_features(
                        logits=logits[:, gen_start:gen_end, :],
                        token_ids=x[:, gen_start:gen_end],
                        digit_token_ids=policy_digit_token_ids,
                        t=i + 1,
                        total_steps=steps_per_block,
                        mask_id=mask_id,
                    )

                    keep_probs = policy_model(token_state, cont_features).clamp(1e-5, 1 - 1e-5)

                    if policy_given_mask is not None:
                        given = policy_given_mask.to(prompt.device)
                        if given.shape != keep_probs.shape:
                            raise ValueError("policy_given_mask must have shape [batch, gen_length]")
                        keep_probs = torch.where(given > 0, torch.ones_like(keep_probs), keep_probs)
                    else:
                        given = None

                    if policy_sample_actions:
                        actions_keep = torch.distributions.Bernoulli(probs=keep_probs).sample().long()
                    else:
                        actions_keep = (keep_probs >= 0.5).long()

                    x0 = torch.where(mask_index, x0, x)
                    pred_segment = x0[:, gen_start:gen_end]
                    next_segment = torch.where(
                        actions_keep > 0,
                        pred_segment,
                        torch.full_like(pred_segment, mask_id),
                    )

                    # Keep clue tokens fixed across steps when provided.
                    if given is not None:
                        curr_segment = x[:, gen_start:gen_end]
                        next_segment = torch.where(given > 0, curr_segment, next_segment)

                    if policy_reward_guided:
                        if policy_reward_candidates < 1:
                            raise ValueError("policy_reward_candidates must be >= 1")

                        best_next_segment = next_segment.clone()
                        best_scores = torch.full(
                            (x.shape[0],),
                            float("inf"),
                            device=x.device,
                            dtype=torch.float32,
                        )

                        candidate_actions = [actions_keep]
                        for _ in range(policy_reward_candidates - 1):
                            candidate_actions.append(torch.distributions.Bernoulli(probs=keep_probs).sample().long())

                        for cand_actions in candidate_actions:
                            cand_actions = cand_actions.clone()
                            if given is not None:
                                cand_actions[given > 0] = 1

                            cand_segment = torch.where(
                                cand_actions > 0,
                                pred_segment,
                                torch.full_like(pred_segment, mask_id),
                            )

                            if given is not None:
                                curr_segment = x[:, gen_start:gen_end]
                                cand_segment = torch.where(given > 0, curr_segment, cand_segment)

                            cand_x = x.clone()
                            cand_x[:, gen_start:gen_end] = cand_segment
                            next_logits = model(cand_x).logits[:, gen_start:gen_end, :]
                            next_pred_tokens = torch.argmax(next_logits, dim=-1)
                            next_pred_digits = _token_ids_to_digits(next_pred_tokens, policy_digit_token_ids)

                            if given is not None:
                                given_digits = _token_ids_to_digits(x[:, gen_start:gen_end], policy_digit_token_ids)
                                next_pred_digits = torch.where(given > 0, given_digits, next_pred_digits)

                            cand_scores = _sudoku_error_score_4x4(next_pred_digits)
                            better = cand_scores < best_scores
                            if better.any():
                                best_scores = torch.where(better, cand_scores, best_scores)
                                better_mask = better.unsqueeze(1).expand_as(best_next_segment)
                                best_next_segment = torch.where(better_mask, cand_segment, best_next_segment)

                        next_segment = best_next_segment

                    x[:, gen_start:gen_end] = next_segment
                    continue

                # Handle remasking strategy
                if remasking == "low_confidence":
                    # Use float32 instead of float64 for better performance
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                elif remasking == "random":
                    x0_p = torch.rand(x0.shape, device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                # Ensure we don't process tokens beyond the current block
                x0_p[:, end_idx:] = -np.inf

                # Update masked tokens
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=x0.device))

                # Select tokens to transfer based on confidence
                for j in range(confidence.shape[0]):
                    num_tokens = num_transfer_tokens[j, i].item()
                    if num_tokens > 0:
                        _, select_indices = torch.topk(confidence[j], k=num_tokens)
                        x[j, select_indices] = x0[j, select_indices]
        return x
