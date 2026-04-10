import argparse
import csv
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


DIGITS = (1, 2, 3, 4)
GRID_SIZE = 4
SUBGRID_SIZE = 2
SEQ_LEN = 16
DEFAULT_MASK_ID = 126336

SUDOKU_SYSTEM_PROMPT = """
Please solve the following 4x4 Sudoku puzzle. The puzzle is provided as a 16-character string reading left-to-right, top-to-bottom, where '0' represents empty cells.

Rules:
- Fill empty cells with digits 1-4
- Each row must contain digits 1-4 exactly once
- Each column must contain digits 1-4 exactly once
- Each 2x2 box must contain digits 1-4 exactly once

Important: Respond with only the final 16-digit solution string.
""".strip()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_grid_str(grid_str: str) -> List[int]:
    grid_str = grid_str.strip()
    if len(grid_str) != SEQ_LEN:
        raise ValueError(f"Expected length {SEQ_LEN}, got {len(grid_str)}")
    vals = [int(ch) for ch in grid_str]
    if any(v < 0 or v > 4 for v in vals):
        raise ValueError("Grid must only contain digits 0-4")
    return vals


def tensor_grid_to_str(grid_tensor: torch.Tensor) -> str:
    return "".join(str(int(x)) for x in grid_tensor.tolist())


def count_duplicates(values: List[int]) -> int:
    counts: Dict[int, int] = {}
    for v in values:
        if v in DIGITS:
            counts[v] = counts.get(v, 0) + 1
    return sum(c - 1 for c in counts.values() if c > 1)


def error_score_4x4(board_tokens: torch.Tensor) -> torch.Tensor:
    bsz = board_tokens.shape[0]
    scores = torch.zeros(bsz, device=board_tokens.device, dtype=torch.float32)

    for b in range(bsz):
        grid = board_tokens[b].tolist()
        for r in range(GRID_SIZE):
            row = grid[r * GRID_SIZE : (r + 1) * GRID_SIZE]
            scores[b] += count_duplicates(row)
        for c in range(GRID_SIZE):
            col = [grid[r * GRID_SIZE + c] for r in range(GRID_SIZE)]
            scores[b] += count_duplicates(col)
        for sr in range(0, GRID_SIZE, SUBGRID_SIZE):
            for sc in range(0, GRID_SIZE, SUBGRID_SIZE):
                block = []
                for dr in range(SUBGRID_SIZE):
                    for dc in range(SUBGRID_SIZE):
                        rr = sr + dr
                        cc = sc + dc
                        block.append(grid[rr * GRID_SIZE + cc])
                scores[b] += count_duplicates(block)

    return scores


def empty_cell_accuracy(pred_board: torch.Tensor, gt_board: torch.Tensor, given_mask: torch.Tensor) -> torch.Tensor:
    """Accuracy only on puzzle-empty cells."""
    target_mask = (given_mask == 0)
    correct = ((pred_board == gt_board) & target_mask).sum(dim=-1).float()
    total = target_mask.sum(dim=-1).clamp(min=1).float()
    return correct / total


def discounted_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    t_steps, bsz = rewards.shape
    out = torch.zeros_like(rewards)
    running = torch.zeros(bsz, device=rewards.device)
    for t in reversed(range(t_steps)):
        running = rewards[t] + gamma * running
        out[t] = running
    return out


def detect_default_train_csv(repo_root: str) -> str:
    candidates = [
        os.path.join(repo_root, "dataset", "4x4_train_sudoku.csv"),
        os.path.join(repo_root, "dataset", "4x4_train_sudoku,csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


class SudokuCSVDataset(Dataset):
    def __init__(self, csv_path: str):
        self.samples: List[Tuple[str, str]] = []
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                puzzle = row["Puzzle"].strip()
                solution = row["Solution"].strip()
                if len(puzzle) == SEQ_LEN and len(solution) == SEQ_LEN:
                    self.samples.append((puzzle, solution))

        if not self.samples:
            raise RuntimeError(f"No valid samples loaded from {csv_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        puzzle, solution = self.samples[idx]
        puzzle_vals = torch.tensor(parse_grid_str(puzzle), dtype=torch.long)
        solution_vals = torch.tensor(parse_grid_str(solution), dtype=torch.long)
        given_mask = (puzzle_vals != 0).long()
        return {
            "puzzle": puzzle_vals,
            "solution": solution_vals,
            "given_mask": given_mask,
        }


@dataclass
class PolicyConfig:
    token_vocab_size: int = 5
    token_emb_dim: int = 32
    feature_dim: int = 6
    hidden_dim: int = 128
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.1


class SudokuPolicyNetwork(nn.Module):
    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg.token_vocab_size, cfg.token_emb_dim)
        self.input_proj = nn.Linear(cfg.token_emb_dim + cfg.feature_dim, cfg.hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.nhead,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)
        self.keep_head = nn.Linear(cfg.hidden_dim, 1)

    def forward(self, token_state: torch.Tensor, cont_features: torch.Tensor) -> torch.Tensor:
        token_emb = self.token_embedding(token_state)
        x = torch.cat([token_emb, cont_features], dim=-1)
        x = self.input_proj(x)
        x = self.encoder(x)
        return torch.sigmoid(self.keep_head(x).squeeze(-1))


def _single_token_id(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"Expected single token for {text!r}, got {ids}")
    return ids[0]


class LladaSudokuAdapter:
    """Frozen LLaDA adapter returning logits over digits 1..4 for 16 Sudoku positions."""

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        torch_dtype: torch.dtype,
        max_prompt_cache: int = 8192,
        mask_id: int = None,
    ):
        self.device = device
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        ).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.mask_id = mask_id if mask_id is not None else getattr(self.model.config, "mask_token_id", None)
        if self.mask_id is None:
            self.mask_id = self.tokenizer.mask_token_id
        if self.mask_id is None:
            self.mask_id = DEFAULT_MASK_ID

        # The model uses token IDs; these IDs represent Sudoku digit values for all board positions.
        self.digit_token_ids = torch.tensor(
            [_single_token_id(self.tokenizer, str(d)) for d in DIGITS],
            dtype=torch.long,
            device=self.device,
        )

        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = self.tokenizer.eos_token_id
        if self.pad_id is None:
            self.pad_id = 0

        self.answer_prefix_ids = self.tokenizer.encode("<answer>\n", add_special_tokens=False)
        self.prompt_cache: Dict[str, List[int]] = {}
        self.max_prompt_cache = max_prompt_cache

        # Mapping 0..4 -> token id where 0 is mask.
        self.value_to_token = torch.full((5,), self.mask_id, dtype=torch.long, device=self.device)
        for i, d in enumerate(DIGITS):
            self.value_to_token[d] = self.digit_token_ids[i]

    def _build_prompt_ids(self, puzzle_str: str) -> List[int]:
        if puzzle_str in self.prompt_cache:
            return self.prompt_cache[puzzle_str]

        question = f"Solve the following Sudoku puzzle: {puzzle_str}\n"
        user_content = f"{SUDOKU_SYSTEM_PROMPT}\n\n{question}"
        messages = [{"role": "user", "content": user_content}]

        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
            )
        else:
            prompt_ids = self.tokenizer(user_content, add_special_tokens=True).input_ids

        prompt_ids = prompt_ids + self.answer_prefix_ids
        if len(self.prompt_cache) >= self.max_prompt_cache:
            self.prompt_cache.clear()
        self.prompt_cache[puzzle_str] = prompt_ids
        return prompt_ids

    @torch.no_grad()
    def get_logits(
        self,
        current_tokens: torch.Tensor,
        puzzle_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        current_tokens: [B,16] with values in {0..4}, where 0 means masked.
        puzzle_tokens: [B,16] original puzzle values for prompt construction.
        Returns logits_4: [B,16,4] corresponding to digits 1..4.
        """
        bsz = current_tokens.shape[0]

        board_token_ids = self.value_to_token[current_tokens]  # [B,16]

        prompt_lists: List[List[int]] = []
        prompt_lens: List[int] = []
        for b in range(bsz):
            puzzle_str = tensor_grid_to_str(puzzle_tokens[b].detach().cpu())
            ids = self._build_prompt_ids(puzzle_str)
            prompt_lists.append(ids)
            prompt_lens.append(len(ids))

        total_lens = [prompt_lens[b] + SEQ_LEN for b in range(bsz)]
        max_len = max(total_lens)

        input_ids = torch.full((bsz, max_len), self.pad_id, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long, device=self.device)

        for b in range(bsz):
            p_ids = torch.tensor(prompt_lists[b], dtype=torch.long, device=self.device)
            p_len = p_ids.shape[0]
            end = p_len + SEQ_LEN
            input_ids[b, :p_len] = p_ids
            input_ids[b, p_len:end] = board_token_ids[b]
            attention_mask[b, :end] = 1

        with torch.autocast(device_type="cuda", enabled=(self.device.type == "cuda"), dtype=torch.bfloat16):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [B,L,V]

        out_logits = []
        for b in range(bsz):
            p_len = prompt_lens[b]
            step_logits = logits[b, p_len : p_len + SEQ_LEN, :]
            step_logits = step_logits[:, self.digit_token_ids]  # [16,4]
            out_logits.append(step_logits)

        return torch.stack(out_logits, dim=0)


def build_state_features(
    current_tokens: torch.Tensor,
    logits_4: torch.Tensor,
    t: int,
    total_steps: int,
) -> torch.Tensor:
    probs = F.softmax(logits_4, dim=-1)
    entropy = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1, keepdim=True)
    t_norm = torch.full_like(entropy, fill_value=float(t) / float(total_steps))
    return torch.cat([probs, entropy, t_norm], dim=-1)


def argmax_board_from_logits(logits_4: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits_4, dim=-1) + 1


def sample_actions_and_logprob(keep_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    dist = torch.distributions.Bernoulli(probs=keep_probs.clamp(1e-5, 1 - 1e-5))
    actions = dist.sample()
    log_prob = dist.log_prob(actions).sum(dim=-1)
    return actions.long(), log_prob


def apply_actions(next_pred_tokens: torch.Tensor, actions_keep: torch.Tensor) -> torch.Tensor:
    return torch.where(actions_keep > 0, next_pred_tokens, torch.zeros_like(next_pred_tokens))


def enforce_given_clues(
    keep_probs: torch.Tensor,
    given_mask: torch.Tensor,
    current_tokens: torch.Tensor,
    puzzle_tokens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    keep_probs = keep_probs.clone()
    keep_probs[given_mask == 1] = 1.0

    current_tokens = current_tokens.clone()
    current_tokens[given_mask == 1] = puzzle_tokens[given_mask == 1]
    return keep_probs, current_tokens


def rollout_episode(
    policy: SudokuPolicyNetwork,
    llada: LladaSudokuAdapter,
    puzzle: torch.Tensor,
    solution: torch.Tensor,
    given_mask: torch.Tensor,
    steps_per_episode: int,
    sample_actions: bool,
) -> Dict[str, torch.Tensor]:
    current_tokens = torch.where(given_mask == 1, puzzle, torch.zeros_like(puzzle))
    step_log_probs = []
    step_rewards = []

    prev_error = None
    last_xhat0 = None
    keep_ratios = []

    for t in range(1, steps_per_episode + 1):
        logits_4 = llada.get_logits(current_tokens=current_tokens, puzzle_tokens=puzzle)
        state_features = build_state_features(current_tokens, logits_4, t=t, total_steps=steps_per_episode)

        keep_probs = policy(current_tokens, state_features)
        keep_probs, current_tokens = enforce_given_clues(keep_probs, given_mask, current_tokens, puzzle)

        if sample_actions:
            actions_keep, log_prob = sample_actions_and_logprob(keep_probs)
        else:
            actions_keep = (keep_probs >= 0.5).long()
            dist = torch.distributions.Bernoulli(probs=keep_probs.clamp(1e-5, 1 - 1e-5))
            log_prob = dist.log_prob(actions_keep.float()).sum(dim=-1)

        actions_keep[given_mask == 1] = 1
        keep_ratios.append(actions_keep.float().mean(dim=-1))

        xhat0 = argmax_board_from_logits(logits_4)
        xhat0[given_mask == 1] = puzzle[given_mask == 1]

        current_error = error_score_4x4(xhat0)
        if prev_error is None:
            reward_t = torch.zeros_like(current_error)
        else:
            reward_t = prev_error - current_error

        prev_error = current_error
        last_xhat0 = xhat0
        current_tokens = apply_actions(xhat0, actions_keep)
        current_tokens[given_mask == 1] = puzzle[given_mask == 1]

        step_log_probs.append(log_prob)
        step_rewards.append(reward_t)

    rewards = torch.stack(step_rewards, dim=0)
    log_probs = torch.stack(step_log_probs, dim=0)
    keep_ratio = torch.stack(keep_ratios, dim=0).mean(dim=0)

    final_acc = empty_cell_accuracy(last_xhat0, solution, given_mask)
    final_err = error_score_4x4(last_xhat0)
    exact_match = (last_xhat0 == solution).all(dim=-1).float()

    return {
        "rewards": rewards,
        "log_probs": log_probs,
        "final_acc": final_acc,
        "final_err": final_err,
        "exact_match": exact_match,
        "keep_ratio": keep_ratio,
    }


def train_one_epoch(
    policy: SudokuPolicyNetwork,
    llada: LladaSudokuAdapter,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    steps_per_episode: int,
    gamma: float,
    baseline_momentum: float,
    baseline_value: float,
) -> Tuple[float, float, float, float, float, float]:
    policy.train()
    epoch_losses = []
    episode_returns = []
    keep_ratios = []
    final_errs = []
    final_accs = []
    exact_rates = []

    for batch in loader:
        puzzle = batch["puzzle"].to(device)
        solution = batch["solution"].to(device)
        given_mask = batch["given_mask"].to(device)

        rollout = rollout_episode(
            policy=policy,
            llada=llada,
            puzzle=puzzle,
            solution=solution,
            given_mask=given_mask,
            steps_per_episode=steps_per_episode,
            sample_actions=True,
        )

        returns = discounted_returns(rollout["rewards"], gamma=gamma)
        batch_return_mean = returns[0].mean().item()
        baseline_value = baseline_momentum * baseline_value + (1.0 - baseline_momentum) * batch_return_mean

        advantage = returns - baseline_value
        loss = -(rollout["log_probs"] * advantage.detach()).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_losses.append(loss.item())
        episode_returns.append(batch_return_mean)
        keep_ratios.extend(rollout["keep_ratio"].detach().cpu().tolist())
        final_errs.extend(rollout["final_err"].detach().cpu().tolist())
        final_accs.extend(rollout["final_acc"].detach().cpu().tolist())
        exact_rates.extend(rollout["exact_match"].detach().cpu().tolist())

    return (
        float(np.mean(epoch_losses)),
        float(np.mean(episode_returns)),
        float(np.mean(keep_ratios) if keep_ratios else 0.0),
        float(np.mean(final_errs) if final_errs else 0.0),
        float(np.mean(final_accs) if final_accs else 0.0),
        float(np.mean(exact_rates) if exact_rates else 0.0),
    )


@torch.no_grad()
def evaluate_policy(
    policy: SudokuPolicyNetwork,
    llada: LladaSudokuAdapter,
    loader: DataLoader,
    device: torch.device,
    steps_per_episode: int,
    sample_actions: bool,
) -> Dict[str, float]:
    policy.eval()

    all_err = []
    all_acc = []
    all_exact = []
    all_keep = []

    for batch in loader:
        puzzle = batch["puzzle"].to(device)
        solution = batch["solution"].to(device)
        given_mask = batch["given_mask"].to(device)

        rollout = rollout_episode(
            policy=policy,
            llada=llada,
            puzzle=puzzle,
            solution=solution,
            given_mask=given_mask,
            steps_per_episode=steps_per_episode,
            sample_actions=sample_actions,
        )

        all_err.extend(rollout["final_err"].detach().cpu().tolist())
        all_acc.extend(rollout["final_acc"].detach().cpu().tolist())
        all_exact.extend(rollout["exact_match"].detach().cpu().tolist())
        all_keep.extend(rollout["keep_ratio"].detach().cpu().tolist())

    return {
        "mean_error": float(np.mean(all_err) if all_err else 0.0),
        "empty_cell_acc": float(np.mean(all_acc) if all_acc else 0.0),
        "exact_match_rate": float(np.mean(all_exact) if all_exact else 0.0),
        "mean_keep_ratio": float(np.mean(all_keep) if all_keep else 0.0),
    }


def main() -> None:
    print("entered main")
    parser = argparse.ArgumentParser()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--train_csv", type=str, default=detect_default_train_csv(repo_root))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--reverse_steps", type=int, default=16)
    parser.add_argument("--baseline_momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--save_dir", type=str, default=os.path.join(repo_root, "eval", "policy_sudoku", "checkpoints"))
    parser.add_argument("--mask_id", type=int, default=DEFAULT_MASK_ID)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    if not os.path.exists(args.train_csv):
        raise FileNotFoundError(f"Train CSV not found: {args.train_csv}")

    os.makedirs(args.save_dir, exist_ok=True)

    train_ds = SudokuCSVDataset(args.train_csv)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    print("created train loader")
    policy_cfg = PolicyConfig()
    policy = SudokuPolicyNetwork(policy_cfg).to(device)
    llada = LladaSudokuAdapter(
        model_name=args.model_name,
        device=device,
        torch_dtype=torch.bfloat16,
        mask_id=args.mask_id,
    )
    print("initialized model and adapter")
    optimizer = AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    baseline_value = 0.0
    best_metric = -1e9
    best_ckpt = os.path.join(args.save_dir, "policy_best.pt")

    for epoch in range(1, args.epochs + 1):
        (
            train_loss,
            train_ret,
            train_keep,
            train_err,
            train_empty_acc,
            train_exact,
        ) = train_one_epoch(
            policy=policy,
            llada=llada,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            steps_per_episode=args.reverse_steps,
            gamma=args.gamma,
            baseline_momentum=args.baseline_momentum,
            baseline_value=baseline_value,
        )
        baseline_value = args.baseline_momentum * baseline_value + (1.0 - args.baseline_momentum) * train_ret

        train_metrics = {
            "mean_error": train_err,
            "empty_cell_acc": train_empty_acc,
            "exact_match_rate": train_exact,
            "mean_keep_ratio": train_keep,
        }

        print(
            f"[Epoch {epoch}] loss={train_loss:.4f} return={train_ret:.4f} keep={train_keep:.4f} "
            f"train_error={train_metrics['mean_error']:.4f} "
            f"train_empty_acc={train_metrics['empty_cell_acc']:.4f} "
            f"train_exact={train_metrics['exact_match_rate']:.4f}"
        )

        metric = train_metrics["exact_match_rate"] - 0.01 * train_metrics["mean_error"]
        if metric > best_metric:
            best_metric = metric
            torch.save(
                {
                    "epoch": epoch,
                    "policy_state_dict": policy.state_dict(),
                    "policy_config": policy_cfg.__dict__,
                    "args": vars(args),
                    "train_metrics": train_metrics,
                    "digit_token_ids": llada.digit_token_ids.detach().cpu().tolist(),
                    "mask_id": llada.mask_id,
                    "model_name": args.model_name,
                },
                best_ckpt,
            )
            print(f"Saved best checkpoint to {best_ckpt}")

    print(f"Training complete. Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
