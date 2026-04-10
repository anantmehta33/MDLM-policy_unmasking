import argparse
import csv
import json
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from train_policy_sudoku import (
    DEFAULT_MASK_ID,
    PolicyConfig,
    SudokuCSVDataset,
    SudokuPolicyNetwork,
    LladaSudokuAdapter,
    argmax_board_from_logits,
    apply_actions,
    build_state_features,
    empty_cell_accuracy,
    enforce_given_clues,
    error_score_4x4,
    sample_actions_and_logprob,
    set_seed,
)


def evaluate_batch(
    policy: SudokuPolicyNetwork,
    llada: LladaSudokuAdapter,
    puzzle: torch.Tensor,
    solution: torch.Tensor,
    given_mask: torch.Tensor,
    reverse_steps: int,
    sample_actions: bool,
) -> Dict[str, torch.Tensor]:
    current_tokens = torch.where(given_mask == 1, puzzle, torch.zeros_like(puzzle))
    last_xhat0 = None

    with torch.no_grad():
        for t in range(1, reverse_steps + 1):
            logits_4 = llada.get_logits(current_tokens=current_tokens, puzzle_tokens=puzzle)
            state_features = build_state_features(current_tokens, logits_4, t=t, total_steps=reverse_steps)

            keep_probs = policy(current_tokens, state_features)
            keep_probs, current_tokens = enforce_given_clues(keep_probs, given_mask, current_tokens, puzzle)

            if sample_actions:
                actions_keep, _ = sample_actions_and_logprob(keep_probs)
            else:
                actions_keep = (keep_probs >= 0.5).long()

            actions_keep[given_mask == 1] = 1

            xhat0 = argmax_board_from_logits(logits_4)
            xhat0[given_mask == 1] = puzzle[given_mask == 1]

            current_tokens = apply_actions(xhat0, actions_keep)
            current_tokens[given_mask == 1] = puzzle[given_mask == 1]
            last_xhat0 = xhat0

    final_acc = empty_cell_accuracy(last_xhat0, solution, given_mask)
    final_err = error_score_4x4(last_xhat0)
    exact_match = (last_xhat0 == solution).all(dim=-1).float()

    return {
        "pred": last_xhat0,
        "final_acc": final_acc,
        "final_err": final_err,
        "exact_match": exact_match,
    }


def tensor_to_grid_str(row: torch.Tensor) -> str:
    return "".join(str(int(x)) for x in row.tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--reverse_steps", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--mask_id", type=int, default=DEFAULT_MASK_ID)
    parser.add_argument("--sample_actions", action="store_true")
    parser.add_argument("--save_json", type=str, default="")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = PolicyConfig(**ckpt["policy_config"])

    policy = SudokuPolicyNetwork(cfg).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()

    llada_model_name = ckpt.get("model_name", args.model_name)
    llada_mask_id = ckpt.get("mask_id", args.mask_id)

    llada = LladaSudokuAdapter(
        model_name=llada_model_name,
        device=device,
        torch_dtype=torch.bfloat16,
        mask_id=llada_mask_id,
    )

    dataset = SudokuCSVDataset(args.test_csv)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    total_examples = 0
    total_exact = 0.0
    total_empty_acc = 0.0
    total_error = 0.0
    total_correct_cells = 0.0
    total_empty_cells = 0.0
    per_item: List[Dict[str, object]] = []

    for batch in loader:
        puzzle = batch["puzzle"].to(device)
        solution = batch["solution"].to(device)
        given_mask = batch["given_mask"].to(device)

        out = evaluate_batch(
            policy=policy,
            llada=llada,
            puzzle=puzzle,
            solution=solution,
            given_mask=given_mask,
            reverse_steps=args.reverse_steps,
            sample_actions=args.sample_actions,
        )

        bsz = puzzle.shape[0]
        total_examples += bsz
        total_exact += out["exact_match"].sum().item()
        total_empty_acc += out["final_acc"].sum().item()
        total_error += out["final_err"].sum().item()

        empty_mask = (given_mask == 0)
        batch_correct_cells = ((out["pred"] == solution) & empty_mask).sum().item()
        batch_empty_cells = empty_mask.sum().item()
        total_correct_cells += batch_correct_cells
        total_empty_cells += batch_empty_cells

        pred_cpu = out["pred"].cpu()
        puzzle_cpu = puzzle.cpu()
        solution_cpu = solution.cpu()
        acc_cpu = out["final_acc"].cpu()
        exact_cpu = out["exact_match"].cpu()
        err_cpu = out["final_err"].cpu()

        for i in range(bsz):
            per_item.append(
                {
                    "puzzle": tensor_to_grid_str(puzzle_cpu[i]),
                    "prediction": tensor_to_grid_str(pred_cpu[i]),
                    "solution": tensor_to_grid_str(solution_cpu[i]),
                    "empty_cell_accuracy": float(acc_cpu[i].item()),
                    "exact_match": bool(exact_cpu[i].item()),
                    "error_score": float(err_cpu[i].item()),
                }
            )

    exact_acc = (total_exact / total_examples) if total_examples > 0 else 0.0
    mean_empty_acc = (total_empty_acc / total_examples) if total_examples > 0 else 0.0
    mean_error = (total_error / total_examples) if total_examples > 0 else 0.0
    aggregate_empty_acc = (total_correct_cells / total_empty_cells) if total_empty_cells > 0 else 0.0

    print("=" * 80)
    print("Checkpoint Evaluation (Sudoku)")
    print(f"checkpoint: {args.checkpoint}")
    print(f"csv:        {args.test_csv}")
    print(f"examples:   {total_examples}")
    print(f"reverse_steps: {args.reverse_steps}")
    print(f"sample_actions: {args.sample_actions}")
    print("-" * 80)
    print(f"Exact Match Accuracy:        {100.0 * exact_acc:.2f}%")
    print(f"Mean Empty-Cell Accuracy:    {100.0 * mean_empty_acc:.2f}%")
    print(f"Aggregate Empty-Cell Acc:    {100.0 * aggregate_empty_acc:.2f}%")
    print(f"Mean Error Score:            {mean_error:.4f}")
    print(f"Correct Empty Cells:         {int(total_correct_cells)} / {int(total_empty_cells)}")
    print("=" * 80)

    if args.save_json:
        payload = {
            "checkpoint": args.checkpoint,
            "csv": args.test_csv,
            "examples": total_examples,
            "reverse_steps": args.reverse_steps,
            "sample_actions": bool(args.sample_actions),
            "metrics": {
                "exact_match_accuracy": exact_acc,
                "mean_empty_cell_accuracy": mean_empty_acc,
                "aggregate_empty_cell_accuracy": aggregate_empty_acc,
                "mean_error_score": mean_error,
                "correct_empty_cells": int(total_correct_cells),
                "total_empty_cells": int(total_empty_cells),
            },
            "items": per_item,
        }
        with open(args.save_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved detailed results to {args.save_json}")


if __name__ == "__main__":
    main()
