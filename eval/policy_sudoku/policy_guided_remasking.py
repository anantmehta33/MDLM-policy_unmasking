import argparse
from typing import Dict

import torch

from train_policy_sudoku import (
    DEFAULT_MASK_ID,
    PolicyConfig,
    SudokuPolicyNetwork,
    build_state_features,
    argmax_board_from_logits,
    apply_actions,
    enforce_given_clues,
    LladaSudokuAdapter,
    sample_actions_and_logprob,
    parse_grid_str,
)


def run_policy_guided_reverse(
    puzzle_str: str,
    policy: SudokuPolicyNetwork,
    llada: LladaSudokuAdapter,
    reverse_steps: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    puzzle = torch.tensor(parse_grid_str(puzzle_str), dtype=torch.long, device=device).unsqueeze(0)
    given_mask = (puzzle != 0).long()

    current_tokens = torch.where(given_mask == 1, puzzle, torch.zeros_like(puzzle))
    history = []

    with torch.no_grad():
        for t in range(1, reverse_steps + 1):
            logits_4 = llada.get_logits(current_tokens=current_tokens, puzzle_tokens=puzzle)
            state_features = build_state_features(current_tokens, logits_4, t=t, total_steps=reverse_steps)

            keep_probs = policy(current_tokens, state_features)
            keep_probs, current_tokens = enforce_given_clues(keep_probs, given_mask, current_tokens, puzzle)

            # Sampled action as requested.
            actions_keep, _ = sample_actions_and_logprob(keep_probs)
            actions_keep[given_mask == 1] = 1

            xhat0 = argmax_board_from_logits(logits_4)
            xhat0[given_mask == 1] = puzzle[given_mask == 1]

            current_tokens = apply_actions(xhat0, actions_keep)
            current_tokens[given_mask == 1] = puzzle[given_mask == 1]

            history.append(
                {
                    "step": t,
                    "keep_probs": keep_probs.detach().cpu().squeeze(0),
                    "actions_keep": actions_keep.detach().cpu().squeeze(0),
                    "xhat0": xhat0.detach().cpu().squeeze(0),
                    "next_tokens": current_tokens.detach().cpu().squeeze(0),
                }
            )

    return {
        "final_tokens": current_tokens.squeeze(0).detach().cpu(),
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--puzzle", type=str, required=True)
    parser.add_argument("--reverse_steps", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--mask_id", type=int, default=DEFAULT_MASK_ID)
    args = parser.parse_args()

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

    result = run_policy_guided_reverse(
        puzzle_str=args.puzzle,
        policy=policy,
        llada=llada,
        reverse_steps=args.reverse_steps,
        device=device,
    )

    final_tokens = "".join(str(int(x.item())) for x in result["final_tokens"])
    print(f"Final tokens: {final_tokens}")


if __name__ == "__main__":
    main()
