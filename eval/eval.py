import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

from generate import generate
from gsm8k import GSM8KDataset
from math500 import MATH500Dataset
from countdown import CTDDataset
from sudoku import SudokuDataset
from human_eval import HumanEvalDataset
from mbpp import MBPPDataset
from parsers import Parser


DATASET_MAP = {
    "gsm8k": GSM8KDataset,
    "math": MATH500Dataset,
    "countdown": CTDDataset,
    "sudoku": SudokuDataset,
    "humaneval": HumanEvalDataset,
    "mbpp": MBPPDataset,
}


def init_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_ddp():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def evaluate(
    model,
    tokenizer,
    dataloader,
    gen_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    steps=64,
    block_length=32,
    remasking_strategy="low_confidence",
):
    model.eval()
    total_processed = torch.tensor(0, device=model.device)
    wall_times = []
    all_generations = []
    device = model.device

    for batch in tqdm(dataloader, disable=(dist.get_rank() != 0)):
        start_time = time.time()
        input_ids = batch["input_ids"].to(device)
        gt_answers = batch["answers"]
        questions = batch["questions"]
        prompts = batch["prompts"]

        out = generate(
            model,
            input_ids,
            tokenizer,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking=remasking_strategy,
        )

        generated_texts = tokenizer.batch_decode(out[:, -gen_length:], skip_special_tokens=False)
        example_result = [
            {
                "question": questions[j],
                "prompt_input": prompts[j],
                "generations": generated_texts[j],
                "ground_truth": gt_answers[j],
            }
            for j in range(len(gt_answers))
        ]
        all_generations.extend(example_result)
        total_processed += len(generated_texts)
        wall_times.append(time.time() - start_time)

        # Print individual results
        if dist.get_rank() == 0:
            idx = random.randint(0, len(questions) - 1)
            print(f"Question: {questions[idx]}")
            print("-" * 50)
            print("Generation:")
            print(generated_texts[idx])
            print("-" * 50)
            print(f"Ground truth: {gt_answers[idx]}")

    avg_wall_time = sum(wall_times) / len(wall_times)
    metrics = {
        "wall_time": avg_wall_time,
        "generations": all_generations,
        "total_processed": total_processed.item(),
    }
    return metrics


def _single_token_id(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"Expected single token for {text!r}, got {ids}")
    return ids[0]


def _load_sudoku_policy(checkpoint_path, device):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from policy_training.train_policy_sudoku import PolicyConfig, SudokuPolicyNetwork

    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = PolicyConfig(**ckpt["policy_config"])
    policy_model = SudokuPolicyNetwork(cfg).to(device)
    policy_model.load_state_dict(ckpt["policy_state_dict"])
    policy_model.eval()
    return policy_model


def evaluate_sudoku_policy_csv(
    model,
    tokenizer,
    policy_checkpoint_path,
    csv_path,
    steps,
    batch_size,
    temperature=0.0,
    cfg_scale=0.0,
    sample_actions=False,
):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Sudoku CSV not found: {csv_path}")

    device = model.device
    mask_id = getattr(model.config, "mask_token_id", None)
    if mask_id is None:
        mask_id = tokenizer.mask_token_id
    if mask_id is None:
        mask_id = 126336

    digit_token_ids = [_single_token_id(tokenizer, str(d)) for d in (1, 2, 3, 4)]
    token_to_digit = {tok: idx + 1 for idx, tok in enumerate(digit_token_ids)}

    policy_model = _load_sudoku_policy(policy_checkpoint_path, device)

    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            puzzle = row["Puzzle"].strip()
            solution = row["Solution"].strip()
            if len(puzzle) == 16 and len(solution) == 16:
                rows.append((puzzle, solution))

    if not rows:
        raise RuntimeError(f"No valid rows in {csv_path}")

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rows = [rows[i] for i in range(rank, len(rows), world_size)]

    local_exact = 0
    local_total = 0
    local_empty_correct = 0
    local_empty_total = 0
    local_records = []

    for bstart in range(0, len(local_rows), batch_size):
        batch_rows = local_rows[bstart : bstart + batch_size]
        bsz = len(batch_rows)

        init_gen_ids = torch.full((bsz, 16), mask_id, dtype=torch.long, device=device)
        given_mask = torch.zeros((bsz, 16), dtype=torch.long, device=device)

        for bi, (puzzle, _) in enumerate(batch_rows):
            for j, ch in enumerate(puzzle):
                if ch != "0":
                    d = int(ch)
                    init_gen_ids[bi, j] = digit_token_ids[d - 1]
                    given_mask[bi, j] = 1

        prompt = torch.empty((bsz, 0), dtype=torch.long, device=device)

        out = generate(
            model,
            prompt,
            tokenizer,
            steps=steps,
            gen_length=16,
            block_length=16,
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking="policy",
            mask_id=mask_id,
            init_gen_ids=init_gen_ids,
            policy_model=policy_model,
            policy_digit_token_ids=digit_token_ids,
            policy_given_mask=given_mask,
            policy_sample_actions=sample_actions,
        )

        preds = out[:, :16].detach().cpu().tolist()
        for bi, (puzzle, solution) in enumerate(batch_rows):
            pred_digits = []
            for tok in preds[bi]:
                pred_digits.append(str(token_to_digit.get(tok, 0)))
            pred_str = "".join(pred_digits)

            exact = int(pred_str == solution)
            local_exact += exact
            local_total += 1

            empty_idx = [k for k, ch in enumerate(puzzle) if ch == "0"]
            local_empty_total += len(empty_idx)
            local_empty_correct += sum(1 for k in empty_idx if pred_str[k] == solution[k])

            if rank == 0:
                local_records.append(
                    {
                        "question": f"Solve the following Sudoku puzzle: {puzzle}",
                        "prompt_input": puzzle,
                        "generations": pred_str,
                        "ground_truth": solution,
                    }
                )

    stat = torch.tensor(
        [local_exact, local_total, local_empty_correct, local_empty_total],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(stat, op=dist.ReduceOp.SUM)
    exact_correct, total, empty_correct, empty_total = stat.tolist()

    metrics = {
        "exact_match_acc": (exact_correct / total) if total else 0.0,
        "empty_cell_acc": (empty_correct / empty_total) if empty_total else 0.0,
        "total_processed": int(total),
        "exact_correct": int(exact_correct),
    }

    return {"metrics": metrics, "generations": local_records}


class CustomDistributedSampler(DistributedSampler):
    """
    From torch docs:
    drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas

    We want drop_last = False, but don't want to have extra padding indices. Hence using a custom sampler.
    """

    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        drop_last=False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil((len(self.dataset) - self.num_replicas) / self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas
        else:
            # If we don't drop the last batch, we need to calculate the number of samples per rank.
            self.total_size = len(self.dataset)
            self.num_samples = len(self.dataset) // self.num_replicas + int(
                rank < (self.total_size % self.num_replicas)
            )

        self.shuffle = shuffle
        self.seed = seed


if __name__ == "__main__":
    init_seed(42)

    # Note: This evaluation script saves only model generations. A separate parser is used later to extract
    # predictions and calculate metrics.

    local_rank = setup_ddp()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/data1/shared/LLaDA-8B-Instruct/")
    parser.add_argument("--few_shot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["gsm8k", "math", "countdown", "sudoku", "game24", "humaneval", "mbpp"],
        default="gsm8k",
    )
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--diffusion_steps", type=int, default=64)
    parser.add_argument("--add_reasoning", action="store_true")
    parser.add_argument("--dont_save", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results/")
    parser.add_argument("--dont_use_box", action="store_true")
    parser.add_argument("--toy_evaluation", action="store_true")
    parser.add_argument("--sudoku_csv", type=str, default="")
    parser.add_argument("--policy_checkpoint_path", type=str, default="")
    parser.add_argument("--remasking_strategy", type=str, default="low_confidence")
    parser.add_argument("--policy_sample_actions", action="store_true")
    args = parser.parse_args()

    args.diffusion_steps = args.gen_length // 2
    num_evals = {"gsm8k": -1, "math": -1, "countdown": 256, "sudoku": 256, "humaneval": -1, "mbpp": -1}

    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(
        local_rank
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if args.checkpoint_path:
        model = PeftModel.from_pretrained(model, args.checkpoint_path, torch_dtype=torch.bfloat16).to(
            local_rank
        )

        if dist.get_world_size() > 1:
            dist.barrier()  # Make sure all processes are ready
            for param in model.parameters():
                dist.broadcast(param.data, src=0)
            print(f"Rank {local_rank}: Parameters synchronized")

    direct_sudoku_policy_eval = (
        args.dataset == "sudoku" and len(args.policy_checkpoint_path) > 0 and len(args.sudoku_csv) > 0
    )

    if not direct_sudoku_policy_eval:
        dataset = DATASET_MAP[args.dataset](
            tokenizer,
            subsample=num_evals[args.dataset],
            num_examples=args.few_shot,
            add_reasoning=True,  # prefill for all models
            toy_evaluation=args.toy_evaluation,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=CustomDistributedSampler(dataset, shuffle=False),
            collate_fn=dataset.collate_fn,
        )

    if len(args.checkpoint_path):
        model_name = args.checkpoint_path.split("/")
        model_name = model_name[-2] + "_" + model_name[-1]
    else:
        model_name = "instruct" if "Instruct" in args.model_path else "base"

    if args.few_shot > 0:
        model_name = model_name + f"_fs{args.few_shot}"

    if len(args.suffix) > 0:
        model_name = model_name + f"_{args.suffix}"

    os.makedirs(args.output_dir, exist_ok=True)
    filename = f"{args.output_dir}/{args.dataset}_{model_name}_{args.gen_length}_{args.diffusion_steps}_{dist.get_rank()}_generations.json"
    print(f"Saving generations to {filename}")

    if direct_sudoku_policy_eval:
        policy_result = evaluate_sudoku_policy_csv(
            model=model,
            tokenizer=tokenizer,
            policy_checkpoint_path=args.policy_checkpoint_path,
            csv_path=args.sudoku_csv,
            steps=args.diffusion_steps,
            batch_size=args.batch_size,
            sample_actions=args.policy_sample_actions,
        )
        metrics = {
            "wall_time": 0.0,
            "total_processed": policy_result["metrics"]["total_processed"],
            "exact_match_acc": policy_result["metrics"]["exact_match_acc"],
            "empty_cell_acc": policy_result["metrics"]["empty_cell_acc"],
            "exact_correct": policy_result["metrics"]["exact_correct"],
            "generations": policy_result["generations"],
        }

        if dist.get_rank() == 0:
            print(
                f"[Sudoku Policy Eval] exact_match_acc={metrics['exact_match_acc']:.4f} "
                f"empty_cell_acc={metrics['empty_cell_acc']:.4f} "
                f"processed={metrics['total_processed']}"
            )
    else:
        metrics = evaluate(
            model,
            tokenizer,
            dataloader,
            gen_length=args.gen_length,
            block_length=args.block_length,
            steps=args.diffusion_steps,
            remasking_strategy=args.remasking_strategy,
        )

    if not args.dont_save and dist.get_rank() == 0:
        with open(filename, "w") as f:
            json.dump(
                {
                    "generations": metrics["generations"],
                    "metrics": {
                        "wall_time": metrics["wall_time"],
                        "total_processed": metrics["total_processed"],
                        "exact_match_acc": metrics.get("exact_match_acc", None),
                        "empty_cell_acc": metrics.get("empty_cell_acc", None),
                    },
                    "model_path": args.model_path,
                    "checkpoint_path": args.checkpoint_path,
                    "policy_checkpoint_path": args.policy_checkpoint_path,
                    "gen_length": args.gen_length,
                    "diffusion_steps": args.diffusion_steps,
                    "block_length": args.block_length,
                },
                f,
                indent=2,
            )

    cleanup_ddp()
