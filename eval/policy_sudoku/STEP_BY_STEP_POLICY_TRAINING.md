# Step-by-Step: Train a 4x4 Sudoku Policy Network for LLaDA Remasking

This guide follows your `sudoku_instructions.txt` exactly and uses only new files under `eval/policy_sudoku/`.

## 1) What You Are Training

- Goal: learn a policy `pi_theta(a_t | S_t)` that decides, at each reverse step, which of the 16 Sudoku positions to keep vs remask.
- Action per token: Bernoulli keep/remask.
- Output: `p_t in [0,1]^16` keep probabilities.
- Constraint: given clues are never remasked.

Files added:
- `eval/policy_sudoku/train_policy_sudoku.py`
- `eval/policy_sudoku/policy_guided_remasking.py`

## 2) Phase 1: State Engineering (`S_t`)

For each of 16 positions, state features are:
- Token state id: `0=<MASK>, 1..4` (embedded by policy).
- LLaDA restricted probs: `[P(1), P(2), P(3), P(4)]`.
- Entropy over the 4-digit distribution.
- Normalized timestep `t/T`.

In code:
- `build_state_features(...)` in `train_policy_sudoku.py`
- Input tensor shape to actor: `[B, 16, D]` where `D = token_emb + 6 continuous features`.

## 3) Phase 2: Actor Network

Architecture in `SudokuPolicyNetwork`:
- Token embedding for categorical token state.
- Concatenate embedding + continuous features.
- 2-layer TransformerEncoder, 4 heads.
- Linear head `hidden -> 1` with sigmoid per token.

This gives `keep_probs` with shape `[B,16]`.

## 4) Phase 3: Dense Reward

At each step:
- Compute `X_hat0^(t)` via argmax of LLaDA logits over digits 1-4.
- Compute error score using Sudoku constraints:
  - duplicates in each row
  - duplicates in each column
  - duplicates in each 2x2 subgrid
- Step reward:
  - `r_t = Error(X_hat0^(t-1)) - Error(X_hat0^(t))`

In code:
- `error_score_4x4(...)`
- Reward is computed inside `train_one_epoch(...)`.

## 5) Phase 4: REINFORCE + Baseline

Per episode:
- Sample `a_t ~ Bernoulli(p_t)`.
- Store `log pi_theta(a_t | S_t)`.
- Store dense rewards `r_t`.
- Compute returns `R_t` with `gamma`.
- Loss:
  - `L = -E_t[ log pi_theta(a_t | S_t) * (R_t - b) ]`
- Baseline `b`: moving average of batch returns.

In code:
- `discounted_returns(...)`
- Policy loss in `train_one_epoch(...)`.

## 6) Train with Real LLaDA (No Placeholder)

This pipeline now uses the real base checkpoint you specified: `GSAI-ML/LLaDA-8B-Instruct`.

Data split:
- Train: `dataset/4x4_train_sudoku.csv` (auto-detected; also supports `4x4_train_sudoku,csv` if that is the actual filename)
- Final evaluation: `dataset/4x4_test_sudoku.csv`

Reverse steps are configurable with `--reverse_steps` (not hardcoded).

```bash
cd /scratch/user/ajayjagan2511/d1/eval/policy_sudoku
python train_policy_sudoku.py \
  --model_name GSAI-ML/LLaDA-8B-Instruct \
  --train_csv /scratch/user/ajayjagan2511/d1/dataset/4x4_train_sudoku.csv \
  --test_csv /scratch/user/ajayjagan2511/d1/dataset/4x4_test_sudoku.csv \
  --epochs 20 \
  --batch_size 8 \
  --reverse_steps 16 \
  --device cuda
```

Checkpoint output:
- `eval/policy_sudoku/checkpoints/policy_best.pt`

## 7) Inference-Time Policy-Guided Reverse Steps

Inference uses sampled Bernoulli actions (as requested) instead of deterministic thresholding.

```bash
cd /scratch/user/ajayjagan2511/d1/eval/policy_sudoku
python policy_guided_remasking.py \
  --checkpoint /scratch/user/ajayjagan2511/d1/eval/policy_sudoku/checkpoints/policy_best.pt \
  --puzzle 3102200002100320 \
  --reverse_steps 16 \
  --device cuda
```

## 8) Real Adapter Details

`LladaSudokuAdapter.get_logits(...)` is fully implemented in `train_policy_sudoku.py`:
- Loads frozen LLaDA model + tokenizer.
- Builds per-puzzle chat prompt.
- Appends `<answer>\n` prefix.
- Injects 16 policy-controlled board tokens (mask or digit tokens).
- Runs forward pass and extracts logits for the 16 board positions.
- Restricts logits to digit token ids for `1,2,3,4`.

## 9) Hyperparameters to Start With

- Reverse steps: `--reverse_steps` set to your desired `T` (e.g., 16/24/32).
- Learning rate: `3e-4`.
- Batch size: `8` to start (increase if memory allows).
- Gamma: `0.99`.
- Gradient clip: `1.0`.
- Baseline momentum: `0.9`.
- Save best by exact-match-rate then mean-error.

## 10) Recommended Validation Metrics

Track per epoch:
- Mean final error score.
- Empty-cell accuracy.
- Exact-match rate.
- Average episode return.
- Average keep ratio.

## 11) How to Integrate with Existing LLaDA Inference

For policy-guided remasking in your reverse loop (similar to `eval/generate.py`):
- At each step, after obtaining model logits:
  - build `S_t`
  - run policy for `keep_probs`
  - force given clue positions to keep=1
  - sample Bernoulli actions
  - remask where action is 0
- This replaces fixed schedule heuristics (`low_confidence`, `random`) with learned decisions.

## 12) Practical Training Plan

1. Start with `--reverse_steps 16` and `--epochs 20`.
2. Monitor exact-match and mean-error on the test set each epoch.
3. Try `--reverse_steps 24` and `32` after baseline run.
4. Compare against fixed remasking baseline on same split.
5. If unstable, add entropy bonus or move to PPO.

## One Remaining Detail to Confirm

1. Please confirm the exact filename in `dataset/` is `4x4_train_sudoku.csv` (dot) or `4x4_train_sudoku,csv` (comma). The script can auto-detect both, but using the exact name in commands helps avoid confusion.
