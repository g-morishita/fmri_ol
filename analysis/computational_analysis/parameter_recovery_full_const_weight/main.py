# parameter_recovery_full.py
import numpy as np
import pandas as pd
from pathlib import Path
from cmdstanpy import CmdStanModel

# ---- your simulators (import path as in your project) ----
from analysis.computational_analysis.simulator import FullModel, AsocialRewardModel

# -----------------------------
# Configuration
# -----------------------------
N_SUBJECTS = 36
N_BLOCKS   = 4
N_TRIALS   = 30

# 1=low noise, 2=high noise (shape: S x B)
NOISE_LEVEL_CONDITION = np.tile([1, 1, 2, 2], (N_SUBJECTS, 1)).astype(int)

# Task / partner
REWARD_PROBS     = np.array([0.25, 0.50, 0.75])   # must match n_choices=3
OTHER_LR         = 0.3
HIGH_NOISE_BETA  = 1.5
LOW_NOISE_BETA   = 20.0

# Stan
STAN_MODEL_PATH  = (Path(__file__).parent / "../stan/cons_weight_full.stan").resolve()
ADAPT_DELTA      = 0.99
MAX_TREEDEPTH    = 15
SEED             = 43

# IMPORTANT: match this to your Stan beta mapping
# - If Stan uses beta = exp(latent_beta) or log1p_exp(latent_beta)  -> set "exp"
# - If Stan uses beta = Phi(latent_beta) (i.e., β in (0,1))         -> set "phi"
BETA_MODE        = "exp"   # "exp" or "phi"

# -----------------------------
# RNG & helpers
# -----------------------------
def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)

def _simulate_betas(rng: np.random.Generator, n_subjects: int) -> np.ndarray:
    """
    Per-subject beta for both conditions (S x 2).
    'exp': log-uniform-ish ~ 10 ** U(0, 1.5) -> [1, 31.6]
    'phi': in (0,1) via Phi(N(0,1))
    """
    if BETA_MODE == "exp":
        return 10.0 ** rng.uniform(0.0, 1.5, size=(n_subjects, 2))
    elif BETA_MODE == "phi":
        z = rng.normal(0.0, 1.0, size=(n_subjects, 2))
        return 0.5 * (1.0 + np.erf(z / np.sqrt(2.0)))
    else:
        raise ValueError(f"Unknown BETA_MODE: {BETA_MODE}")

# -----------------------------
# True parameter sampling
# -----------------------------
def sample_model_params(n_subjects: int, rng: np.random.Generator):
    """
    Returns (lrs_reward, lrs_action, weights_for_A, betas),
    each shaped (S x 2) with [:,0]=low, [:,1]=high.
    """
    lrs_reward    = rng.uniform(0.05, 0.995, size=(n_subjects, 2))
    lrs_action    = rng.uniform(0.05, 0.995, size=(n_subjects, 2))
    weights_for_A = rng.uniform(0.05, 0.995, size=(n_subjects, 2))
    betas         = _simulate_betas(rng, n_subjects)
    return lrs_reward, lrs_action, weights_for_A, betas

# -----------------------------
# Data simulation
# -----------------------------
def simulate_choice_data(n_subjects: int,
                         n_blocks:   int,
                         n_trials:   int,
                         noise_level_condition: np.ndarray,
                         lrs_reward: np.ndarray,
                         lrs_action: np.ndarray,
                         weights_for_A: np.ndarray,
                         betas: np.ndarray,
                         rng: np.random.Generator):
    """
    Simulates the within-subject, two-condition task.
    Returns a dict ready for Stan with 1-based choices and integer rewards.
    """
    n_choices = REWARD_PROBS.size
    self_choices  = np.full((n_subjects, n_blocks, n_trials), -1, dtype=int)
    other_choices = np.full((n_subjects, n_blocks, n_trials), -1, dtype=int)
    other_rewards = np.full((n_subjects, n_blocks, n_trials), -1, dtype=int)

    for i in range(n_subjects):
        for b in range(n_blocks):
            cond = int(noise_level_condition[i, b])   # 1=low, 2=high
            lr_r  = float(lrs_reward[i,    cond - 1])
            lr_a  = float(lrs_action[i,    cond - 1])
            wA    = float(weights_for_A[i, cond - 1])
            beta  = float(betas[i,         cond - 1])

            # Agents
            learner = FullModel(reward_lr=lr_r, action_lr=lr_a, weight_for_A=wA, beta=beta)
            partner = AsocialRewardModel(
                reward_lr=OTHER_LR,
                beta=HIGH_NOISE_BETA if cond == 2 else LOW_NOISE_BETA
            )

            for t in range(n_trials):
                # Partner acts first
                a_o0 = int(partner.make_choice())           # 0..n_choices-1
                r_o  = int(rng.random() < REWARD_PROBS[a_o0])

                # Update partner & learner from other's observation
                partner.update_values(r_o, a_o0)
                learner.update_values(r_o, a_o0)

                # Learner acts
                a_s0 = int(learner.make_choice())

                # Store as 1-based indices for Stan
                other_choices[i, b, t] = a_o0 + 1
                other_rewards[i, b, t] = r_o
                self_choices[i,  b, t] = a_s0 + 1

    stan_data = {
        "n_subjects": n_subjects,
        "n_blocks": n_blocks,
        "n_trials": n_trials,
        "n_choices": int(n_choices),
        "self_choices":  self_choices.tolist(),
        "other_choices": other_choices.tolist(),
        "other_rewards": other_rewards.tolist(),
        "noise_level_condition": noise_level_condition.tolist(),  # 1=low, 2=high
    }
    return stan_data

# -----------------------------
# Fitting
# -----------------------------
def estimate(stan_data: dict, stan_file: Path, results_path: Path, seed: int):
    model = CmdStanModel(stan_file=str(stan_file))
    fit = model.sample(
        data=stan_data,
        seed=seed,
        adapt_delta=ADAPT_DELTA,
        max_treedepth=MAX_TREEDEPTH
    )
    # Save summary + diagnostics
    results_path.parent.mkdir(parents=True, exist_ok=True)
    fit.summary().to_csv(results_path)
    diag = fit.diagnose()
    (results_path.parent / "diagnostics.txt").write_text(diag)
    print(diag)
    return fit

# -----------------------------
# Extraction (matches your Stan names)
# -----------------------------
def _pair_to_draws3(fit, low_name: str, high_name: str) -> np.ndarray:
    """
    Returns draws shaped (n_draws, n_subjects, 2) with [:,:,0]=low, [:,:,1]=high
    """
    low  = fit.stan_variable(low_name)   # (draws, S)
    high = fit.stan_variable(high_name)  # (draws, S)
    if low.ndim != 2 or high.ndim != 2 or low.shape != high.shape:
        raise ValueError(f"Unexpected shapes: {low_name} {low.shape}, {high_name} {high.shape}")
    return np.stack([low, high], axis=2)

def _extract_means(draws_3d: np.ndarray):
    """
    draws_3d: (n_draws, n_subjects, 2) with [:,:,0]=low, [:,:,1]=high
    Returns (high_mean, low_mean) each shape (n_subjects,)
    """
    low_mean  = draws_3d[:, :, 0].mean(axis=0)
    high_mean = draws_3d[:, :, 1].mean(axis=0)
    return high_mean, low_mean

def extract_individual_parameters(fit):
    lr_reward_draws     = _pair_to_draws3(fit, "lr_reward_low_noise",    "lr_reward_high_noise")
    lr_action_draws     = _pair_to_draws3(fit, "lr_action_low_noise",    "lr_action_high_noise")
    weight_for_A_draws  = _pair_to_draws3(fit, "weight_for_A_low_noise", "weight_for_A_high_noise")
    beta_draws          = _pair_to_draws3(fit, "beta_low_noise",         "beta_high_noise")

    lr_reward_high,  lr_reward_low  = _extract_means(lr_reward_draws)
    lr_action_high,  lr_action_low  = _extract_means(lr_action_draws)
    wA_high,         wA_low         = _extract_means(weight_for_A_draws)
    beta_high,       beta_low       = _extract_means(beta_draws)

    return (lr_reward_high, lr_reward_low,
            lr_action_high, lr_action_low,
            wA_high, wA_low,
            beta_high, beta_low)

# -----------------------------
# Main
# -----------------------------
def main(seed: int = SEED):
    rng = _rng(seed)

    # True parameters
    lrs_reward, lrs_action, weights_for_A, betas = sample_model_params(N_SUBJECTS, rng)

    # Ground-truth dataframe
    df = pd.DataFrame({
        "id": np.arange(1, N_SUBJECTS + 1),
        "lr_reward_low_true":     lrs_reward[:, 0],
        "lr_reward_high_true":    lrs_reward[:, 1],
        "lr_action_low_true":     lrs_action[:, 0],
        "lr_action_high_true":    lrs_action[:, 1],
        "weight_for_A_low_true":  weights_for_A[:, 0],
        "weight_for_A_high_true": weights_for_A[:, 1],
        "beta_low_true":          betas[:, 0],
        "beta_high_true":         betas[:, 1],
    })

    # Simulate choices
    stan_data = simulate_choice_data(
        n_subjects=N_SUBJECTS,
        n_blocks=N_BLOCKS,
        n_trials=N_TRIALS,
        noise_level_condition=NOISE_LEVEL_CONDITION,
        lrs_reward=lrs_reward,
        lrs_action=lrs_action,
        weights_for_A=weights_for_A,
        betas=betas,
        rng=rng
    )

    # Fit Stan model
    results_dir = Path(f"results/{seed}")
    fit = estimate(stan_data, STAN_MODEL_PATH, results_dir / "model_summary.csv", seed)

    # Extract per-subject posterior means
    (lr_reward_high, lr_reward_low,
     lr_action_high, lr_action_low,
     wA_high, wA_low,
     beta_high, beta_low) = extract_individual_parameters(fit)

    # Merge estimates
    df["lr_reward_low_est"]     = lr_reward_low
    df["lr_reward_high_est"]    = lr_reward_high
    df["lr_action_low_est"]     = lr_action_low
    df["lr_action_high_est"]    = lr_action_high
    df["weight_for_A_low_est"]  = wA_low
    df["weight_for_A_high_est"] = wA_high
    df["beta_low_est"]          = beta_low
    df["beta_high_est"]         = beta_high

    # Save
    out_csv = results_dir / "true_est_comparison.csv"
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv.resolve()}")

if __name__ == "__main__":
    main(SEED)
