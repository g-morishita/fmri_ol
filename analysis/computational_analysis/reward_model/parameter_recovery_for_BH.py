import pandas as pd
import numpy as np
from analysis.computational_analysis.simulator import SocialRewardModel, AsocialRewardModel
from pathlib import Path
from cmdstanpy import CmdStanModel

OTHER_LR = 0.3
HIGH_NOISE_BETA = 1.5
LOW_NOISE_BETA = 20.0
REWARD_PROBS = [0.25, 0.5, 0.75]  # probabilities for each choice
STAN_MODEL_PATH = Path("analysis/computational_analysis/stan/reward.stan")


def simulate_choice_data(n_subjects, n_blocks, n_trials,
                         noise_level_condition,  # shape (S,B), values in {1,2}
                         lrs, betas,             # shape (S,2) each; [:,0]=low, [:,1]=high
                         SocialRewardModel, AsocialRewardModel,
                         rng=None):
    """
    Returns stan_data dict with 1-based choices and integer rewards.
    """
    rng = np.random.default_rng() if rng is None else rng

    self_choices  = np.full((n_subjects, n_blocks, n_trials), -1, dtype=int)
    other_choices = np.full((n_subjects, n_blocks, n_trials), -1, dtype=int)
    other_rewards = np.full((n_subjects, n_blocks, n_trials), -1, dtype=int)

    for i in range(n_subjects):
        for b in range(n_blocks):
            cond = int(noise_level_condition[i, b])  # 1=low, 2=high
            lr_i   = float(lrs[i, cond - 1])
            beta_i = float(betas[i, cond - 1])

            model = SocialRewardModel(reward_lr=lr_i, beta=beta_i)
            partner_model = AsocialRewardModel(
                reward_lr=OTHER_LR,
                beta=HIGH_NOISE_BETA if cond == 2 else LOW_NOISE_BETA
            )

            for t in range(n_trials):
                # Partner chooses (assume 0-based index returned)
                a_o0 = int(partner_model.make_choice())  # 0..2
                # Partner reward from Bernoulli with action-specific prob
                r_o = int(rng.random() < REWARD_PROBS[a_o0])

                partner_model.update_values(r_o, a_o0)
                model.update_values(r_o, a_o0)

                # Self chooses (assume 0-based from simulator)
                a_s0 = int(model.make_choice())

                # Store as 1-based for Stan
                other_choices[i, b, t] = a_o0 + 1
                other_rewards[i, b, t] = r_o
                self_choices[i,  b, t] = a_s0 + 1

    stan_data = {
        "n_subjects": n_subjects,
        "n_blocks": n_blocks,
        "n_trials": n_trials,
        "n_choices": 3,
        "self_choices": self_choices.tolist(),
        "other_choices": other_choices.tolist(),
        "other_rewards": other_rewards.tolist(),
        # if your Stan data also needs this:
        "noise_level_condition": noise_level_condition.tolist(),
    }
    return stan_data


def sample_model_params(n_subjects):
    # Sample model parameters
    lrs = np.random.uniform(0.05, 0.995, size=(n_subjects,2))
    betas = 10 ** np.random.uniform(0, 1.5, size=(n_subjects,2))

    return lrs, betas


def estimate(stan_data, stan_file, results_path):
    model = CmdStanModel(stan_file=stan_file)
    fit = model.sample(data=stan_data, seed=42, adapt_delta=0.99, max_treedepth=15)

    fit.summary().to_csv(results_path)
    print(fit.diagnose())

    return fit


def extract_individual_parameters(fit):
    # Draws shape: (draws, n_subjects, 2)
    lr_draws   = fit.stan_variable("lr")
    beta_draws = fit.stan_variable("beta")

    # Average across draws to get per-subject parameters
    lr_high_noise = lr_draws[:, :, 1].mean(axis=0)  # High noise condition
    lr_low_noise  = lr_draws[:, :, 0].mean(axis=0)  # Low noise condition

    betas_high_noise = beta_draws[:, :, 1].mean(axis=0)
    betas_low_noise  = beta_draws[:, :, 0].mean(axis=0)


    return lr_high_noise, lr_low_noise, betas_high_noise, betas_low_noise


def main(seed):
    n_subjects = 36
    n_blocks = 4
    n_trials = 30
    noise_level_condition = np.tile([1, 1, 2, 2], (n_subjects, 1)).astype(int)  # 1=low noise, 2=high noise

    # Sample model parameters for each subject
    lrs, betas = sample_model_params(n_subjects)

    df = pd.DataFrame({
        "id": np.arange(1, n_subjects + 1),
        "lr_low": lrs[:, 0],
        "lr_high": lrs[:, 1],
        "beta_low": betas[:, 0],
        "beta_high": betas[:, 1],
    })

    # Simulate choice data
    stan_data = simulate_choice_data(
        n_subjects, n_blocks, n_trials,
        noise_level_condition, lrs, betas,
        SocialRewardModel, AsocialRewardModel
    )

    # Fit the Stan model
    results_path = Path(f"results/BH_parameter_recovery_model_summary_seed={seed}.csv")
    fit = estimate(stan_data, STAN_MODEL_PATH, results_path)

    # Extract individual parameters from the fit
    lr_high_noise, lr_low_noise, betas_high_noise, betas_low_noise = extract_individual_parameters(fit)

    df["est_lr_low"] = lr_low_noise
    df["est_lr_high"] = lr_high_noise
    df["est_beta_low"] = betas_low_noise
    df["est_beta_high"] = betas_high_noise
    df.to_csv(f"results/BH_parameter_recovery_seed={seed}.csv", index=False)


if __name__ == "__main__":
    main(42)