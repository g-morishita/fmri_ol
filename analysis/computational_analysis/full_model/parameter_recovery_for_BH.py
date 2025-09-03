import pandas as pd
import numpy as np
from analysis.computational_analysis.simulator import FullModel, AsocialRewardModel
from pathlib import Path
from cmdstanpy import CmdStanModel

OTHER_LR = 0.3
HIGH_NOISE_BETA = 1.5
LOW_NOISE_BETA = 20.0
REWARD_PROBS = [0.25, 0.5, 0.75]  # probabilities for each choice
STAN_MODEL_PATH = Path("analysis/computational_analysis/stan/full.stan")


def simulate_choice_data(n_subjects, n_blocks, n_trials,
                         noise_level_condition,  # shape (S,B), values in {1,2}
                         lrs_reward, lrs_action, weights_for_A, betas,             # shape (S,2) each; [:,0]=low, [:,1]=high
                         FullModel, AsocialRewardModel,
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
            lr_reward_i   = float(lrs_reward[i, cond - 1])
            lr_action_i = float(lrs_action[i, cond - 1])
            weight_for_A_i = float(weights_for_A[i, cond - 1])
            beta_i = float(betas[i, cond - 1])

            model = FullModel(reward_lr=lr_reward_i, action_lr=lr_action_i, weight_for_A=weight_for_A_i, beta=beta_i)
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
    lrs_rewards = np.random.uniform(0.05, 0.995, size=(n_subjects,2))
    lrs_action = np.random.uniform(0.05, 0.995, size=(n_subjects,2))
    weights_for_A = np.random.uniform(0.05, 0.995, size=(n_subjects,2))
    betas = 10 ** np.random.uniform(0, 1.5, size=(n_subjects,2))

    return lrs_rewards, lrs_action, weights_for_A, betas


def estimate(stan_data, stan_file, results_path):
    model = CmdStanModel(stan_file=stan_file)
    fit = model.sample(data=stan_data, seed=42, adapt_delta=0.99, max_treedepth=15)

    fit.summary().to_csv(results_path)
    print(fit.diagnose())

    return fit


def _extract_mean(draws, is_high_noise):
    return draws[:, :, 1].mean(axis=0),  draws[:, :, 0].mean(axis=0) # 1 if high noise, 0 if low noise

def extract_individual_parameters(fit):
    # Draws shape: (draws, n_subjects, 2)
    lr_reward_draws = fit.stan_variable("lr_reward")
    lr_action_draws = fit.stan_variable("lr_action")
    weights_for_A_draws = fit.stan_variable("weight_for_A")
    beta_draws = fit.stan_variable("beta")

    # Average across draws to get per-subject parameters
    lr_reward_high_noise, lr_reward_low_noise = _extract_mean(lr_reward_draws, is_high_noise=True)
    lr_action_high_noise, lr_action_low_noise = _extract_mean(lr_action_draws, is_high_noise=True)
    weights_for_A_high_noise, weights_for_A_low_noise = _extract_mean(weights_for_A_draws, is_high_noise=True)
    beta_high_noise, beta_low_noise = _extract_mean(beta_draws, is_high_noise=True)

    return lr_reward_high_noise, lr_reward_low_noise, \
              lr_action_high_noise, lr_action_low_noise, \
                weights_for_A_high_noise, weights_for_A_low_noise, \
                beta_high_noise, beta_low_noise


def main(seed):
    n_subjects = 36
    n_blocks = 4
    n_trials = 30
    noise_level_condition = np.tile([1, 1, 2, 2], (n_subjects, 1)).astype(int)  # 1=low noise, 2=high noise

    # Sample model parameters for each subject
    lrs_reward, lrs_action, weights_for_A, betas = sample_model_params(n_subjects)

    df = pd.DataFrame({
        "id": np.arange(1, n_subjects + 1),
        "lr_reward_low": lrs_reward[:, 0],
        "lr_reward_high": lrs_reward[:, 1],
        "lr_action_low": lrs_action[:, 0],
        "lr_action_high": lrs_action[:, 1],
        "weight_for_A_low": weights_for_A[:, 0],
        "weight_for_A_high": weights_for_A[:, 1],
        "beta_low": betas[:, 0],
        "beta_high": betas[:, 1],
    })

    # Simulate choice data
    stan_data = simulate_choice_data(
        n_subjects, n_blocks, n_trials,
        noise_level_condition, lrs_reward, lrs_action, weights_for_A, betas,
        FullModel, AsocialRewardModel
    )

    # Fit the Stan model
    results_path = Path(f"results/BH_full_parameter_recovery_model_summary_seed={seed}.csv")
    fit = estimate(stan_data, STAN_MODEL_PATH, results_path)

    # Extract individual parameters from the fit
    lr_reward_high, lr_reward_low, lr_action_high, lr_action_low, weight_for_A_high, weight_for_A_low, beta_high, beta_low = extract_individual_parameters(fit)

    # Store estimated parameters in the DataFrame
    df["est_lr_reward_high"] = lr_reward_high
    df["est_lr_reward_low"] = lr_reward_low
    df["est_lr_action_high"] = lr_action_high
    df["est_lr_action_low"] = lr_action_low
    df["est_weight_for_A_high"] = weight_for_A_high
    df["est_weight_for_A_low"] = weight_for_A_low
    df["est_beta_high"] = beta_high
    df["est_beta_low"] = beta_low

    df.to_csv(f"results/BH_full_parameter_recovery_seed={seed}.csv", index=False)


if __name__ == "__main__":
    main(43)