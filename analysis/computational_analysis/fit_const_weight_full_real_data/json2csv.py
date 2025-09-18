import numpy as np
import pandas as pd
from scipy.stats import norm
import json

with open("results/fit_variables.json", "r") as f:
    posterior = json.load(f)

    target_estimates = [
    "mu_latent_lr_reward",
    "mu_latent_lr_action",
    "mu_latent_weight_for_A",
    "mu_latent_beta",
    ]   

    high = {}
    low = {}
    for k, v in posterior.items():
        if k not in target_estimates:
            continue
        v = np.array(v)
        high[k] = norm.cdf(v[:, 1])
        low[k] = norm.cdf(v[:, 0])
        if k == "mu_latent_beta":
            high[k] = np.exp(v[:, 1])
            low[k] = np.exp(v[:, 0])

        print(f"{k} for high-noise: mean {np.mean(high[k])}, std {np.std(high[k])}")
        print(f"{k} for low-noise: mean {np.mean(low[k])}, std {np.std(low[k])}")
    pd.DataFrame(high).to_csv("results/model_estimates_high.csv", index=False)
    pd.DataFrame(low).to_csv("results/model_estimates_low.csv", index=False)
