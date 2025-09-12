import pandas as pd
import numpy as np
import cmdstanpy
from pathlib import Path
import json
import os
import arviz as az

BASE_DIR = Path(__file__).resolve().parent  # Directory of the current script

def read_stan_data():
    data_path = (BASE_DIR / "../../../data/stan_model/stan_data.json").resolve()
    
    with open(data_path, "r") as f:
        data = json.load(f)

    return data


def fit_full_model(data):
    model_path = BASE_DIR / "../stan/reward.stan"
    model = cmdstanpy.CmdStanModel(stan_file=model_path)
    fit = model.sample(data=data, chains=4, parallel_chains=4, iter_warmup=1000, iter_sampling=1000, seed=123, adapt_delta=0.999, max_treedepth=15)
    print(fit.diagnose())
    fit.summary().to_csv(BASE_DIR / "results/fit_summary.csv")
    posterior = fit.stan_variables()
    with open(BASE_DIR / "results/fit_variables.json", "w") as f:
        json.dump({k: v.tolist() for k, v in posterior.items()}, f)
    return fit


def calculate_waic(log_lik):
    elpd_i = np.log(np.exp(log_lik).mean(axis=0)) - np.var(log_lik, axis=0)
    waic = -2 * elpd_i.sum()
    se_waic = np.sqrt(len(elpd_i) * np.var(-2 * elpd_i))  # This is different from the output of arviz
    return waic, se_waic


def main():
    stan_data = read_stan_data()
    fit = fit_full_model(stan_data)
    log_lik = fit.stan_variable("log_lik")
    # waic, se_waic = calculate_waic(log_lik)
    # print(f"WAIC: {waic}, SE_WAIC: {se_waic}")
    idata = az.from_cmdstanpy(posterior=fit, log_likelihood="log_lik")
    waic_res = az.waic(idata, pointwise=True)
    print(waic_res)
    waic_res.to_csv(BASE_DIR / "results/waic.csv")


if __name__ == "__main__":
    main()