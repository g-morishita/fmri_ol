import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class Setting:
    BASE_DIR = Path(__file__).parent.resolve()
    RESULTS_DIR = BASE_DIR / "results"
    COND_LABELS = ["Low Noise Partner", "High Noise Partner"]

    VARS_OF_INTEREST =  [
    "mu_latent_lr_reward",
    "mu_latent_lr_action",
    "mu_latent_weight_for_A",
    "mu_latent_beta",
    ]   


# --- plotting helper (single figure per variable; matplotlib only) ---
def plot_two_condition_hist(x1, x2, title, file_name, transform=False):
    if transform:
        x1 = transform(x1)
        x2 = transform(x2)

    plt.figure(figsize=(6, 4.5))
    # density-normalized histograms
    bins = "auto"
    plt.hist(x1, bins=bins, density=True, alpha=0.5, label=Setting.COND_LABELS[0])
    plt.hist(x2, bins=bins, density=True, alpha=0.5, label=Setting.COND_LABELS[1])
    plt.axhline(0, linewidth=0.8)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(Setting.RESULTS_DIR / file_name, dpi=300)
    plt.close()



def read_posterior():
    with open("results/fit_variables.json", "r") as f:
        posterior = json.load(f)
    return posterior


def main():
    posterior = read_posterior()
    for base in Setting.VARS_OF_INTEREST:
        # Expect shape (n_draws, 2). If your JSON is a list, convert to array.
        arr = np.asarray(posterior[base])
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"{base} expected shape (n, 2), got {arr.shape}")

        x1, x2 = arr[:, 0], arr[:, 1]

            # Pretty title & filename
        title = f"{base} — population distributions"
        fname = f"{base}_two_condition_density.png"
        plot_two_condition_hist(x1, x2, title, fname)

        if base != "mu_latent_beta":
            plot_two_condition_hist(x1, x2, title.replace("latent", ""), fname.replace(".png", "_probit.png"), transform=norm.cdf)


if __name__ == "__main__":
    main()