# plot_simple_slopes.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
RESULTS_DIR = Path("results/combined")   # adjust if needed
CSV_FILE    = RESULTS_DIR / "simple_slopes_pm1.csv"
OUT_DIR     = Path("figures/plots_slopes")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODERATOR_NAME = "is_partner_high_exp"

COLORS = {
    -1: "green",   # low noise
    +1: "orange"   # high noise
}

def main():
    df = pd.read_csv(CSV_FILE)

    # loop over each focal predictor
    for focal, subdf in df.groupby("focal"):
        fig, ax = plt.subplots(figsize=(5, 4))

        # iterate over moderator levels
        for _, row in subdf.iterrows():
            m = row[MODERATOR_NAME]
            slope = row["slope"]
            se = row["SE"]

            xpos = -0.5 if m == -1 else 0.5
            label = "Low noise" if m == -1 else "High noise"

            # error bar = ±SE
            ax.errorbar(
                xpos, slope,
                yerr=se,
                fmt="o", capsize=6,
                color=COLORS[m], label=label
            )

        ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)

        ax.set_xticks([-0.5, 0.5])
        ax.set_xticklabels(["", ""])
        ax.set_xlim(-0.7, 0.7)
        # ax.set_ylabel("Slope (logit scale)")
        # ax.set_title(f"Simple slopes for {focal}")
        # ax.legend().set_visible(False)  # legend redundant if x-ticks already labeled
        # remove line around the plot
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)

        out_path = OUT_DIR / f"{focal}_slopes.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
