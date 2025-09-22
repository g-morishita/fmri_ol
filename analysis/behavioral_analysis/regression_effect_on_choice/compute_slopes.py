# save_simple_slopes_pm1.py
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import norm

RESULTS_DIR = Path("results/combined")  # adjust if needed
BETA_CSV = RESULTS_DIR / "combined_fixed_effects_full_vcv.csv"
VCV_CSV  = RESULTS_DIR / "combined_vcov_full.csv"

OUT_FILE = RESULTS_DIR / "simple_slopes_pm1.csv"

FOCAL_LIST = ["PR_minus_0", "PC_minus_0"]   # add more as needed
MODERATOR_NAME = "is_partner_high_exp"

def main():
    beta_df = pd.read_csv(BETA_CSV)
    V = pd.read_csv(VCV_CSV, index_col=0)

    beta = beta_df.set_index("name")["Estimate"]

    out_rows = []

    for focal in FOCAL_LIST:
        # interaction term name
        inter1 = f"{focal}:{MODERATOR_NAME}"
        inter2 = f"{MODERATOR_NAME}:{focal}"
        if inter1 in beta.index:
            inter = inter1
        elif inter2 in beta.index:
            inter = inter2
        else:
            raise KeyError(f"No interaction term found for {focal} × {MODERATOR_NAME}")

        for m in [-1, 1]:
            slope = beta[focal] + m * beta[inter]
            var = (
                V.loc[focal, focal]
                + V.loc[inter, inter]
                + 2 * m * V.loc[focal, inter]
            )
            se = np.sqrt(var)
            z = slope / se
            p = 2 * (1 - norm.cdf(abs(z)))

            out_rows.append({
                "focal": focal,
                MODERATOR_NAME: m,
                "slope": slope,
                "SE": se,
                "z": z,
                "p": p
            })

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(OUT_FILE, index=False)
    print(f"Saved simple slopes (±1 coding) to {OUT_FILE}")

if __name__ == "__main__":
    main()
