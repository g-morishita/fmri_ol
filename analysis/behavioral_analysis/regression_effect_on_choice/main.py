import pandas as pd
import numpy as np
from pymer4.models import Lmer
from scipy.stats import norm

def create_dataset(
        df,
        target_choice,
        period
):
    """
    Build a trial-level panel with lagged self/partner choice and reward
    codes ready for mixed-effects (or GEE) modelling.

    Parameters
    ----------
    df : DataFrame
        Must contain at least these columns:
        ─ id, block, trial
        ─ self_choice, partner_choice
        ─ partner_reward   (0 = no reward, 1 = rewarded)
    target_choice : int | str
        The option you treat as “+1”.  Anything else becomes –1.
    period : int
        How many **previous trials, inclusive of the current one** to keep.
        `period=4` ⇒ C_t_minus_0 … C_t_minus_4 (five columns per variable).

    Returns
    -------
    DataFrame
        Original rows (minus those without a complete lag window) +
        lagged C/PC/PR columns, z-scored PC/PR lags, and binary outcome
        `choice_Y` (1 = chose target, 0 = otherwise).
    """
    out = df.copy()

    # ──────────────────────────────────────────────────────────────────────
    # 1. Code the current-trial variables (vectorised, no .apply row loop)
    # ──────────────────────────────────────────────────────────────────────
    out["C_t"]  = np.where(out["self_choice"]    == target_choice,  1, -1)
    out["PC_t"] = np.where(out["partner_choice"] == target_choice,  1, -1)
    out["PR_t"] = np.where(
        out["partner_reward"].eq(0),                       0,        # no reward
        np.where(out["partner_choice"].eq(target_choice),  1, -1)    # rewarded
    )

    # ──────────────────────────────────────────────────────────────────────
    # 2. Order by person-block-trial so shifts line up
    # ──────────────────────────────────────────────────────────────────────
    out = out.sort_values(["id", "block", "trial"])

    # ──────────────────────────────────────────────────────────────────────
    # 3. Create lagged columns in one pass
    # ──────────────────────────────────────────────────────────────────────
    lagged = {}
    keys   = ["C_t", "PC_t", "PR_t"]
    gb     = out.groupby(["id", "block"])

    for lag in range(period + 1):            # 0 = current trial, 1 = previous …
        for key in keys:
            lagged[f"{key[:-2]}_minus_{lag}"] = gb[key].shift(lag)

    out = pd.concat([out, pd.DataFrame(lagged, index=out.index)], axis=1)

    # ──────────────────────────────────────────────────────────────────────
    # 4. Keep only rows that have a full lag window
    # ──────────────────────────────────────────────────────────────────────
    out = out.dropna(subset=lagged).reset_index(drop=True)

    # ──────────────────────────────────────────────────────────────────────
    # 5. Binary outcome
    # ──────────────────────────────────────────────────────────────────────
    out["choice_Y"] = (out["C_t"] == 1).astype(int)

    # out.to_csv("lagged_choice_data.csv", index=False)

    # ──────────────────────────────────────────────────────────────────────
    # 6. Z-score partner-choice & partner-reward lags (C_t lags stay ±1)
    # ──────────────────────────────────────────────────────────────────────
    z_cols = [c for c in lagged if c.startswith(("PC", "PR"))]
    out[z_cols] = out[z_cols].apply(lambda s: (s - s.mean()) / s.std(ddof=0))

    # out.to_csv("standardized_lagged_choice_data.csv", index=False)

    return out


def run_regression(data, period, outcome="choice_Y", interaction_var="is_partner_high_exp"):
    """
    Build and fit a mixed-effects logistic regression model with period lags
    of PR_t_minus_i and PC_t_minus_i interacting with the interaction_var.

    The full model formula is:
      outcome ~ (PR_t_minus_0 + ... + PR_t_minus_{period-1} + PC_t_minus_0 + ... + PC_t_minus_{period-1})
                * interaction_var + (1 + (PR_t_minus_0 + ... + PC_t_minus_{period-1}) | id)
    """
    # Build lists of PR_t and PC_t terms for i in [0, period-1].
    pr_terms = [f"PR_minus_{i}" for i in range(0, period + 1)]
    pc_terms = [f"PC_minus_{i}" for i in range(0, period + 1)]
    all_terms = pr_terms + pc_terms

    fixed_effects = " + ".join(all_terms)
    # Construct the full formula with an interaction.
    formula = (
        f"{outcome} ~ ({fixed_effects}) * {interaction_var} "
        f"+ (1 + {fixed_effects} | id)"
    )

    model = Lmer(formula, data=data, family="binomial")
    model_fit = model.fit(control="optimizer='bobyqa', optCtrl = list(maxfun=5e5)")
    return model_fit


if __name__ == "__main__":
    data = pd.read_csv("../../../preprocess/preprocessed_data.csv")

    results = []
    for target_choice in range(3):
        out = create_dataset(data, target_choice, 0)

        result = run_regression(out, period=0).reset_index(names='name')
        result["Variance"] = result["SE"] ** 2

        results.append(result)

    combined_estimate = {"name": results[0]["name"].tolist(), "estimate": np.ones(results[0].shape[0]), "s.e.": np.ones(results[0].shape[0])}

    for i, name in enumerate(combined_estimate["name"]):
        for target_choice in range(3):
            mask = results[target_choice]["name"] == name
            r = results[target_choice][mask]
            combined_estimate["estimate"][i] += r["Estimate"] / r["Variance"]
            combined_estimate["s.e."][i] += 1 / r["Variance"]

        combined_estimate["estimate"][i] /= combined_estimate["s.e."][i]
        combined_estimate["s.e."][i] = np.sqrt(1 / combined_estimate["s.e."][i])

        combined_estimate["t_stat"] = combined_estimate["estimate"] / combined_estimate["s.e."]
        combined_estimate["p_val"] = 2 * (1 - norm.cdf(np.abs(combined_estimate["t_stat"])))

    pd.DataFrame(combined_estimate).to_csv("combined_estimate.csv", index=False)



