# combine_lmer_fixed_effects.py
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import norm
from pymer4.models import Lmer

# ──────────────────────────────────────────────────────────────────────────────
# Settings
# ──────────────────────────────────────────────────────────────────────────────
class Setting:
    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = (BASE_DIR / "../../../data/preprocessed/preprocessed_data.csv").resolve()
    PERIOD = 0
    OUTDIR = (BASE_DIR / "results").resolve()

    @staticmethod
    def create_dirs():
        (Setting.OUTDIR / "per_model").mkdir(parents=True, exist_ok=True)
        (Setting.OUTDIR / "combined").mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Data engineering
# ──────────────────────────────────────────────────────────────────────────────
def create_dataset(df, target_choice, period):
    """
    Build a trial-level panel with lagged self/partner choice and reward
    codes ready for mixed-effects modelling.
    """
    out = df.copy()

    # 1) Current-trial codes
    out["C_t"]  = np.where(out["self_choice"]    == target_choice,  1, -1)
    out["PC_t"] = np.where(out["partner_choice"] == target_choice,  1, -1)
    out["PR_t"] = np.where(
        out["partner_reward"].eq(0), 0,
        np.where(out["partner_choice"].eq(target_choice), 1, -1)
    )

    # 2) Order so shifts line up
    out = out.sort_values(["id", "block", "trial"])

    # 3) Lagged columns
    lagged = {}
    keys = ["C_t", "PC_t", "PR_t"]
    gb = out.groupby(["id", "block"])
    for lag in range(period + 1):
        for key in keys:
            lagged[f"{key[:-2]}_minus_{lag}"] = gb[key].shift(lag)

    out = pd.concat([out, pd.DataFrame(lagged, index=out.index)], axis=1)

    # 4) Keep only full windows
    out = out.dropna(subset=lagged).reset_index(drop=True)

    # 5) Binary outcome
    out["choice_Y"] = (out["C_t"] == 1).astype(int)

    # 6) Z-score PC/PR lags (C lags stay ±1)
    z_cols = [c for c in lagged if c.startswith(("PC", "PR"))]
    out["is_partner_high_exp"] = np.where(out["is_partner_high_exp"], 1, -1)
    out[z_cols] = out[z_cols].apply(lambda s: (s - s.mean()) / s.std(ddof=0))

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Model fitting
# ──────────────────────────────────────────────────────────────────────────────
def run_regression(data, period, outcome="choice_Y", interaction_var="is_partner_high_exp"):
    """
    Fit mixed-effects logistic regression with PR/PC lags interacting with interaction_var.
    """
    pr_terms = [f"PR_minus_{i}" for i in range(0, period + 1)]
    pc_terms = [f"PC_minus_{i}" for i in range(0, period + 1)]
    all_terms = pr_terms + pc_terms
    fixed_effects = " + ".join(all_terms)

    formula = (
        f"{outcome} ~ ({fixed_effects}) * {interaction_var} "
        f"+ (1 + ({fixed_effects}) * {interaction_var} | id)"
    )

    model = Lmer(formula, data=data, family="binomial")
    summary_df = model.fit(control="optimizer='bobyqa', optCtrl = list(maxfun=5e5)")

    return model, summary_df.reset_index(names="name")


# ──────────────────────────────────────────────────────────────────────────────
# Extract fixed effects + VCV from pymer4, robust to version differences
# ──────────────────────────────────────────────────────────────────────────────
def get_fixef_and_vcov(model):
    """
    Extract fixed-effect estimates and their full variance-covariance matrix
    from a fitted pymer4 Lmer object, robust to pymer4/rpy2 versions.

    Returns
    -------
    fixef : pd.Series
    vcov  : pd.DataFrame
    """
    # 1) Fast path: pymer4 already exposes them
    b = getattr(model, "fixef", None)
    V = getattr(model, "vcov", None)
    if b is not None and V is not None:
        if not isinstance(b, pd.Series):
            b = pd.Series(b)
        if not isinstance(V, pd.DataFrame):
            V = pd.DataFrame(V, index=b.index, columns=b.index)
        # enforce identical order / names
        V = V.loc[b.index, b.index]
        return b, V

    # 2) Fallback: pull from R's lme4 object via rpy2 with proper conversion
    try:
        from rpy2.robjects import r
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects import default_converter
        from rpy2.robjects import numpy2ri

        # underlying merMod/glmerMod
        mer = model.model_obj

        # vcov can be a Matrix::dpoMatrix; coerce to base matrix first
        V_r = r['vcov'](mer)
        V_r_mat = r['as.matrix'](V_r)

        # convert to numpy 2-D array
        with localconverter(default_converter + numpy2ri.converter):
            V_np = np.asarray(V_r_mat)

        # get names from the R object to guarantee alignment
        rn = list(r['rownames'](V_r_mat))
        cn = list(r['colnames'](V_r_mat))

        # fixed-effect vector (pymer4 Series is preferred for names)
        if hasattr(model, "fixef") and isinstance(model.fixef, pd.Series):
            b = model.fixef
        else:
            # fall back to the summary/coefs table
            coefs = getattr(model, "coefs", None)  # older pymer4
            if coefs is not None and isinstance(coefs, pd.DataFrame) and "Estimate" in coefs:
                b = pd.Series(coefs["Estimate"].values, index=coefs.index)
            else:
                # As a last resort, use R's fixef and convert
                b_r = r['fixef'](mer)
                with localconverter(default_converter + numpy2ri.converter):
                    b_np = np.asarray(r['as.vector'](b_r))
                b_names = list(r['names'](b_r))
                b = pd.Series(b_np, index=b_names)

        # build DataFrame; then reindex to b’s order to keep consistency
        V_df = pd.DataFrame(V_np, index=rn, columns=cn)

        # ensure that all beta names exist in VCV and align
        common = [nm for nm in b.index if nm in V_df.index]
        if not common:
            raise RuntimeError("No overlapping coefficient names between fixef and vcov.")

        b = b.loc[common]
        V_df = V_df.loc[common, common]

        return b, V_df

    except Exception as e:
        raise RuntimeError(
            "Could not extract VCV from pymer4 via rpy2. "
            "Confirm rpy2 and lme4 are installed and importable, and that the model fitted successfully."
        ) from e

# ──────────────────────────────────────────────────────────────────────────────
# Multivariate fixed-effects combiner (assumes independence across models)
# ──────────────────────────────────────────────────────────────────────────────
def combine_fixed_effects_full_vcv(models: list[Lmer]):
    """
    Combine same fixed-effect vector across models using full inverse-variance weighting.
    Returns:
      beta_comb (pd.Series), V_comb (pd.DataFrame)
    """
    betas, Vs, names_list = [], [], []
    for m in models:
        b, V = get_fixef_and_vcov(m)
        names_list.append(list(b.index))
        betas.append(b)
        Vs.append(V)

    # align on common names (order of first model)
    base = names_list[0]
    common = [nm for nm in base if all(nm in nl for nl in names_list)]
    if not common:
        raise ValueError("No common fixed-effect names across models.")

    betas = [b.loc[common] for b in betas]
    Vs = [V.loc[common, common] for V in Vs]

    # sum of precisions and precision-weighted betas
    precisions = [np.linalg.inv(V.values) for V in Vs]
    S = np.zeros_like(precisions[0])
    t = np.zeros(len(common))
    for P, b in zip(precisions, betas):
        S += P
        t += P @ b.values

    Vc = np.linalg.inv(S)
    bc = Vc @ t

    beta_comb = pd.Series(bc, index=common, name="Estimate")
    V_comb = pd.DataFrame(Vc, index=common, columns=common)
    return beta_comb, V_comb


# ──────────────────────────────────────────────────────────────────────────────
# (Optional) Diagonal-only combiner to mirror your original approach
# ──────────────────────────────────────────────────────────────────────────────
def combine_fixed_effects_diag_only(summaries: list[pd.DataFrame]):
    """
    Combine per-coefficient using only diagonal variances from summary tables.
    Returns a DataFrame with columns: name, Estimate, SE, z, p
    """
    # Start from the coefficient set of the first summary
    names = summaries[0]["name"].tolist()
    est = np.zeros(len(names))
    wsum = np.zeros(len(names))

    for summ in summaries:
        # must have Estimate, SE, and name columns
        s = summ.set_index("name").loc[names]
        var = s["SE"].values ** 2
        est += s["Estimate"].values / var
        wsum += 1.0 / var

    est /= wsum
    se = np.sqrt(1.0 / wsum)
    z = est / se
    p = 2 * (1 - norm.cdf(np.abs(z)))

    return pd.DataFrame({"name": names, "Estimate": est, "SE": se, "z": z, "p": p})


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    Setting.create_dirs()
    data = pd.read_csv(Setting.DATA_PATH)

    models = []
    summaries = []

    # Fit one model per target_choice ∈ {0,1,2}
    for target_choice in range(3):
        out = create_dataset(data, target_choice, Setting.PERIOD)
        model, summary_df = run_regression(out, period=Setting.PERIOD)
        models.append(model)
        summaries.append(summary_df)

        # Save per-model artifacts
        per_dir = Setting.OUTDIR / "per_model" / f"target_choice={target_choice}"
        per_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(per_dir / "summary.csv", index=False)

        # Save per-model fixef and VCV
        b, V = get_fixef_and_vcov(model)
        b.to_csv(per_dir / "fixef.csv", header=["Estimate"])
        V.to_csv(per_dir / "vcov.csv")

    # ── Combined (full VCV) ───────────────────────────────────────────────────
    beta_comb, V_comb = combine_fixed_effects_full_vcv(models)
    se_comb = np.sqrt(np.diag(V_comb))
    z_comb = beta_comb.values / se_comb
    p_comb = 2 * (1 - norm.cdf(np.abs(z_comb)))
    combined_table = pd.DataFrame({
        "name": beta_comb.index,
        "Estimate": beta_comb.values,
        "SE": se_comb,
        "z": z_comb,
        "p": p_comb,
    })
    combined_dir = Setting.OUTDIR / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    combined_table.to_csv(combined_dir / "combined_fixed_effects_full_vcv.csv", index=False)
    V_comb.to_csv(combined_dir / "combined_vcov_full.csv")

    # ── Optional: Diagonal-only combo (to compare with your original approach)
    diag_only = combine_fixed_effects_diag_only(summaries)
    diag_only.to_csv(combined_dir / "combined_fixed_effects_diag_only.csv", index=False)

    # Small console summary
    print("\nSaved per-model summaries and VCVs under:", Setting.OUTDIR / "per_model")
    print("Saved combined results under:", combined_dir)
    print("\nTop of combined (full VCV) table:")
    print(combined_table.head())


if __name__ == "__main__":
    main()
