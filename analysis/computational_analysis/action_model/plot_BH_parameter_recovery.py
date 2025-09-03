#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _metrics(true, est):
    true = np.asarray(true, dtype=float)
    est  = np.asarray(est, dtype=float)
    mask = np.isfinite(true) & np.isfinite(est)
    true, est = true[mask], est[mask]
    if true.size == 0:
        return dict(r=np.nan, rmse=np.nan, bias=np.nan, slope=np.nan, intercept=np.nan)
    r = np.corrcoef(true, est)[0, 1]
    rmse = float(np.sqrt(np.mean((est - true) ** 2)))
    bias = float(np.mean(est - true))
    slope, intercept = np.polyfit(true, est, 1)
    return dict(r=float(r), rmse=rmse, bias=bias, slope=float(slope), intercept=float(intercept))

def _scatter(true, est, title, out_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(true, est)
    # identity line
    mn = np.nanmin([np.min(true), np.min(est)])
    mx = np.nanmax([np.max(true), np.max(est)])
    pad = 0.05 * (mx - mn if mx > mn else 1.0)
    lo, hi = mn - pad, mx + pad
    xs = np.linspace(lo, hi, 100)
    ax.plot(xs, xs)
    ax.set_xlabel("True")
    ax.set_ylabel("Estimated")
    ax.set_title(title)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def main(seed):
    csv_path = Path(f"results/BH_action_parameter_recovery_seed={seed}.csv")
    outdir = Path("results")
    outdir.mkdir(parents=True, exist_ok=True)

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    pairs = [
        ("lr_low",   "est_lr_low",   "LR (low noise)",   outdir / "BH_action_recovery_lr_low.png"),
        ("lr_high",  "est_lr_high",  "LR (high noise)",  outdir / "BH_action_recovery_lr_high.png"),
        ("beta_low", "est_beta_low", "Beta (low noise)", outdir / "BH_action_recovery_beta_low.png"),
        ("beta_high","est_beta_high","Beta (high noise)",outdir / "BH_action_recovery_beta_high.png"),
    ]
 
    rows = []
    for tcol, ecol, label, fname in pairs:
        if tcol not in df.columns or ecol not in df.columns:
            rows.append({"param": label, "r": np.nan, "rmse": np.nan, "bias": np.nan, "slope": np.nan, "intercept": np.nan, "file": str(fname)})
            continue
        m = _metrics(df[tcol], df[ecol])
        rows.append({"param": label, **m, "file": str(fname)})
        _scatter(df[tcol].values, df[ecol].values, f"{label}\nr={m['r']:.3f}, RMSE={m['rmse']:.3g}", fname)

    summary = pd.DataFrame(rows)
    summary_path = outdir / "action_recovery_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(summary.to_string(index=False))
    print(f"\nSaved CSV: {summary_path}")
    for _, row in summary.iterrows():
        print(f"Saved plot: {row['file']}")

if __name__ == "__main__":
    main(42)
