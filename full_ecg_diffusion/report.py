"""JSON + Markdown + overlay plots for a realism evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import CLASS_ORDER, FS


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(obj), indent=2), encoding="utf-8")


def _f(v, nd=3):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def markdown(res: dict) -> str:
    d, r, f, c = res["detector"], res["rhythm"], res["features"], res["copying"]
    L = [f"# Realism report: {res['title']}", "",
         f"**Overall: {res['overall']}** (detector {res['detector_verdict']}, features "
         f"{'pass' if f['pass'] else 'fail'}, rhythm {'pass' if r['pass'] else 'fail'}, "
         f"copying {'pass' if c['pass'] else 'fail'})", "",
         f"Real set: `{', '.join(res['real_specs'])}`; harmonised through layout: {res['harmonized']}. "
         f"n per side: {res['n_per_class']}", "",
         "## Detector (real vs synthetic, 5-fold, patient-grouped)", "",
         f"- AUC **{_f(d['auc'])}** (95% CI {_f(d['auc_ci95'][0])}-{_f(d['auc_ci95'][1])}); "
         f"pass <= {res['thresholds']['detector_pass']}, fail >= {res['thresholds']['detector_fail']}",
         f"- per class: " + ", ".join(f"{k} {_f(v)}" for k, v in d["per_class_auc"].items()),
         f"- real-vs-real control AUC: {_f(res['control']['auc']) if res.get('control') else 'skipped'}", "",
         "## Rhythm checks", "",
         f"- R-peak detector calibrated on real: {r['detector_calibrated_on_real']}",
         "", "| class | real HR | syn HR | real rr_cv | syn rr_cv | syn detectable |", "|---|---|---|---|---|---|"]
    for n in CLASS_ORDER:
        if n in r["real"] and r["real"][n]["n"]:
            L.append(f"| {n} | {_f(r['real'][n]['median_hr'],1)} | {_f(r['syn'][n]['median_hr'],1)} | "
                     f"{_f(r['real'][n]['median_rr_cv'])} | {_f(r['syn'][n]['median_rr_cv'])} | "
                     f"{_f(r['syn'][n]['detectable_frac'],2)} |")
    L += ["", "- checks: " + ", ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in r["checks"].items()), "",
          "## Feature judge (KS, pooled over classes)", "",
          f"- median KS {_f(f['median_ks'])} (<= {res['thresholds']['feature_median_ks']}), max KS {_f(f['max_ks'])} "
          f"(<= {res['thresholds']['feature_max_ks']}); real-vs-real floor median {_f(f['real_vs_real_floor']['median_ks'])}",
          f"- worst: " + ", ".join(f"{k} {_f(f['pooled_ks'][k])}" for k in f["worst_features"]), "",
          "## Copying", "",
          f"- synthetic mean NN corr {_f(c['syn']['mean_nn_corr'])}, >{res['thresholds']['copy_corr']}: {_f(c['syn']['frac_gt_thr'])}; "
          f"held-out real mean {_f(c['heldout_real']['mean_nn_corr'])}, >thr: {_f(c['heldout_real']['frac_gt_thr'])} "
          f"(reference n={c['n_ref']})", ""]
    if res.get("notes"):
        L += ["## Notes", ""] + [f"- {n}" for n in res["notes"]] + [""]
    return "\n".join(L)


def plots(out_dir: Path, x_real, y_real, x_syn, y_syn, real_beats, syn_beats) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 4, figsize=(16, 3.2))
    for c, name in enumerate(CLASS_ORDER):
        hr_r = [b["hr"] for b, yy in zip(real_beats, y_real) if yy == c and b["detectable"]]
        hr_s = [b["hr"] for b, yy in zip(syn_beats, y_syn) if yy == c and b["detectable"]]
        bins = np.linspace(30, 180, 31)
        ax[c].hist(hr_r, bins, alpha=0.5, density=True, label="real")
        ax[c].hist(hr_s, bins, alpha=0.5, density=True, label="synthetic")
        ax[c].set_title(f"{name} HR (bpm)")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "hr_by_class.png", dpi=110)
    plt.close(fig)

    t = np.arange(x_real.shape[1]) / FS
    fig, ax = plt.subplots(4, 2, figsize=(14, 9), sharex=True)
    for c, name in enumerate(CLASS_ORDER):
        for k, (x, y, lab) in enumerate([(x_real, y_real, "real"), (x_syn, y_syn, "synthetic")]):
            idx = np.flatnonzero(y == c)
            if len(idx):
                ax[c, k].plot(t, x[idx[0], :, 1], lw=0.8)
            ax[c, k].set_title(f"{name} {lab}, lead II (z)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "examples_lead_ii.png", dpi=110)
    plt.close(fig)
