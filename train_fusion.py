"""Train the quality-index fusion stage from a dataset.npz (VMAF distillation).

Usage:
  python train_fusion.py dataset.npz --out model.json [--test-frac 0.25]
                         [--seed 0] [--no-gbt]

Protocol (as agreed in planning):
  - content-grouped split: no source content straddles train/test
  - naturalness anchor fit on reference rows (fallback: top-decile VMAF)
  - RidgeFusion baseline always trained; GBT (if xgboost installed) must
    beat ridge by > 0.02 pooled held-out SRCC to be selected
  - bundle saved as versioned JSON (ridge + anchor inline; GBT sidecar)
"""

import argparse
import datetime
import json
import math

import numpy as np

from adaptive_filters.dataset import load_dataset
from adaptive_filters.features.stats import spearman, pearson
from adaptive_filters.fusion import RidgeFusion
from adaptive_filters.naturalness import NaturalnessModel

GBT_MARGIN = 0.02  # ship ridge on ties


def grouped_split(groups, test_frac, seed):
    contents = sorted(set(groups))
    rng = np.random.default_rng(seed)
    rng.shuffle(contents)
    n_test = max(1, int(round(test_frac * len(contents))))
    test_set = set(contents[:n_test])
    tr = [i for i, g in enumerate(groups) if g not in test_set]
    te = [i for i, g in enumerate(groups) if g in test_set]
    return np.array(tr), np.array(te), sorted(test_set)


def evaluate(y_true, y_pred, groups):
    per_content = {}
    for c in sorted(set(groups)):
        m = [i for i, g in enumerate(groups) if g == c]
        if len(m) >= 3:
            per_content[c] = spearman(y_true[m], y_pred[m])
    rmse = math.sqrt(float(np.mean((y_true - y_pred) ** 2)))
    return {
        "srcc_pooled": spearman(y_true, y_pred),
        "plcc_pooled": pearson(y_true, y_pred),
        "rmse": rmse,
        "srcc_per_content": per_content,
    }


def pick_ridge_alpha(x, y, groups, seed):
    """Small inner grouped CV over the alpha grid; falls back to 5.0."""
    contents = sorted(set(groups))
    if len(contents) < 3:
        return 5.0
    rng = np.random.default_rng(seed + 1)
    val_content = contents[int(rng.integers(len(contents)))]
    val = np.array([i for i, g in enumerate(groups) if g == val_content])
    tr = np.array([i for i, g in enumerate(groups) if g != val_content])
    best_alpha, best_srcc = 5.0, -2.0
    for alpha in (0.1, 1.0, 5.0, 10.0, 50.0, 100.0):
        model = RidgeFusion(alpha=alpha).fit(x[tr], y[tr])
        s = spearman(y[val], model.predict(x[val]))
        if s > best_srcc:
            best_alpha, best_srcc = alpha, s
    return best_alpha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", help="dataset .npz from build_dataset.py")
    ap.add_argument("--out", default="model.json", help="output bundle path")
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--holdout", default=None,
                    help="comma-separated content ids to hold out "
                         "(overrides the seeded split; for exact LOO)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-gbt", action="store_true")
    ap.add_argument("--exclude", default=None,
                    help="regex: drop matching features (ablations, e.g. "
                         "'_ldct_' to remove the learned probe)")
    args = ap.parse_args()

    d = load_dataset(args.dataset)
    x, y, names = d["x"], d["y"], d["names"]
    is_ref = d["is_ref"]

    if args.exclude:
        import re
        rx = re.compile(args.exclude)
        keep = [i for i, n in enumerate(names) if not rx.search(n)]
        dropped = len(names) - len(keep)
        x = x[:, keep]
        names = [names[i] for i in keep]
        print(f"Ablation: dropped {dropped} features matching "
              f"'{args.exclude}'")

    # supervised rows exclude the reference corpus
    sup = ~is_ref
    xs, ys = x[sup], y[sup]
    gs = [g for g, r in zip(d["groups"], is_ref) if not r]

    if args.holdout:
        test_set = set(args.holdout.split(","))
        missing = test_set - set(gs)
        if missing:
            raise SystemExit(f"holdout content(s) not in dataset: {missing}")
        tr = np.array([i for i, g in enumerate(gs) if g not in test_set])
        te = np.array([i for i, g in enumerate(gs) if g in test_set])
        test_contents = sorted(test_set)
    else:
        tr, te, test_contents = grouped_split(gs, args.test_frac, args.seed)
    g_te = [gs[i] for i in te]
    print(f"Rows: {len(ys)} supervised + {int(is_ref.sum())} reference | "
          f"features: {len(names)}")
    print(f"Held-out contents: {', '.join(test_contents)}\n")

    # ---- naturalness anchor (layer B)
    if is_ref.any():
        anchor_rows = x[is_ref]
        anchor_desc = f"{len(anchor_rows)} reference frames"
    else:
        thr = np.percentile(ys, 90)
        anchor_rows = xs[ys >= thr]
        anchor_desc = f"{len(anchor_rows)} top-decile frames (no refs in dataset)"
    anchor = NaturalnessModel().fit(anchor_rows)
    print(f"Anchor fit on {anchor_desc}")

    # ---- ridge baseline (layer A)
    g_tr = [gs[i] for i in tr]
    alpha = pick_ridge_alpha(xs[tr], ys[tr], g_tr, args.seed)
    ridge = RidgeFusion(alpha=alpha).fit(xs[tr], ys[tr])
    ridge_metrics = evaluate(ys[te], ridge.predict(xs[te]), g_te)
    print(f"\nRidge (alpha={alpha}): held-out SRCC={ridge_metrics['srcc_pooled']:.3f} "
          f"PLCC={ridge_metrics['plcc_pooled']:.3f} RMSE={ridge_metrics['rmse']:.2f}")
    for c, s in ridge_metrics["srcc_per_content"].items():
        print(f"    {c}: SRCC={s:.3f}")

    # ---- optional GBT challenger
    model_type, gbt_metrics, gbt = "ridge", None, None
    if not args.no_gbt:
        try:
            from adaptive_filters.fusion import GbtFusion
            gbt = GbtFusion().fit(xs[tr], ys[tr])
            gbt_metrics = evaluate(ys[te], np.asarray(gbt.predict(xs[te]),
                                                      dtype=np.float64), g_te)
            print(f"\nGBT: held-out SRCC={gbt_metrics['srcc_pooled']:.3f} "
                  f"PLCC={gbt_metrics['plcc_pooled']:.3f} "
                  f"RMSE={gbt_metrics['rmse']:.2f}")
            if gbt_metrics["srcc_pooled"] > ridge_metrics["srcc_pooled"] + GBT_MARGIN:
                model_type = "gbt"
            print(f"Selected: {model_type} "
                  f"(GBT must beat ridge by >{GBT_MARGIN} SRCC)")
        except ImportError as e:
            print(f"\nGBT skipped: {e}")

    # ---- save bundle
    bundle = {
        "version": 1,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "feature_names": names,
        "model_type": model_type,
        "ridge": ridge.to_dict(),
        "anchor": anchor.to_dict(),
        "gbt_file": None,
        "metrics": {"ridge": ridge_metrics, "gbt": gbt_metrics,
                    "held_out_contents": test_contents},
    }
    if model_type == "gbt":
        gbt_path = args.out + ".xgb.json"
        gbt.model.save_model(gbt_path)
        bundle["gbt_file"] = gbt_path
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=1)
    print(f"\nSaved bundle -> {args.out}")


if __name__ == "__main__":
    main()
