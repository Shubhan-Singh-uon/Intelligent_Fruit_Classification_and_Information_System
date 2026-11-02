"""
File: export_ood_stats.py
Project: Intelligent Fruit Classification and Information System
Description:
    Builds artifacts/m4_ood_stats.npz (class centroids and inverse covariance of the
    deployed model's penultimate features), which the application needs to compute the
    feature-distance signal at inference time. Also helps choose DISTANCE_REJECT_THRESHOLD
    in config.py with evidence, using the agreed calibration rule:

        1. Coverage: keep at least 95% of genuine Apple/Banana test images.
        2. Then check the false-acceptance rate: how many unsupported (OOD) images still pass.

    Retaining 95% of genuine images does not by itself show that enough unsupported images
    are rejected, so both numbers are always printed together.

Author: Shubhan Singh
Responsible Pillar: Pillar 2 - Machine Learning & Evaluation

---------------------------------------------------------------------------------------------
Inputs come from the ML notebook's own outputs (Sections 16 and 18). Only the
embeddings file is required; --ood-csv adds the false-acceptance columns.
No images, dataset or checkpoint are needed. The statistics are rebuilt exactly as in
notebook Section 16: class centroids from the training features, one shared Ledoit-Wolf 
covariance of the centred features, pseudo-inverse.
---------------------------------------------------------------------------------------------
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

# Allow `python src/scripts/export_ood_stats.py` from the project root (same usage as init_db.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.fruit_classifier.config import (  # noqa: E402
    CLASS_NAMES,
    CONFIDENCE_THRESHOLD,
    DISTANCE_REJECT_THRESHOLD,
    OOD_STATS_PATH,
)

# Value printed by the notebook for M4_PC1 (cutoff_analysis_PC1.csv, "configured_cutoff").
NOTEBOOK_TRAIN_P95_M4_PC1 = 35.7255744934082

# Agreed calibration target: share of genuine test images the distance gate must keep.
RETENTION_TARGET = 0.95


def build_stats(train_f, train_y):
    """Class centroids and the inverse of one shared Ledoit-Wolf covariance (notebook Section 16)."""
    centroids = np.stack([train_f[train_y == c].mean(axis=0) for c in range(len(CLASS_NAMES))])
    residuals = train_f - centroids[train_y]
    cov_inv = np.linalg.pinv(LedoitWolf().fit(residuals).covariance_)
    return centroids, cov_inv


def retention_cutoff(test_dist, target=RETENTION_TARGET):
    """Smallest observed distance such that at least `target` of the genuine test images are <= it."""
    return float(np.quantile(test_dist, target, method="inverted_cdf"))


def threshold_table(test_dist, ood, model_id, cutoffs, notes=None):
    """One row per candidate cutoff: genuine images kept, and (with OOD data) unsupported images let through."""
    notes = notes or {}
    rows = []
    for cutoff in cutoffs:
        row = {"cutoff": round(cutoff, 2), "test_kept_%": round(100 * (test_dist <= cutoff).mean(), 1)}
        if ood is not None:
            dist_ok = ood[f"{model_id}_feature_distance"] <= cutoff
            conf_ok = ood[f"{model_id}_confidence"] >= CONFIDENCE_THRESHOLD
            tta_ok = ood[f"{model_id}_tta_consistency"] != "LOW"
            wrongly_accepted = dist_ok & conf_ok & tta_ok
            row["ood_pass_distance_%"] = round(100 * dist_ok.mean(), 1)
            row["ood_wrongly_accepted_%"] = round(100 * wrongly_accepted.mean(), 1)
            by_label = wrongly_accepted.groupby(ood["label"]).mean()
            row["worst_ood_category"] = (
                "none" if by_label.max() == 0 else f"{by_label.idxmax()} ({100 * by_label.max():.0f}%)"
            )
        row["note"] = notes.get(round(cutoff, 2), "")
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    # Imported here so that the helpers above can be used (and tested) without torch installed.
    from src.ml.model_service import mahalanobis_min_distance

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings", required=True, help="M4_PC1_embeddings.npz from the training notebook")
    parser.add_argument("--ood-csv", help="ood_examples_results_PC1.csv from the training notebook")
    parser.add_argument("--model-id", default="M4", help="column prefix used in the OOD csv (default M4)")
    parser.add_argument("--out", default=str(OOD_STATS_PATH), help="where to write the statistics file")
    args = parser.parse_args()

    data = np.load(args.embeddings)
    missing = {"train_f", "train_y", "test_f"} - set(data.files)
    if missing:
        sys.exit(f"{args.embeddings} is missing arrays {sorted(missing)}; found {data.files}.")
    train_f, train_y, test_f = data["train_f"], data["train_y"], data["test_f"]

    centroids, cov_inv = build_stats(train_f, train_y)
    train_dist = mahalanobis_min_distance(train_f, centroids, cov_inv)
    test_dist = mahalanobis_min_distance(test_f, centroids, cov_inv)
    train_p95 = float(np.percentile(train_dist, 95))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, centroids=centroids, cov_inv=cov_inv)
    print(f"Saved {out_path}  (centroids {centroids.shape}, cov_inv {cov_inv.shape})")

    # Parity check: the notebook reported this exact number for M4_PC1.
    relative_gap = abs(train_p95 - NOTEBOOK_TRAIN_P95_M4_PC1) / NOTEBOOK_TRAIN_P95_M4_PC1
    verdict = "MATCHES the notebook" if relative_gap < 0.01 else "DIFFERS from the notebook - check the embeddings file"
    print(f"\n95th percentile of training distances: {train_p95:.4f}  ({verdict}: {NOTEBOOK_TRAIN_P95_M4_PC1:.4f} for M4_PC1)")
    print(
        f"Test distances: median {np.median(test_dist):.2f}, 90th pct {np.percentile(test_dist, 90):.2f}, "
        f"max {test_dist.max():.2f}  ({len(test_dist)} images)"
    )

    ood = None
    if args.ood_csv:
        ood = pd.read_csv(args.ood_csv)
        needed = [f"{args.model_id}_{s}" for s in ("confidence", "feature_distance", "tta_consistency")] + ["label"]
        absent = [c for c in needed if c not in ood.columns]
        if absent:
            sys.exit(f"{args.ood_csv} is missing columns {absent}.")
        print(f"OOD examples: {len(ood)} images")

    target_cutoff = retention_cutoff(test_dist)
    notes = {
        round(target_cutoff, 2): f"<- keeps {RETENTION_TARGET:.0%} of test images (agreed target)",
        round(train_p95, 2): "<- training 95th pct (notebook default)",
    }
    notes[round(DISTANCE_REJECT_THRESHOLD, 2)] = (notes.get(round(DISTANCE_REJECT_THRESHOLD, 2), "") + " <- config.py now").strip()
    cutoffs = sorted({train_p95, 40.0, 45.0, 50.0, 60.0, 70.0, target_cutoff, DISTANCE_REJECT_THRESHOLD})
    table = threshold_table(test_dist, ood, args.model_id, cutoffs, notes)

    print("\nWhat each candidate DISTANCE_REJECT_THRESHOLD would do")
    print("  test_kept_%            genuine Apple/Banana test images kept (coverage; want >= 95)")
    if ood is not None:
        print("  ood_pass_distance_%    OOD images that pass the distance gate alone")
        print("  ood_wrongly_accepted_% OOD images that pass all three signals = false acceptance (want low)")
    print(table.to_string(index=False))

    print(f"\nAgreed rule: keep >= {RETENTION_TARGET:.0%} of genuine images, then check false acceptance.")
    print(f"  Cutoff that meets the coverage target: {target_cutoff:.2f}")
    if ood is not None:
        row = threshold_table(test_dist, ood, args.model_id, [target_cutoff]).iloc[0]
        print(
            f"  At that cutoff {row['ood_wrongly_accepted_%']}% of OOD images are still wrongly accepted "
            f"({row['ood_pass_distance_%']}% pass the distance gate alone; worst category: {row['worst_ood_category']})."
        )
        print("  If that false-acceptance rate is too high, the distance signal alone cannot meet both goals.")
        print("  The OOD set has only ~100 images and was also used to tune cutoffs, so treat these rates as optimistic.")
    print(f"Currently set in config.py: DISTANCE_REJECT_THRESHOLD = {DISTANCE_REJECT_THRESHOLD}")


if __name__ == "__main__":
    main()