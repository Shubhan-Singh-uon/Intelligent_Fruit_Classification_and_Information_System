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

Usage (run from the project root):
    python src/scripts/export_ood_stats.py --embeddings artifacts/M4_PC1_embeddings.npz --ood-csv artifacts/ood_examples_results_PC1.csv --checkpoint artifacts/M4_PC1.pt

--checkpoint is optional but recommended: with the classifier head it also reports how genuine and
unsupported images would be split into Accepted / Needs review / Rejected, for the thresholds
currently in config.py and for a small grid of alternatives. The confidence gate matters as much
as the distance gate, so tune them together.

Inputs come from the training notebook's own outputs (Sections 16 and 18). Only the
embeddings file is required; --ood-csv adds the false-acceptance columns.
No images, dataset or checkpoint are needed. The statistics are rebuilt exactly as in
notebook Section 16: class centroids from the training features, one shared
Ledoit-Wolf covariance of the centred features, pseudo-inverse.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch  # only needed for this optional report
from sklearn.covariance import LedoitWolf

# Allow `python src/scripts/export_ood_stats.py` from the project root.
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


def head_confidence(features, weight, bias):
    """Highest class probability for each feature vector, from the 2-class linear head."""
    logits = features @ weight.T + bias
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs.max(axis=1)


def outcome_shares(dist, conf, unstable, distance_threshold, confidence_threshold):
    """Percent of images that the policy would Accept / send to Review / Reject (same rule order as
    ml/prediction_policy.py: too far -> Rejected; then low confidence or unstable -> Needs review)."""
    dist, conf, unstable = np.asarray(dist), np.asarray(conf), np.asarray(unstable, dtype=bool)
    rejected = dist > distance_threshold
    review = ~rejected & ((conf < confidence_threshold) | unstable)
    accepted = ~rejected & ~review
    return {"accept": 100 * accepted.mean(), "review": 100 * review.mean(), "reject": 100 * rejected.mean()}


def outcome_table(test_dist, test_conf, ood, model_id, settings):
    """One row per (distance, confidence) setting: outcome shares for genuine test images and, with OOD
    data, for unsupported images. Genuine images are assumed stable under the test-time views."""
    rows = []
    for distance_threshold, confidence_threshold, label in settings:
        genuine = outcome_shares(test_dist, test_conf, np.zeros(len(test_dist), bool),
                                 distance_threshold, confidence_threshold)
        row = {"setting": label, "T": distance_threshold, "C": confidence_threshold,
               "genuine_accept_%": round(genuine["accept"], 1), "genuine_review_%": round(genuine["review"], 1),
               "genuine_reject_%": round(genuine["reject"], 1)}
        if ood is not None:
            unsupported = outcome_shares(
                ood[f"{model_id}_feature_distance"].to_numpy(), ood[f"{model_id}_confidence"].to_numpy(),
                (ood[f"{model_id}_tta_consistency"] == "LOW").to_numpy(), distance_threshold, confidence_threshold)
            row["unsupported_ACCEPT_%"] = round(unsupported["accept"], 1)
            row["unsupported_review_%"] = round(unsupported["review"], 1)
            row["unsupported_reject_%"] = round(unsupported["reject"], 1)
        rows.append(row)
    return pd.DataFrame(rows)


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
    parser.add_argument("--checkpoint", help="M4_PC1.pt: adds the Accepted / Needs review / Rejected report")
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

    if args.checkpoint:
        head = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        test_conf = head_confidence(test_f, head["classifier.1.weight"].numpy(), head["classifier.1.bias"].numpy())
        settings = [(DISTANCE_REJECT_THRESHOLD, CONFIDENCE_THRESHOLD, "config.py now")]
        settings += [(t, c, "") for t in (60, 65, 70) for c in (0.99, 0.999, 0.9995)]
        print("\nHow the policy would split images (distance gate T, confidence gate C)")
        print("  genuine_* : real Apple/Banana test images. Want: mostly accept, few review, almost no reject.")
        print("  unsupported_ACCEPT_% : unsupported images that would still be accepted. Want: close to 0.")
        print(outcome_table(test_dist, test_conf, ood, args.model_id, settings).to_string(index=False))
        print("Small samples (about 100 unsupported images, tuned on the same images): treat differences of a few points as noise.")


if __name__ == "__main__":
    main()
