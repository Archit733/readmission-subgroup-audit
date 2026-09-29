#!/usr/bin/env python3
"""
Main pipeline: load config -> train models -> run subgroup audit -> test
mechanisms -> save all results to reports/. This replaces the original
01-08 numbered scripts with one config-driven, logged, tested pipeline.

Usage:
    python scripts/run_pipeline.py                  # uses config.yaml
    python scripts/run_pipeline.py --config other.yaml
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import joblib
import pandas as pd

from readmit_audit.config import load_config
from readmit_audit.data import load_raw, prepare_features, split_data, get_subgroups
from readmit_audit.modeling import build_baseline_pipeline, build_strong_pipeline, train_model, predict_proba_positive
from readmit_audit.audit import bootstrap_auc_ci, subgroup_audit, groups_differ, expected_calibration_error
from readmit_audit.serving import make_serving_artifact
from readmit_audit.fairness import top_fraction_threshold
from readmit_audit.mechanism import compute_charlson_scores, bin_diagnosis_count, bin_charlson_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("pipeline")


def main(config_path: str):
    cfg = load_config(config_path)
    results_dir = Path(cfg["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load & prepare ----
    df = load_raw(cfg)
    X = prepare_features(df, cfg)
    X_train, X_test, y_train, y_test, idx_train, idx_test = split_data(df, X, cfg)
    subgroups = get_subgroups(df, idx_test, cfg)
    logger.info(f"Split: {len(X_train)} train / {len(X_test)} test")

    # ---- Baseline model ----
    baseline = build_baseline_pipeline(X_train, cfg)
    baseline = train_model(baseline, X_train, y_train)
    baseline_probs = predict_proba_positive(baseline, X_test)
    baseline_point, baseline_lo, baseline_hi = bootstrap_auc_ci(
        y_test, baseline_probs, n_boot=cfg["audit"]["n_bootstrap"],
        random_state=cfg["split"]["random_state"],
    )
    logger.info(f"Baseline AUC: {baseline_point:.4f} [{baseline_lo:.4f}, {baseline_hi:.4f}]")

    # ---- Strong model (robustness check) ----
    strong = build_strong_pipeline(X_train, cfg)
    strong = train_model(strong, X_train, y_train)
    strong_probs = predict_proba_positive(strong, X_test)
    strong_point, strong_lo, strong_hi = bootstrap_auc_ci(
        y_test, strong_probs, n_boot=cfg["audit"]["n_bootstrap"],
        random_state=cfg["split"]["random_state"],
    )
    logger.info(f"Strong model AUC: {strong_point:.4f} [{strong_lo:.4f}, {strong_hi:.4f}]")

    # ---- Subgroup audit (on the strong model -- it's the more defensible one) ----
    y_test_reset = y_test.reset_index(drop=True)
    AGE_ORDER = [f"[{i}-{i+10})" for i in range(0, 100, 10)]
    group_orders = {"age": AGE_ORDER}  # age needs an explicit chronological order; others use natural appearance order
    all_subgroup_results = {}
    for col in cfg["data"]["subgroup_columns"]:
        result = subgroup_audit(y_test_reset, strong_probs, subgroups[col], cfg,
                                 group_order=group_orders.get(col))
        all_subgroup_results[col] = result
        logger.info(f"Subgroup audit [{col}]:\n{result.to_string(index=False)}")
        result.to_csv(results_dir / f"subgroup_audit_{col}.csv", index=False)

    # ---- Calibration ----
    ece_overall = expected_calibration_error(y_test_reset, strong_probs, cfg["audit"]["n_calibration_bins"])
    logger.info(f"Overall ECE: {ece_overall:.4f}")

    # ---- Mechanism: direct diagnosis-count binning ----
    num_diag = df.loc[idx_test, "number_diagnoses"].reset_index(drop=True)
    diag_bins = num_diag.apply(bin_diagnosis_count)
    diag_result = subgroup_audit(y_test_reset, strong_probs, diag_bins, cfg,
                                  group_order=[str(i) for i in range(1, 10)] + ["10+"])
    logger.info(f"Diagnosis-count binning (direct test):\n{diag_result.to_string(index=False)}")
    diag_result.to_csv(results_dir / "mechanism_diagnosis_binning.csv", index=False)

    # ---- Mechanism: Charlson comorbidity index ----
    logger.info("Computing Charlson scores (diabetes component excluded by design)...")
    charlson_scores = compute_charlson_scores(df)
    charlson_test = charlson_scores.loc[idx_test].reset_index(drop=True)
    charlson_bins = charlson_test.apply(bin_charlson_score)
    charlson_result = subgroup_audit(y_test_reset, strong_probs, charlson_bins, cfg,
                                      group_order=["0", "1-2", "3-4", "5+"])
    logger.info(f"Charlson index binning:\n{charlson_result.to_string(index=False)}")
    charlson_result.to_csv(results_dir / "mechanism_charlson.csv", index=False)

    # ---- Model-invariance conclusion ----
    # Compare the best- and worst-performing age bands by POINT ESTIMATE, not
    # by position in the table -- the literal youngest/oldest bands can have
    # small n and wide CIs that overlap everything, which would wrongly look
    # like "no real gap" even when a clear gap exists elsewhere in the range.
    age_result = all_subgroup_results["age"].dropna(subset=["auc"])
    if len(age_result) >= 2:
        best = age_result.loc[age_result["auc"].idxmax()]
        worst = age_result.loc[age_result["auc"].idxmin()]
        gap_is_real = groups_differ(best, worst)
        logger.info(
            f"Best vs worst age band: {best['group']} (AUC={best['auc']:.3f}) vs "
            f"{worst['group']} (AUC={worst['auc']:.3f}) -- statistically real gap: {gap_is_real}"
        )

    # ---- Save artifacts for downstream use (dashboard, API, etc.) ----
    joblib.dump({
        "baseline_model": baseline, "strong_model": strong,
        "baseline_auc": (baseline_point, baseline_lo, baseline_hi),
        "strong_auc": (strong_point, strong_lo, strong_hi),
        "y_test": y_test_reset, "strong_probs": strong_probs, "baseline_probs": baseline_probs,
        "subgroups": subgroups, "X_test": X_test,
        "ece_overall": ece_overall,
        "charlson_test": charlson_test, "num_diag_test": num_diag,
    }, results_dir / "pipeline_artifacts.joblib")

    flag_fraction = cfg.get("serving", {}).get("flag_fraction", 0.20)
    flag_threshold = top_fraction_threshold(strong_probs, flag_fraction)
    joblib.dump(make_serving_artifact(
        strong, X_train.columns, all_subgroup_results["age"], strong_point, flag_threshold, flag_fraction,
        reference=X_train.sample(min(20000, len(X_train)), random_state=cfg["split"]["random_state"])),
        results_dir / "model_for_serving.joblib")

    logger.info(f"Done. All results saved to {results_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
