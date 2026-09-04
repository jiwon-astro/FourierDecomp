from pathlib import Path

import numpy as np
import pandas as pd

from FourierDecomp.catalog import (
    build_gaia_fit_quality_table,
    merge_manifest_revisions,
    revision_jobs_from_manifest,
    upsert_revision_decision,
    validate_revision_manifest,
)
from FourierDecomp.period_finder import (
    blocked_time_cv_period_score,
    build_period_candidate_bank,
)


def _fd_frame():
    return pd.DataFrame({
        "ID": ["1", "2"],
        "pulsation": ["DCEP_FUNDAMENTAL", "DCEP_FUNDAMENTAL"],
        "P": [2.0, 3.0],
        "E": [10.0, 10.0],
        "M_fit": [3, 3],
        "A1": [0.5, 0.5],
        "A2": [0.2, 0.2],
        "A3": [0.1, 0.1],
        "Q1": [0.1, 0.1],
        "Q2": [0.3, 0.3],
        "Q3": [0.4, 0.4],
        "N_g": [40, 40], "N_bp": [35, 35], "N_rp": [35, 35],
        "sig_g": [0.3, 0.3], "sig_bp": [0.3, 0.3], "sig_rp": [0.3, 0.3],
        "rms_g": [0.03, 0.03], "rms_bp": [0.04, 0.04], "rms_rp": [0.04, 0.04],
        "gmax_g": [0.1, 0.1], "gmax_bp": [0.1, 0.1], "gmax_rp": [0.1, 0.1],
        "flag": [0, 0],
    })


def test_gaia_external_r21_outlier_is_review_only_when_periods_match():
    gaia = pd.DataFrame({
        "SOURCE_ID": ["1", "2"],
        "pf": [2.0, 3.0],
        "p1_o": [np.nan, np.nan],
        "type_best_classification": ["DCEP", "DCEP"],
        "mode_best_classification": ["FUNDAMENTAL", "FUNDAMENTAL"],
        "r21_g": [0.41, 0.90],
        "r21_g_error": [0.01, 0.01],
        "r31_g": [0.21, 0.21],
        "r31_g_error": [0.02, 0.02],
    })
    audit = build_gaia_fit_quality_table(_fd_frame(), gaia)
    by_id = audit.set_index("ID")
    assert not bool(by_id.at["1", "R21_gaia_outlier"])
    assert bool(by_id.at["2", "R21_gaia_outlier"])
    assert "gaia_R21_outlier" in by_id.at["2", "review_reasons"]

    gaia.loc[gaia["SOURCE_ID"].eq("2"), "pf"] = 30.0
    audit = build_gaia_fit_quality_table(_fd_frame(), gaia).set_index("ID")
    assert not bool(audit.at["2", "R21_gaia_outlier"])
    assert np.isnan(audit.at["2", "R21_external_z"])
    assert "gaia_period_disagreement" in audit.at["2", "review_reasons"]


def test_period_candidate_bank_deduplicates_same_family():
    bank = build_period_candidate_bank(
        123, base_period=2.0, gaia_period=2.00001,
        rrfit_periods=[3.0], manual_periods=[4.0], include_half_double=True)
    assert len(bank) == 4
    assert "base;gaia" in set(bank["sources"])


def test_blocked_time_cv_prefers_the_generating_period():
    rng = np.random.default_rng(17)
    t = np.sort(rng.uniform(0.0, 80.0, 120))
    mag = 16.0 + 0.3 * np.sin(2.0 * np.pi * t / 2.5) + rng.normal(0.0, 0.01, len(t))
    error = np.full(len(t), 0.01)
    bands = np.array(["g"] * len(t))
    true_score = blocked_time_cv_period_score((t, mag, error, bands), 2.5)
    wrong_score = blocked_time_cv_period_score((t, mag, error, bands), 2.0)
    assert true_score["cv_rmse"] < wrong_score["cv_rmse"]


def test_one_source_manifest_update_and_job_selection(tmp_path):
    manifest_path = tmp_path / "period_decisions.csv"
    manifest = upsert_revision_decision(
        manifest_path, "1", base_period=2.0, proposed_period=2.1,
        decision="adopt_period", confidence="clear",
        reason_code="coherent_all_bands", reviewer="tester")
    validate_revision_manifest(manifest, base_catalog=_fd_frame())
    ids, period_map = revision_jobs_from_manifest(manifest, base_catalog=_fd_frame())
    assert ids == [1]
    assert period_map == {"1": 2.1}


def test_manifest_merge_never_changes_base_file(tmp_path):
    base = _fd_frame()
    refit = base.loc[base["ID"].eq("1")].copy()
    refit["P"] = 2.1
    base_path = tmp_path / "base.dat"
    refit_path = tmp_path / "refit.dat"
    manifest_path = tmp_path / "decisions.csv"
    output_path = tmp_path / "revised.dat"
    base.to_csv(base_path, sep=" ", index=False)
    refit.to_csv(refit_path, sep=" ", index=False)
    before = base_path.read_bytes()
    upsert_revision_decision(
        manifest_path, "1", base_period=2.0, proposed_period=2.1,
        decision="adopt_period", confidence="clear")
    result = merge_manifest_revisions(
        base_path, refit_path, manifest_path, output_path)
    revised = pd.read_csv(output_path, sep=r"\s+", dtype={"ID": str})
    assert result["refit_selected"] == 1
    assert revised.set_index("ID").at["1", "P"] == 2.1
    assert base_path.read_bytes() == before
