from pathlib import Path

import numpy as np
import pandas as pd

from FourierDecomp import LC, RRFit, decomposition, quality
from FourierDecomp.LSQ import LSQ_fit, unpack_theta
from FourierDecomp.IO import build_fd_header
from FourierDecomp.IO import RRFitLC
from FourierDecomp.catalog import (
    build_gaia_fit_quality_table,
    merge_manifest_revisions,
    revision_jobs_from_manifest,
    upsert_revision_decision,
    validate_revision_manifest,
)
from FourierDecomp.period_finder import (
    blocked_time_cv_period_score,
    bootstrap_period_candidate_support,
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


def test_rrfit_template_parser_and_sine_convention(tmp_path):
    template_path = tmp_path / "templates.dat"
    template_path.write_text(
        "test.t1\n"
        "9.999 9.999 0.5 999 9.999\n"
        "1.0 0.0\n"
        "0.0 0.0\n"
    )
    templates = RRFit.load_rrfit_templates(template_path)
    phase = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    curve = RRFit.evaluate_rrfit_template(templates[1], phase)
    assert np.allclose(curve, [0.0, 1.0, 0.0, -1.0, 0.0], atol=1e-12)


def test_rrfit_solution_families_are_clustered_without_auto_selection():
    summary = pd.DataFrame({
        "source_id": ["1", "1", "1"],
        "bandpair": ["g+bp", "g+rp", "g+bp"],
        "template_index": [1, 2, 3],
        "epoch": [10.0, 10.1, 11.0],
        "period": [2.0, 2.01, 5.0],
        "amp_1": [0.3, 0.3, 0.3],
        "amp_2": [0.2, 0.2, 0.2],
        "mean_1": [15.0, 15.0, 15.0],
        "mean_2": [15.5, 14.5, 15.5],
        "chi2": [10.0, 11.0, 20.0],
        "returncode": [0, 0, 0],
    })
    families = RRFit.build_rrfit_solution_families(
        summary, relative_tolerance=0.01)
    assert families["period_family_index"].nunique() == 2
    two_day = families.loc[families["period"].lt(3.0)]
    assert two_day["solution_id"].nunique() == 1
    assert int(two_day["n_bandpairs"].iloc[0]) == 2


def test_rrfit_fixed_period_jobs_do_not_open_a_period_window(tmp_path):
    source = RRFitLC(
        sid="1", fitlc_path=str(tmp_path / "1.fitlc"),
        t=np.linspace(0.0, 10.0, 30),
        mag=np.zeros(30), emag=np.full(30, 0.02),
        bands=np.resize(np.array(["g", "bp", "rp"]), 30),
    )
    jobs, metadata = RRFit.build_rrfit_jobs(
        source, tmp_path, mode="gaia", fixed_period=2.345,
        bandpairs=(("g", "bp"), ("g", "rp")), save=False)
    assert len(jobs) == 2
    assert metadata["period_mode"] == "fixed"
    assert np.isclose(metadata["fixed_period"], 2.345)
    assert all(job.pmin == job.pmax == 2.345 for job in jobs)
    assert {job.bandpair for job in jobs} == {"g+bp", "g+rp"}


def test_periodic_curve_diagnostics_identifies_gap_only_region():
    phase = np.linspace(0.0, 1.0, 256, endpoint=False)
    curve = np.sin(2.0 * np.pi * phase)
    observed = np.r_[np.linspace(0.0, 0.35, 20),
                     np.linspace(0.65, 0.99, 20)]
    diagnostic = quality.periodic_curve_diagnostics(
        phase, curve, observed_phase=observed, gap_threshold=0.20)
    assert diagnostic["curve_amplitude"] > 1.9
    assert diagnostic["gap_curvature_p95"] > 0
    assert diagnostic["tip_nearest_observation"] >= 0


def test_phase_aligned_curve_error_removes_phase_origin():
    phase = np.linspace(0.0, 1.0, 128, endpoint=False)
    reference = np.sin(2.0 * np.pi * phase) + 0.2 * np.sin(4.0 * np.pi * phase)
    candidate = np.roll(reference, 17)
    result = quality.phase_aligned_curve_error(reference, candidate)
    assert result["shape_rmse"] < 1e-12
    assert result["tip_phase_error"] < 1e-12


def test_same_lightcurve_soft_anchor_penalizes_added_ripple_without_template():
    phase = np.r_[np.linspace(0.0, 0.36, 35, endpoint=False),
                  np.linspace(0.64, 1.0, 35, endpoint=False)]
    t = 20.0 + 2.0 * phase
    mag = 15.0 + 0.30 * np.cos(2.0 * np.pi * phase)
    error = np.full(t.size, 0.02)
    args = (t, mag, error, [np.ones(t.size, dtype=bool)])
    anchor = decomposition.build_data_driven_soft_anchor(
        2.0, args, np.array([0]), epoch0=20.0, order=3,
        n_grid=200, global_floor=0.05, tolerance=0.05)

    theta3 = LSQ_fit(
        2.0, args, 3, np.array([0]), opt_method="lsq",
        quality_weight=True, epoch0=20.0, coef_mode="ab")
    m0, amp, alpha, beta, period, epoch = unpack_theta(
        theta3, 1, 3, include_amp=True, coef_mode="ab")
    alpha6 = np.zeros(6)
    beta6 = np.zeros(6)
    alpha6[:3] = alpha
    beta6[:3] = beta
    theta_smooth = np.hstack([m0, amp, alpha6, beta6, period, epoch])
    theta_ripple = theta_smooth.copy()
    theta_ripple[2 + 5] = 0.15

    smooth_penalty = decomposition.data_driven_soft_anchor_penalty(
        theta_smooth, 1, 6, anchor, coef_mode="ab")
    ripple_penalty = decomposition.data_driven_soft_anchor_penalty(
        theta_ripple, 1, 6, anchor, coef_mode="ab")
    assert anchor["kind"] == "same-lightcurve-low-order"
    assert np.max(anchor["weights"]) > np.min(anchor["weights"])
    assert smooth_penalty < 1e-10
    assert ripple_penalty > smooth_penalty + 0.1


def test_rrfit_initializer_uses_period_epoch_but_not_template_bank(monkeypatch):
    period = 2.4
    epoch = 100.0
    phase = np.linspace(0.0, 1.0, 90, endpoint=False)
    t = epoch + period * phase
    mag = 15.0 + 0.25 * np.cos(2.0 * np.pi * phase)
    error = np.full(t.size, 0.02)
    bands = np.array(["I"] * t.size)
    monkeypatch.setattr(
        decomposition, "df_ident",
        pd.DataFrame({"ID": ["test"], "pulsation": ["unknown"]}),
        raising=False)
    monkeypatch.setattr(
        decomposition, "df_rrfit",
        pd.DataFrame({
            "ID": ["test"], "P": [period], "EPOCH": [epoch],
            "T": [1],
        }), raising=False)
    monkeypatch.setattr(decomposition, "templates", None, raising=False)
    row = decomposition.fourier_decomp(
        "test", mode="ogle", init="rrfit", period_fit=False,
        use_optim=False, adaptive_lam=False, use_refit=False,
        epoch_data=(t, mag, error, bands))
    assert row is not None
    record = dict(zip(build_fd_header("ogle"), row))
    assert np.isclose(float(record["P"]), period)
    assert np.isclose(float(record["E"]), epoch)


def test_rrfit_period_relation_labels_harmonic_and_window_alias():
    harmonic = RRFit.period_relation_to_reference(2.5, 5.0)
    assert harmonic["alias_relation"] == "harmonic_2"
    window_period = 1.0 / (1.0 / 5.0 + 1.0)
    alias = RRFit.period_relation_to_reference(
        window_period, 5.0, alias_frequencies=[1.0])
    assert alias["alias_relation"] == "window_m1+fw"


def test_rrfit_review_plot_reads_saved_template_without_fd(tmp_path, monkeypatch):
    t = np.linspace(0.0, 12.0, 45)
    bands = np.resize(np.array(["g", "bp", "rp"]), len(t))
    mag = 15.0 + 0.2 * np.sin(2.0 * np.pi * t / 2.0)
    error = np.full(len(t), 0.02)
    monkeypatch.setattr(
        LC, "epoch_arrays",
        lambda *args, **kwargs: (t, mag, error, bands))
    summary = pd.DataFrame({
        "source_id": ["1", "1"], "bandpair": ["g+bp", "g+rp"],
        "template_index": [1, 1], "epoch": [0.0, 0.0],
        "period": [2.0, 2.0], "amp_1": [0.2, 0.2],
        "amp_2": [0.2, 0.2], "mean_1": [15.0, 15.0],
        "mean_2": [15.0, 15.0], "chi2": [10.0, 11.0],
        "returncode": [0, 0],
    })
    templates = {1: RRFit.RRFitTemplate(
        1, "test", np.array([1.0]), np.array([0.0]))}
    output = tmp_path / "review.png"
    figure, solutions = LC.plot_rrfit_source_review(
        "1", summary, templates, ls_data={}, output_path=output)
    assert output.exists()
    assert solutions["solution_id"].nunique() == 1
    import matplotlib.pyplot as plt
    plt.close(figure)


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


def test_block_bootstrap_support_prefers_generating_period():
    rng = np.random.default_rng(23)
    t = np.sort(rng.uniform(0.0, 90.0, 90))
    mag = 15.0 + 0.25 * np.sin(2.0 * np.pi * t / 3.0)
    error = np.full(len(t), 0.02)
    bands = np.array(["g"] * len(t))
    support = bootstrap_period_candidate_support(
        (t, mag, error, bands), [3.0, 2.2], n_boot=8,
        random_state=1, order=2, n_splits=3)
    assert support.iloc[0]["period"] == 3.0
    assert support.iloc[0]["bootstrap_support"] > 0.5


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
