import numpy as np

from driftsentry.drift import DriftMonitor, population_stability_index


def test_psi_near_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=5000), rng.normal(size=5000)
    assert population_stability_index(a, b) < 0.05


def test_psi_large_for_shifted_distribution():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=5000), rng.normal(1.5, 1.0, size=5000)
    assert population_stability_index(a, b) > 0.25


def test_psi_handles_constant_reference():
    assert population_stability_index(np.zeros(100), np.zeros(50)) < 1e-3
    assert population_stability_index(np.zeros(100), np.ones(50)) > 0.25


def test_monitor_flags_shift_but_not_same():
    rng = np.random.default_rng(1)
    ref_err = rng.gamma(2.0, 0.1, size=3000)
    ref_feat = rng.normal(size=(3000, 2))
    monitor = DriftMonitor(ref_err, ref_feat, ["f0", "f1"])

    same = monitor.check(rng.gamma(2.0, 0.1, size=1000), rng.normal(size=(1000, 2)))
    assert not same["drift_detected"]

    shifted_feat = np.column_stack([rng.normal(2.0, 1.0, 1000), rng.normal(size=1000)])
    shifted = monitor.check(rng.gamma(2.0, 0.3, size=1000), shifted_feat)
    assert shifted["drift_detected"]
    assert shifted["error_drift"]["drifted"]
    assert shifted["drifted_features"] == ["f0"]
