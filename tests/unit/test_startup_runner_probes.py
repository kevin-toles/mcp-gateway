"""Startup test must verify services start without leaving them running (2026-10-20).

Observed: the runner woke every on-demand service with a plain
``POST :8079/activity/{svc}`` — recorded as real use — so llama-server (~4.8 GB)
stayed Hot 30 min + Warm 10 min after every run, and the LaunchAgent also ran it
daily at 09:00. GREEN: the runner probes (``?probe=true``: start, verify, stop,
no activity recorded) one service at a time and hands the results to the test
suite, which must not re-wake services itself.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = (ROOT / "scripts" / "hwc_e2e_test_runner.sh").read_text()
STARTUP_TEST = (ROOT / "tests" / "integration" / "test_hwc_e2e_startup.py").read_text()


def test_runner_probes_instead_of_reporting_activity():
    assert '/activity/$svc?probe=true' in RUNNER
    assert '/activity/$svc"' not in RUNNER, "a plain /activity call records real use"


def test_runner_hands_probe_results_to_the_test_suite():
    assert "HWC_PROBE_RESULTS" in RUNNER
    assert "HWC_PROBE_RESULTS" in STARTUP_TEST


def test_startup_test_never_reports_activity():
    assert "/activity/{service}\")" not in STARTUP_TEST
    assert "probe=true" in STARTUP_TEST
