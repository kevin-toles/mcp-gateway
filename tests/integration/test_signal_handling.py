"""Integration test for shim-mcp-gateway signal handling — F11.

Verifies that the Rust shim's signal handlers (SIGTERM, SIGINT) work correctly:
  1. The binary starts and binds to :8090.
  2. Sending SIGTERM triggers the graceful shutdown path (RUNNING=false, drain loop).
  3. Sending SIGINT triggers the same graceful shutdown path.
  4. Return code is 0 after graceful signal handling.
  5. Stdout contains the expected "shim: shutting down" and "drain complete" messages.

The shim runs briefly (no client connections) and does NOT need the Python
mcp-gateway backend — signal handling is in the accept loop which starts
immediately after binding.

Run with:
    INTEGRATION=1 pytest tests/integration/test_signal_handling.py -m integration -v
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

# Paths — resolve relative to this file location.
REPO_ROOT = Path(__file__).resolve().parents[2]
CARGO_MANIFEST = REPO_ROOT / "Cargo.toml"
BINARY_PATH = REPO_ROOT / "target" / "debug" / "shim-mcp-gateway"

# How long to wait for the shim to start binding its TCP listener.
SHIM_STARTUP_WAIT = 3.0

# Max time to wait for the shim to exit after a signal.
SHIM_SHUTDOWN_WAIT = 10.0


# ── Helpers ────────────────────────────────────────────────────────────────


def _build_shim() -> Path:
    """Build the shim binary if it doesn't exist. Returns path to binary."""
    if BINARY_PATH.exists():
        return BINARY_PATH

    result = subprocess.run(
        ["cargo", "build", "--manifest-path", str(CARGO_MANIFEST)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"cargo build failed:\n{result.stdout}\n{result.stderr}")
    assert BINARY_PATH.exists(), f"Binary not found after build: {BINARY_PATH}"
    return BINARY_PATH


def _free_port_8090() -> None:
    """Kill any process holding port 8090 to avoid 'Address already in use'."""
    result = subprocess.run(
        ["lsof", "-ti:8090"],
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        pids = result.stdout.strip().split()
        subprocess.run(["kill", "-9"] + pids, capture_output=True)
        time.sleep(0.2)  # allow OS to release the port


def _read_stdout_nonblocking(
    proc: subprocess.Popen, timeout: float
) -> list[bytes]:
    """Read all available stdout without blocking, up to *timeout* seconds."""
    import select

    out: list[bytes] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.stdout is None or proc.stdout.closed:
            break
        r, _, _ = select.select([proc.stdout], [], [], 0.1)
        if r:
            try:
                line = proc.stdout.readline()
                if not line:
                    break
                out.append(line)
            except (ValueError, OSError):
                break
        else:
            # No data buffered yet — return what we have (outer loop retries).
            break
    return out


@pytest.fixture(scope="module")
def shim_binary() -> Path:
    """Build-once per module — return path to debug binary."""
    return _build_shim()


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture()
def shim_process(shim_binary: Path):
    """Start the shim, yield the Popen handle, then terminate.

    Kills any stale shim on :8090 first, then starts the binary.
    The shim binds to :8090 and enters its accept loop. No Python gateway
    is needed — the accept loop is signal-aware. Connections will get 502s
    but that's irrelevant for signal testing.
    """
    _free_port_8090()

    proc = subprocess.Popen(
        [str(shim_binary)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for the shim to bind and print its startup message.
    deadline = time.monotonic() + SHIM_STARTUP_WAIT
    started = False
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            pytest.fail(
                f"Shim exited prematurely with code {proc.returncode}"
            )
        # Read all available stdout without blocking.
        # Use a short timeout here because the outer loop handles retrying.
        lines = _read_stdout_nonblocking(proc, 0.1)
        for line in lines:
            if b"shim-mcp-gateway: listening on" in line:
                started = True
                break
        if started:
            break

    if not started:
        proc.kill()
        stderr = b""
        if proc.stderr:
            stderr = proc.stderr.read()
        pytest.fail(
            "Shim did not print startup message within timeout.\n"
            f"stderr: {stderr.decode(errors='replace')}"
        )

    yield proc

    # Cleanup: ensure the process is dead.
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)


# ── Tests ───────────────────────────────────────────────────────────────────


class TestSignalHandlingSIGTERM:
    """Tests for SIGTERM graceful shutdown."""

    def test_sigterm_causes_clean_exit(self, shim_process):
        """SIGTERM should cause the shim to exit with return code 0."""
        os.kill(shim_process.pid, signal.SIGTERM)
        try:
            shim_process.wait(timeout=SHIM_SHUTDOWN_WAIT)
        except subprocess.TimeoutExpired:
            shim_process.kill()
            pytest.fail("Shim did not exit within timeout after SIGTERM")
        assert shim_process.returncode == 0, (
            f"Expected return code 0 after SIGTERM, got {shim_process.returncode}"
        )

    def test_sigterm_stdout_shutdown_message(self, shim_process):
        """Signal handler writes shutdown message to stderr from signal context.

        The signal handler uses libc::write(STDERR_FILENO, ...) for async-signal-safety,
        so the message appears on stderr. The main loop also prints to both stdout and
        stderr after the accept loop drains, but that may be lost on SIGINT due to pipe
        buffer discipline. Check stderr as the reliable source.
        """
        os.kill(shim_process.pid, signal.SIGTERM)
        stdout, stderr = shim_process.communicate(timeout=SHIM_SHUTDOWN_WAIT)
        assert b"shim: shutting down (SIGTERM/SIGINT received)" in stderr or \
               b"shim: shutting down (SIGTERM/SIGINT received)" in stdout, (
            f"Expected shutdown message in stderr (or stdout). Got:\n"
            f"  stdout: {stdout.decode()}\n"
            f"  stderr: {stderr.decode()}"
        )

    def test_sigterm_stdout_drain_message(self, shim_process):
        """Drain message should appear in stdout or stderr after SIGTERM."""
        os.kill(shim_process.pid, signal.SIGTERM)
        stdout, stderr = shim_process.communicate(timeout=SHIM_SHUTDOWN_WAIT)
        assert b"shim: drain complete, exiting" in stdout or \
               b"shim: drain complete, exiting" in stderr, (
            f"Expected drain message in stdout or stderr. Got:\n"
            f"  stdout: {stdout.decode()}\n"
            f"  stderr: {stderr.decode()}"
        )

    def test_sigterm_second_signal_kills_immediately(self, shim_process):
        """A second SIGTERM should re-raise via SIG_DFL and kill the process.

        After the first SIGTERM sets RUNNING=false, the handler re-registers
        SIG_DFL. A second SIGTERM should therefore cause the process to exit
        with a signal termination (128 + SIGTERM = 143), not return code 0.

        Must call communicate() to consume stdout/stderr pipes and prevent
        the subprocess from hanging on a full pipe buffer.
        """
        os.kill(shim_process.pid, signal.SIGTERM)
        # Wait briefly — the first signal causes a graceful shutdown with drain,
        # so we give it time to set RUNNING=false without forcing the process
        # to drain the full 5s window.
        time.sleep(0.5)
        os.kill(shim_process.pid, signal.SIGTERM)
        stdout, stderr = shim_process.communicate(timeout=SHIM_SHUTDOWN_WAIT)
        # On macOS/Linux, a process killed by SIGTERM after SIG_DFL re-registration
        # will either exit 143 (128+SIGTERM) or -15 / 0 depending on timing.
        # We accept any non-zero or signal-based exit.
        assert shim_process.returncode != 0, (
            "Expected non-zero exit after second SIGTERM (SIG_DFL re-registration), "
            f"got {shim_process.returncode}. stdout: {stdout.decode(errors='replace')}"
        )


class TestSignalHandlingSIGINT:
    """Tests for SIGINT graceful shutdown."""

    def test_sigint_causes_clean_exit(self, shim_process):
        """SIGINT should cause the shim to exit with return code 0.

        Must call communicate() to consume stdout/stderr pipes and prevent
        the subprocess from hanging on a full pipe buffer.
        """
        os.kill(shim_process.pid, signal.SIGINT)
        try:
            stdout, stderr = shim_process.communicate(timeout=SHIM_SHUTDOWN_WAIT)
        except subprocess.TimeoutExpired:
            shim_process.kill()
            pytest.fail("Shim did not exit within timeout after SIGINT")
        assert shim_process.returncode == 0, (
            f"Expected return code 0 after SIGINT, got {shim_process.returncode}. "
            f"stdout: {stdout.decode(errors='replace')}"
        )

    def test_sigint_stdout_shutdown_message(self, shim_process):
        """Shutdown message should appear in stderr after SIGINT.

        On macOS, SIGINT terminates the foreground process group, which can cause
        stdout pipe buffer loss. The signal handler writes to stderr via async-signal-safe
        libc::write, so stderr is the reliable assertion target.
        """
        os.kill(shim_process.pid, signal.SIGINT)
        stdout, stderr = shim_process.communicate(timeout=SHIM_SHUTDOWN_WAIT)
        assert b"shim: shutting down (SIGTERM/SIGINT received)" in stderr or \
               b"shim: shutting down (SIGTERM/SIGINT received)" in stdout, (
            f"Expected shutdown message in stderr (or stdout). Got:\n"
            f"  stdout: {stdout.decode()}\n"
            f"  stderr: {stderr.decode()}"
        )

    def test_sigint_stdout_drain_message(self, shim_process):
        """Drain message should appear in stdout or stderr after SIGINT."""
        os.kill(shim_process.pid, signal.SIGINT)
        stdout, stderr = shim_process.communicate(timeout=SHIM_SHUTDOWN_WAIT)
        assert b"shim: drain complete, exiting" in stdout or \
               b"shim: drain complete, exiting" in stderr, (
            f"Expected drain message in stdout or stderr. Got:\n"
            f"  stdout: {stdout.decode()}\n"
            f"  stderr: {stderr.decode()}"
        )

    def test_sigint_second_signal_kills_immediately(self, shim_process):
        """A second SIGINT should re-raise via SIG_DFL and kill the process.

        Must call communicate() to consume stdout/stderr pipes and prevent
        the subprocess from hanging on a full pipe buffer.
        """
        os.kill(shim_process.pid, signal.SIGINT)
        time.sleep(0.5)
        os.kill(shim_process.pid, signal.SIGINT)
        stdout, stderr = shim_process.communicate(timeout=SHIM_SHUTDOWN_WAIT)
        assert shim_process.returncode != 0, (
            "Expected non-zero exit after second SIGINT (SIG_DFL re-registration), "
            f"got {shim_process.returncode}. stdout: {stdout.decode(errors='replace')}"
        )


class TestSignalHandlingNoSignal:
    """Baseline: shim without signals stays alive (and can be terminated)."""

    def test_shim_starts_and_binds(self, shim_process):
        """Shim process should be alive after startup with no signal sent."""
        assert shim_process.poll() is None, "Shim exited before any signal was sent"
        os.kill(shim_process.pid, signal.SIGTERM)
        shim_process.wait(timeout=SHIM_SHUTDOWN_WAIT)
