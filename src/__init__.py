"""
PS-06 Competition System package

Runtime compatibility hooks for third-party dependencies.
"""


def _ensure_scipy_signal_hann() -> None:
    """Restore scipy.signal.hann removed in SciPy 1.11+ for libs that still expect it."""
    try:
        import scipy.signal  # noqa: F401
        from scipy.signal import windows as signal_windows
    except Exception:
        return

    if not hasattr(scipy.signal, "hann") and hasattr(signal_windows, "hann"):
        try:
            scipy.signal.hann = signal_windows.hann  # type: ignore[attr-defined]
        except Exception:
            pass


_ensure_scipy_signal_hann()
