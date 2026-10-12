"""Scalar uplink model: power/noise PSDs in W/Hz, efficiency in bit/s/Hz."""

from math import isfinite, log, log1p


def dbm_per_hz_to_watts_per_hz(value: float) -> float:
    """Convert a power spectral density from dBm/Hz to W/Hz."""
    return 10.0 ** ((value - 30.0) / 10.0)


NOISE_PSD_W_PER_HZ = dbm_per_hz_to_watts_per_hz(-174.0)
TX_PSD_W_PER_HZ = dbm_per_hz_to_watts_per_hz(1.0)


def spectral_efficiency(sinr_linear: float) -> float:
    """Shannon efficiency; log1p preserves small, positive linear SINRs."""
    if not isfinite(sinr_linear) or sinr_linear < 0.0:
        raise ValueError("SINR must be finite, non-negative, and in linear units.")
    return log1p(sinr_linear) / log(2.0)
