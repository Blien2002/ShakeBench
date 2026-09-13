"""Minimal coverage for the retained harmonic transfer-fit math."""

from __future__ import annotations

import numpy as np
import pytest

from robosuite.utils.shakebench_isolator import compare_harmonic_fit, fit_harmonic_transfer


def test_harmonic_transfer_fit_recovers_a_known_gain_and_phase():
    frequency_hz = 2.5
    time = np.linspace(0.0, 8.0, 8001)
    input_signal = np.sin(2.0 * np.pi * frequency_hz * time)
    gain, phase = 1.8, 0.4
    output_signal = gain * np.sin(2.0 * np.pi * frequency_hz * time + phase)

    fit = fit_harmonic_transfer(time, input_signal, output_signal, frequency_hz)

    assert abs(fit.transfer_complex) == pytest.approx(gain, rel=1e-6)
    assert np.angle(fit.transfer_complex) == pytest.approx(phase, abs=1e-6)
    assert compare_harmonic_fit(fit, fit.transfer_complex)["passed"] is True
