import numpy as np
import pytest

import AmpMap as am


def stripes(n, k1, k2, amplitude=1.0, phase=0.0):
    """A single density wave with wavevector (k1, k2) FFT bins on an n x n image."""
    y, x = np.mgrid[:n, :n]
    return amplitude * np.cos(2 * np.pi * (k1 * x + k2 * y) / n + phase)


def lockin(image, k1, k2, sigma=2.0):
    n = image.shape[0]
    c = n // 2                                   # zero frequency of the fftshifted FFT
    v1 = np.array([c + k1, c + k2], dtype=float)
    v2 = np.array([c + k2, c + k1], dtype=float)
    z = am.CDWampmap2D(image, np.array([0.0, 0.0]), v1, v2, sigma, "plasma")
    inner = (slice(n // 8, -(n // 8)), slice(n // 8, -(n // 8)))
    return np.abs(z)[inner], np.unwrap(np.unwrap(np.angle(z)[inner], axis=1), axis=0)


@pytest.mark.parametrize("n", [128, 127])        # even and odd image sizes
def test_amplitude_of_pure_wave_is_half_its_amplitude(n):
    amp, _ = lockin(stripes(n, 16, 8, amplitude=2.0), 16, 8)
    assert amp.mean() == pytest.approx(1.0, rel=0.02)
    assert np.ptp(amp) < 0.02


@pytest.mark.parametrize("n", [128, 127])
def test_lockin_phase_of_pure_wave_is_constant(n):
    _, phase = lockin(stripes(n, 16, 8), 16, 8)
    assert np.ptp(phase) < 0.05


def test_phase_follows_shift_of_the_wave():
    # cos(q.r + phi) demodulated with reference point r0 = 0 has lock-in phase phi
    _, phase = lockin(stripes(128, 16, 8, phase=np.pi / 2), 16, 8)
    assert np.angle(np.exp(1j * (np.median(phase) - np.pi / 2))) == pytest.approx(0, abs=0.05)


def test_output_shape_and_type():
    z = am.CDWampmap2D(stripes(64, 8, 4), np.array([0.0, 0.0]),
                       np.array([40.0, 36.0]), np.array([36.0, 40.0]), 3.0, "plasma")
    assert z.shape == (64, 64) and np.iscomplexobj(z)
