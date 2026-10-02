import numpy as np
import pytest

import tools

N = 128
K = 16                        # lattice frequency of the reference, in FFT bins
C = N // 2                    # fftshift puts the zero frequency at N // 2
G_VECTORS = [(C + K, C), (C, C + K)]
INNER = (slice(16, -16), slice(16, -16))   # ignore edge effects of the FFT mask


def lattice(scale_x=1.0, scale_y=1.0):
    """Square lattice whose period is `scale` times the reference period N / K."""
    y, x = np.mgrid[:N, :N]
    return np.cos(2 * np.pi * K * x / (N * scale_x)) + np.cos(2 * np.pi * K * y / (N * scale_y))


def strains(image):
    exx, eyy, exy, omega, phases = tools.standard_gpa(image, G_VECTORS, mask_radius=4, sigma=0)
    return [m[INNER].mean() for m in (exx, eyy, exy, omega)], phases


def test_perfect_lattice_has_zero_strain():
    (exx, eyy, exy, omega), _ = strains(lattice())
    assert exx == pytest.approx(0, abs=1e-3)
    assert eyy == pytest.approx(0, abs=1e-3)
    assert exy == pytest.approx(0, abs=1e-3)
    assert omega == pytest.approx(0, abs=1e-3)


@pytest.mark.parametrize("scale", [1.02, 0.97])
def test_uniform_isotropic_strain(scale):
    expected = (scale - 1) / scale        # strain relative to the reference g-vectors
    (exx, eyy, exy, omega), _ = strains(lattice(scale, scale))
    assert exx == pytest.approx(expected, abs=1e-3)
    assert eyy == pytest.approx(expected, abs=1e-3)
    assert exy == pytest.approx(0, abs=1e-3)


def test_strain_along_x_only():
    (exx, eyy, exy, omega), _ = strains(lattice(scale_x=1.02))
    assert exx == pytest.approx(0.02 / 1.02, abs=1e-3)
    assert eyy == pytest.approx(0, abs=1e-3)


def test_geometric_phase_of_perfect_lattice_is_flat():
    _, phases = strains(lattice())
    assert len(phases) == 2
    for phase in phases:
        assert np.ptp(phase[INNER]) < 0.05


def test_returns_zero_maps_with_fewer_than_two_g_vectors():
    exx, eyy, exy, omega, phases = tools.standard_gpa(lattice(), G_VECTORS[:1], mask_radius=4)
    assert phases == []
    for m in (exx, eyy, exy, omega):
        assert m.shape == (N, N) and not m.any()
