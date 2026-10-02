import numpy as np
import pytest

import tools


def lattice(n=96, k=12):
    y, x = np.mgrid[:n, :n]
    return 0.5 + 0.25 * (np.cos(2 * np.pi * k * x / n) + np.cos(2 * np.pi * k * y / n))


@pytest.mark.parametrize("angle", [10, 33.5, -45])
def test_rotated_image_is_finite(angle):
    rotated = tools.rotate_image(lattice(), angle)
    assert np.isfinite(rotated).all()
    assert rotated.shape[0] == rotated.shape[1]          # the app needs square images


@pytest.mark.parametrize("angle", [10, -45])
def test_filters_and_fft_after_rotation_stay_finite(angle):
    rotated = tools.rotate_image(lattice(), angle)
    assert np.isfinite(tools.apply_gaussian_lowpass(rotated, 2)).all()
    assert np.isfinite(tools.apply_median_filter(rotated, 3)).all()
    assert np.isfinite(np.fft.fft2(rotated - rotated.mean())).all()


def test_corners_are_filled_with_the_image_mean():
    image = lattice()
    rotated = tools.rotate_image(image, 45)
    assert rotated[0, 0] == pytest.approx(image.mean())
    assert rotated[-1, -1] == pytest.approx(image.mean())


def test_rotation_by_90_matches_rot90():
    image = lattice()
    np.testing.assert_allclose(tools.rotate_image(image, 90), np.rot90(image), atol=1e-6)


def test_zero_rotation_is_identity():
    image = lattice()
    np.testing.assert_allclose(tools.rotate_image(image, 0), image, atol=1e-9)
