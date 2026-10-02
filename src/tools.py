# tools.py - Image processing tools for STM/TEM images
import numpy as np
from scipy import ndimage, signal
from scipy.ndimage import median_filter, gaussian_filter
from skimage import exposure
from skimage.restoration import wiener, denoise_nl_means


def apply_wiener_filter(image, noise_variance=None):
    """
    Apply Wiener filter for image denoising.

    Parameters:
    -----------
    image : ndarray
        Input image
    noise_variance : float, optional
        Estimated noise variance. If None, estimated automatically.

    Returns:
    --------
    ndarray : Filtered image
    """
    if noise_variance is None:
        noise_variance = np.var(image) * 0.01  # Default: 1% of image variance

    # Estimate PSF (Point Spread Function) - using simple Gaussian
    psf = np.ones((5, 5)) / 25
    filtered = wiener(image, psf, balance=noise_variance)
    return filtered


def apply_absf_filter(image, sigma=1.0):
    """
    Apply Adaptive Bilateral Smoothing Filter (ABSF).

    Parameters:
    -----------
    image : ndarray
        Input image
    sigma : float
        Standard deviation for Gaussian kernel

    Returns:
    --------
    ndarray : Filtered image
    """
    # Simplified ABSF using bilateral-like approach
    from skimage.filters import gaussian

    # Create range and spatial kernels
    spatial = gaussian_filter(image, sigma=sigma)

    # Combine spatial and range information
    diff = image - spatial
    filtered = spatial + 0.5 * diff

    return filtered


def apply_nonlinear_filter(image, patch_size=7, patch_distance=11, h=0.1):
    """
    Apply Non-Local Means denoising filter.

    Parameters:
    -----------
    image : ndarray
        Input image
    patch_size : int
        Size of patches used for denoising
    patch_distance : int
        Maximum distance to search for similar patches
    h : float
        Cut-off distance (higher h means more smoothing)

    Returns:
    --------
    ndarray : Filtered image
    """
    # Normalize image to [0, 1] for NLM
    img_norm = (image - image.min()) / (image.max() - image.min())

    filtered = denoise_nl_means(
        img_norm,
        patch_size=patch_size,
        patch_distance=patch_distance,
        h=h,
        fast_mode=True
    )

    # Denormalize
    filtered = filtered * (image.max() - image.min()) + image.min()
    return filtered


def apply_butterworth_lowpass(image, cutoff_freq=0.2, order=2):
    """
    Apply Butterworth lowpass filter in frequency domain.

    Parameters:
    -----------
    image : ndarray
        Input image
    cutoff_freq : float
        Normalized cutoff frequency (0 to 0.5)
    order : int
        Filter order (higher = sharper cutoff)

    Returns:
    --------
    ndarray : Filtered image
    """
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    # Create frequency grid
    u = np.fft.fftfreq(rows)
    v = np.fft.fftfreq(cols)
    U, V = np.meshgrid(u, v, indexing='ij')
    D = np.sqrt(U**2 + V**2)

    # Butterworth filter
    H = 1 / (1 + (D / cutoff_freq)**(2 * order))

    # Apply filter in frequency domain
    F = np.fft.fft2(image)
    F_filtered = F * H
    filtered = np.real(np.fft.ifft2(F_filtered))

    return filtered


def apply_gaussian_lowpass(image, sigma=2.0):
    """
    Apply Gaussian lowpass filter.

    Parameters:
    -----------
    image : ndarray
        Input image
    sigma : float
        Standard deviation of Gaussian kernel

    Returns:
    --------
    ndarray : Filtered image
    """
    return gaussian_filter(image, sigma=sigma)


def apply_median_filter(image, size=3):
    """
    Apply median filter for noise reduction.

    Parameters:
    -----------
    image : ndarray
        Input image
    size : int
        Size of the median filter kernel

    Returns:
    --------
    ndarray : Filtered image
    """
    return median_filter(image, size=size)


def apply_bandpass_filter(image, low_freq=0.05, high_freq=0.3):
    """
    Apply band-pass filter in frequency domain.

    Parameters:
    -----------
    image : ndarray
        Input image
    low_freq : float
        Lower cutoff frequency
    high_freq : float
        Upper cutoff frequency

    Returns:
    --------
    ndarray : Filtered image
    """
    rows, cols = image.shape

    # Create frequency grid
    u = np.fft.fftfreq(rows)
    v = np.fft.fftfreq(cols)
    U, V = np.meshgrid(u, v, indexing='ij')
    D = np.sqrt(U**2 + V**2)

    # Band-pass filter
    H = ((D >= low_freq) & (D <= high_freq)).astype(float)

    # Apply filter
    F = np.fft.fft2(image)
    F_filtered = F * H
    filtered = np.real(np.fft.ifft2(F_filtered))

    return filtered


def line_by_line_leveling(image, order=1):
    """
    Apply line-by-line leveling (common in STM).
    Removes polynomial trends from each scan line.

    Parameters:
    -----------
    image : ndarray
        Input image
    order : int
        Polynomial order (1=linear, 2=quadratic, etc.)

    Returns:
    --------
    ndarray : Leveled image
    """
    leveled = np.copy(image)
    rows, cols = image.shape

    for i in range(rows):
        line = image[i, :]
        x = np.arange(cols)

        # Fit polynomial
        coeffs = np.polyfit(x, line, order)
        poly = np.polyval(coeffs, x)

        # Subtract polynomial
        leveled[i, :] = line - poly + np.mean(line)

    return leveled


def polynomial_background_subtraction(image, order=2):
    """
    Subtract polynomial background from entire image.

    Parameters:
    -----------
    image : ndarray
        Input image
    order : int
        Polynomial order

    Returns:
    --------
    ndarray : Background-subtracted image
    """
    rows, cols = image.shape

    # Create coordinate grids
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)

    # Flatten for polynomial fitting
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = image.flatten()

    # Build polynomial terms
    terms = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            terms.append((X_flat**i) * (Y_flat**j))

    A = np.column_stack(terms)

    # Fit polynomial
    coeffs, _, _, _ = np.linalg.lstsq(A, Z_flat, rcond=None)

    # Compute background
    background = A @ coeffs
    background = background.reshape(rows, cols)

    # Subtract background
    result = image - background + np.mean(image)

    return result


def drift_correction(image, reference_row=0):
    """
    Simple drift correction by aligning each row to a reference.

    Parameters:
    -----------
    image : ndarray
        Input image
    reference_row : int
        Row index to use as reference

    Returns:
    --------
    ndarray : Drift-corrected image
    """
    corrected = np.copy(image)
    rows = image.shape[0]

    ref_line = image[reference_row, :]

    for i in range(rows):
        if i == reference_row:
            continue

        # Cross-correlation to find shift
        correlation = signal.correlate(ref_line, image[i, :], mode='same')
        shift = np.argmax(correlation) - len(ref_line) // 2

        # Apply shift
        corrected[i, :] = np.roll(image[i, :], shift)

    return corrected


def despeckle(image, threshold=3.0):
    """
    Remove outliers (hot/cold pixels) using median-based detection.

    Parameters:
    -----------
    image : ndarray
        Input image
    threshold : float
        Number of standard deviations for outlier detection

    Returns:
    --------
    ndarray : Despeckled image
    """
    # Apply median filter
    med = median_filter(image, size=3)

    # Find outliers
    diff = np.abs(image - med)
    std = np.std(diff)
    outliers = diff > threshold * std

    # Replace outliers with median values
    result = np.copy(image)
    result[outliers] = med[outliers]

    return result


def enhance_contrast_clahe(image, clip_limit=0.03, tile_size=8):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Parameters:
    -----------
    image : ndarray
        Input image
    clip_limit : float
        Clipping limit for contrast
    tile_size : int
        Size of tiles for local histogram equalization

    Returns:
    --------
    ndarray : Contrast-enhanced image
    """
    # Normalize to [0, 1]
    img_norm = (image - image.min()) / (image.max() - image.min())

    # Apply CLAHE
    enhanced = exposure.equalize_adapthist(
        img_norm,
        kernel_size=(tile_size, tile_size),
        clip_limit=clip_limit
    )

    # Denormalize
    enhanced = enhanced * (image.max() - image.min()) + image.min()

    return enhanced


def gradient_filter(image, direction='both'):
    """
    Apply gradient filter for edge detection.

    Parameters:
    -----------
    image : ndarray
        Input image
    direction : str
        'x', 'y', or 'both' (magnitude)

    Returns:
    --------
    ndarray : Gradient-filtered image
    """
    # Sobel filters
    grad_x = ndimage.sobel(image, axis=1)
    grad_y = ndimage.sobel(image, axis=0)

    if direction == 'x':
        return grad_x
    elif direction == 'y':
        return grad_y
    else:  # 'both'
        return np.sqrt(grad_x**2 + grad_y**2)


def crop_image(image, x_start, x_end, y_start, y_end):
    """
    Crop image to specified region.

    Parameters:
    -----------
    image : ndarray
        Input image
    x_start, x_end : int
        X-axis crop boundaries
    y_start, y_end : int
        Y-axis crop boundaries

    Returns:
    --------
    ndarray : Cropped image
    """
    return image[y_start:y_end, x_start:x_end]


def rotate_image(image, angle, reshape=True):
    """
    Rotate image by specified angle.

    Parameters:
    -----------
    image : ndarray
        Input image
    angle : float
        Rotation angle in degrees (counter-clockwise)
    reshape : bool
        If True, expand output to fit entire rotated image

    Returns:
    --------
    ndarray : Rotated image
    """
    return ndimage.rotate(image, angle, reshape=reshape, mode='constant', cval=np.nan)


def flip_image(image, direction='horizontal'):
    """
    Flip image horizontally or vertically.

    Parameters:
    -----------
    image : ndarray
        Input image
    direction : str
        'horizontal' or 'vertical'

    Returns:
    --------
    ndarray : Flipped image
    """
    if direction == 'horizontal':
        return np.fliplr(image)
    elif direction == 'vertical':
        return np.flipud(image)
    else:
        raise ValueError("Direction must be 'horizontal' or 'vertical'")


# ============================================================================
# FFT-Specific Processing Functions
# ============================================================================

def apply_window_fft(image, window_type='hanning'):
    """
    Apply windowed FFT to reduce edge artifacts.

    Parameters:
    -----------
    image : ndarray
        Input real-space image
    window_type : str
        Type of window: 'hanning', 'hamming', 'blackman', 'bartlett'

    Returns:
    --------
    ndarray : FFT with window applied (complex array, fftshifted)
    """
    rows, cols = image.shape

    # Create 2D window
    if window_type == 'hanning':
        win_row = np.hanning(rows)
        win_col = np.hanning(cols)
    elif window_type == 'hamming':
        win_row = np.hamming(rows)
        win_col = np.hamming(cols)
    elif window_type == 'blackman':
        win_row = np.blackman(rows)
        win_col = np.blackman(cols)
    elif window_type == 'bartlett':
        win_row = np.bartlett(rows)
        win_col = np.bartlett(cols)
    else:
        win_row = np.ones(rows)
        win_col = np.ones(cols)

    window_2d = np.outer(win_row, win_col)

    # Subtract mean before windowing (following PyAtoms approach)
    image_centered = image - np.mean(image)
    # Apply window and compute FFT
    windowed_image = image_centered * window_2d
    fft_result = np.fft.fftshift(np.fft.fft2(windowed_image))

    return fft_result


def apply_gaussian_mask_fft(fft_data, center_x, center_y, sigma, inverse=False):
    """
    Apply Gaussian mask to FFT around specified center.

    Parameters:
    -----------
    fft_data : ndarray
        FFT data (complex array, should be fftshifted)
    center_x, center_y : int
        Center coordinates for the mask
    sigma : float
        Standard deviation of Gaussian (in pixels)
    inverse : bool
        If True, inverts the mask (keeps outside, removes inside)

    Returns:
    --------
    ndarray : Masked FFT data
    """
    rows, cols = fft_data.shape
    y, x = np.ogrid[:rows, :cols]

    # Create Gaussian mask centered at (center_x, center_y)
    mask = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))

    if inverse:
        mask = 1 - mask

    return fft_data * mask


def apply_lowpass_filter_fft(fft_data, center_x, center_y, radius):
    """
    Apply low-pass filter to FFT around specified center.

    Parameters:
    -----------
    fft_data : ndarray
        FFT data (complex array, should be fftshifted)
    center_x, center_y : int
        Center coordinates for the filter
    radius : float
        Cutoff radius in pixels

    Returns:
    --------
    ndarray : Filtered FFT data
    """
    rows, cols = fft_data.shape
    y, x = np.ogrid[:rows, :cols]

    # Create circular mask
    mask = ((x - center_x)**2 + (y - center_y)**2) <= radius**2

    return fft_data * mask


def apply_highpass_filter_fft(fft_data, center_x, center_y, radius):
    """
    Apply high-pass filter to FFT around specified center.

    Parameters:
    -----------
    fft_data : ndarray
        FFT data (complex array, should be fftshifted)
    center_x, center_y : int
        Center coordinates for the filter
    radius : float
        Cutoff radius in pixels

    Returns:
    --------
    ndarray : Filtered FFT data
    """
    rows, cols = fft_data.shape
    y, x = np.ogrid[:rows, :cols]

    # Create circular mask (inverted)
    mask = ((x - center_x)**2 + (y - center_y)**2) > radius**2

    return fft_data * mask


def change_fft_origin(fft_data, new_origin_x, new_origin_y):
    """
    Change the origin of FFT to a new point.

    Parameters:
    -----------
    fft_data : ndarray
        FFT data (complex array, should be fftshifted)
    new_origin_x, new_origin_y : int
        New origin coordinates

    Returns:
    --------
    ndarray : FFT with shifted origin
    """
    rows, cols = fft_data.shape
    center_x, center_y = cols // 2, rows // 2

    # Calculate shift amounts
    shift_x = new_origin_x - center_x
    shift_y = new_origin_y - center_y

    # Shift the FFT data
    shifted = np.roll(fft_data, (shift_y, shift_x), axis=(0, 1))

    return shifted


# ============================================================================
# GPA (Geometric Phase Analysis) Functions
# ============================================================================

def refine_peak_center(fft_magnitude, x_guess, y_guess, window_size=10):
    """
    Refine peak center using center of mass calculation.

    Parameters:
    -----------
    fft_magnitude : ndarray
        FFT magnitude image
    x_guess, y_guess : int
        Initial guess for peak position
    window_size : int
        Window size for center of mass calculation

    Returns:
    --------
    tuple : (refined_x, refined_y)
    """
    from scipy.ndimage import center_of_mass

    rows, cols = fft_magnitude.shape

    # Define window bounds
    x_min = max(0, int(x_guess - window_size))
    x_max = min(cols, int(x_guess + window_size))
    y_min = max(0, int(y_guess - window_size))
    y_max = min(rows, int(y_guess + window_size))

    # Extract window
    window = fft_magnitude[y_min:y_max, x_min:x_max]

    # Calculate center of mass
    cy, cx = center_of_mass(window)

    # Convert back to full image coordinates
    refined_x = x_min + cx
    refined_y = y_min + cy

    return refined_x, refined_y


def _g_cycles_per_pixel(g_pos, rows, cols):
    """FFT peak position (x, y) in pixels of the fftshifted FFT -> g in cycles/pixel."""
    return np.array([(g_pos[0] - cols // 2) / cols, (g_pos[1] - rows // 2) / rows])


def standard_gpa(image, g_vectors, mask_radius, sigma=5.0, edge_blur=5):
    """
    Standard GPA implementation.

    Parameters:
    -----------
    image : ndarray
        Real-space image
    g_vectors : list of tuples
        List of (gx, gy) g-vector positions in FFT space
    mask_radius : float
        Radius of Gaussian mask around each g-vector
    sigma : float
        Gaussian filter sigma for phase smoothing
    edge_blur : int
        Edge blurring parameter

    Returns:
    --------
    tuple : (exx, eyy, exy, omega, phases) strain and rotation maps, plus list of unwrapped phases
    """
    from skimage.restoration import unwrap_phase

    rows, cols = image.shape

    # Subtract mean before FFT
    fft_image = np.fft.fftshift(np.fft.fft2(image - np.mean(image)))

    # Initialize phase arrays
    phases = []

    Y, X = np.mgrid[:rows, :cols]

    for gx, gy in g_vectors:
        # Create Gaussian mask around g-vector
        mask = np.exp(-((X - gx)**2 + (Y - gy)**2) / (2 * mask_radius**2))

        # Apply mask in frequency domain
        masked_fft = fft_image * mask

        # Inverse FFT to get complex image H_g(r) = A exp(i(2*pi*g.r + P_g(r)))
        complex_image = np.fft.ifft2(np.fft.ifftshift(masked_fft))

        # Geometric phase P_g: remove the carrier 2*pi*g.r of the reference
        # lattice (g in cycles/pixel), otherwise the phase is dominated by
        # the lattice fringes themselves instead of their distortion.
        g = _g_cycles_per_pixel((gx, gy), rows, cols)
        carrier = 2 * np.pi * (g[0] * X + g[1] * Y)
        phase = np.angle(complex_image * np.exp(-1j * carrier))
        phase_unwrapped = unwrap_phase(phase)

        # Smooth phase
        if sigma > 0:
            phase_unwrapped = gaussian_filter(phase_unwrapped, sigma=sigma)

        phases.append(phase_unwrapped)

    # Calculate strain components from phase gradients
    if len(phases) >= 2:
        # Reference g-vectors in cycles/pixel, relative to the FFT centre
        G = np.array([_g_cycles_per_pixel(g_vectors[0], rows, cols),
                      _g_cycles_per_pixel(g_vectors[1], rows, cols)])
        G_inv = np.linalg.inv(G)

        # P_g = -2*pi * g.u  =>  u = -G^-1 [P_g1, P_g2] / (2*pi)
        phi1 = phases[0] - np.mean(phases[0])
        phi2 = phases[1] - np.mean(phases[1])
        u = -np.einsum("ij,jrc->irc", G_inv, np.array([phi1, phi2])) / (2 * np.pi)
        ux, uy = u[0], u[1]

        # Calculate strain tensor from displacement gradients
        dux_dx = np.gradient(ux, axis=1)
        dux_dy = np.gradient(ux, axis=0)
        duy_dx = np.gradient(uy, axis=1)
        duy_dy = np.gradient(uy, axis=0)

        # Strain components
        exx = dux_dx
        eyy = duy_dy
        exy = 0.5 * (dux_dy + duy_dx)  # Shear strain
        omega = 0.5 * (duy_dx - dux_dy)  # Rotation

        return exx, eyy, exy, omega, phases
    else:
        # Not enough g-vectors
        return (np.zeros_like(image), np.zeros_like(image),
                np.zeros_like(image), np.zeros_like(image), [])


def adaptive_gpa(image, g_vectors, mask_radius, window_size=32, step=16, sigma=5.0):
    """
    Adaptive GPA implementation with local windowing.

    Parameters:
    -----------
    image : ndarray
        Real-space image
    g_vectors : list of tuples
        List of (gx, gy) g-vector positions
    mask_radius : float
        Radius of Gaussian mask
    window_size : int
        Size of local window for adaptive processing
    step : int
        Step size for window sliding
    sigma : float
        Smoothing parameter

    Returns:
    --------
    tuple : (exx, eyy, exy, omega, phases) strain and rotation maps, plus phase data
    """
    # For now, use standard GPA - adaptive implementation would require
    # more complex windowing logic
    return standard_gpa(image, g_vectors, mask_radius, sigma=sigma)

