from ipywidgets import interactive
import matplotlib.pyplot as plt
import matplotlib as mpl
import pylab as pl
from matplotlib.gridspec import GridSpec

from numpy import fft
import numpy as np
import math as m

# Shortcuts:
from numpy import fft as npf
from numpy import cos as cos
from numpy import sin as sin



def hexatoms(pix, L, a, theta, honeycomb, origin_x=0, origin_y=0):
    """
    Simulates hexagonal lattice, returns the 2D image array Z
        
    Creates a (pix x pix) image of side length L (nm) array of hexagonal atoms with
    periodicity a (nm) rotated from the x-direction by an angle theta. 
    If honeycomb == 1, it creates the same image but with honeycomb pattern. 
    For either choice, the image height is normalized to [0,1].

    
    Args:
        pix [pixels]: number of pixels 
        L [nm]: side length of the sample 
        a [nm]: periodicity, lattice parameter
        theta [degrees]: angle to rotate lattice by theta degrees from the x-direction
        honeycomb [0 or 1]: Creates the lattice with/without honeycomb pattern
                            0: no honeycomb pattern. 1: honeycomb pattern
        origin_x, origin_y [pixels]: sets the origin of the lattice. Default is set to 0,0
    """
    
    colormap='plasma'
    
    # Reciprocal lattice vectors for hexagonal crystal
    # For hex lattices, b = 2*pi/a_vect , where a_vect is the direct lattice vectors in real space
    # For hex lattices: a1_vect = a[1/2, sqrt(3)/2], a2_vect = a[-1/2, sqrt(3)/2]
    k1 = np.array([0, 1])*(4*np.pi/(m.sqrt(3)*a))
    k2 = np.array([-m.sqrt(3)/2, -1/2])*(4*np.pi/(m.sqrt(3)*a))
    k3 = -(k1 + k2) # Get the 3rd wavevector by taking a linear combo of k1,k2 
    ## Sidenote: It's usually better to find the pixel positions for v3, but this should also work
    
    
    # If theta is NOT 0 --> rotate the lattice
    if theta != 0:
        # Rotate the lattice by theta (assuming theta is in degrees)
        theta_rad = np.deg2rad(theta) # convert theta to radians

        # Create rotation matrix to multiply reciprocal lattice vectors to rotate the image 
        # Rotation matrix =  [[cos, -sin] 
        #                     [sin, cos]]
        rotmat = np.array([[cos(theta_rad), -sin(theta_rad)], 
                           [sin(theta_rad), cos(theta_rad)]])

        # Rotate wavevectors by angle theta, using rotation matrix 
        k1 = np.matmul(k1,rotmat)
        k2 = np.matmul(k2,rotmat)
        k3 = np.matmul(k3, rotmat)
        
    
    # Create 2D meshgrid of points
    x = np.linspace(0, L, pix) # Creates an array of 'pix' number of values, ranging from [0 --> L]
    [X,Y] = np.meshgrid(x,x) 
    ## ^^ IMPORTANT: make sure its typed [X,Y] instead of [Y,X], 
    ##    otherwise you might need to transpose the output
    

    # Set the center of the image based on the origin pixel positions
    origin = [origin_x, origin_y]
    
    # If origin is [0,0], set the center of the image to just be the center pixel
    if origin == [0,0]: 
        ctrX = x[int(np.floor(pix/2))]
        ctrY = ctrX
    # If the origin is defined otherwise, set the center of the image to be ...also the center pixel? 
    else:
        ctrX = pix/2
        ctrY = pix/2
        
    
    # Create the image with or without a honeycomb pattern
    # Define the 'height' of the topography to be either +1 or -1 depending on the value of honeycomb
    if honeycomb == 1: # Creates the image with a honeycomb pattern
        A = -1
    else:              # No honeycomb pattern (equivalent to plotting dots as atoms)
        A = 1
        
    
    # Calculate the distance from a point on the meshgrid to the center of image
    # Distance from point on meshgrid to center of x/y axes
    x_i = X - ctrX    # these are also meshgrids
    y_i = Y - ctrY
    

    # Create variables for each wavevector component - for convenience only 
    # (just to make the code for Z below look less cluttered)
    k1x, k1y = k1[0], k1[1]
    k2x, k2y = k2[0], k2[1]
    k3x, k3y = k3[0], k3[1]
    
    #################################
    ##### CREATE THE IMAGE  #########
    #################################
    
    # Remember that electron wavefunctions are complex exponentials
    # Taking the real part only we get a cosine because e^ikx = cos(kx) + isin(kx)
    # Multiplying by A (which is either +1 or -1) normalizes the image height to be within [0,1]
    # Multiply by 2 to account for both +k and -k frequencies
    Z = 2 * A * (cos((k1x * x_i) + (k1y * y_i)) 
               + cos((k2x * x_i) + (k2y * y_i)) 
               + cos((k3x * x_i) + (k3y * y_i))) 

    return Z # Return the image array 

def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    #plt.sca(last_axes)
    return cbar

def CDWampmap2D(Z,r0,v1,v2,sigma,colormap):
    """
    Spatial lockin technique to extract the local amplitude of charge density wave (CDW) with
    independent wavevectors v1,v2 in the raw data Z and spatial-avg constant sigma (Gaussian half-width), 
    with ~0.6*sigma_CDW being the min value.
    
    Plots the raw data Z alongside the extracted amplitudes for the v1, v2 wavevectors with the specified colormap
    
    Args:
        Z [array]: 2D image array of the atoms. raw image data
        r0 [pixels]: vector of the pixel positions of a maxima in the real space image
        v1 [pixels]: vector of the pixel positions of a peak in the FFT of Z
        v2 [pixels]: vector of the pixel positions of another peak in the FFT of Z
                     Note: v1, v2 are linearly independent wavevectors
        sigma [pixels?]: gaussian half-width. Radius of the gaussian mask used to filter the complex images;
                         Ideally, sigma is approximately half of the pixel distance 
                         between 2 neighboring atoms in the real space image
        colormap [str]: name of the matplotlib colormap used to plot the output 
    """
    
    

    # Get the number of pixels/samples of the image
    pix = len(Z[0])
    
    # Calculate the center pixel depending on if N is even or odd
    # (assuming Z is a square NxN image array)
    if len(Z)%2 == 0:
        ctr_pt = np.floor(pix/2) + 1
    else:
        ctr_pt = np.floor(pix/2)
    
    ctr = np.array([ctr_pt,ctr_pt])
    
    # Rename the reference atom pixel positions x0,y0
    x0,y0 = r0[0],r0[1]
    
    # Calculate the reciprocal lattice wavevectors, normalizing by the number of pixels
    q1 = (v1 - ctr)*2*np.pi / pix
    q2 = (v2 - ctr)*2*np.pi / pix
    
    # Take the FFT of Z
    fftZ = npf.fftshift(npf.fft2(Z - np.mean(np.mean(Z)))) # always do fftshift after fft 
    # subtract by the mean of the mean of Z - we are removing the DC/constant background at k = 0
    # (mean(mean(Z)) is a scalar value). This will remove the peak at k = 0
    
    
    # Create meshgrid of points
    x = np.linspace(0,pix,pix)
    [X,Y] = np.meshgrid(x,x) ## MAKE SURE THIS IS DEFINED CORRECTLY!!!!!!! ie [X,Y] not [Y,X]
    # If you define this as [Y,X] you might need to transpose the final output
    
    
    # FOURIER AMP MAPPING
    # Multiply the image by a complex exponential, oscillating with the same frequency corresponding to the v1 peak
    Z1 = Z * np.exp(-1j*((q1[0]*(X-x0) + q1[1]*(Y-y0)))) # the (x-x0) part makes the maximum of this oscillating function line up with the maximum (at x0) in the real image?
    Z2 = Z * np.exp(-1j*((q2[0]*(X-x0) + q2[1]*(Y-y0)))) # to see this ^^^, need to plot the real part of Z1,Z2 (NOT absolute value) 
    
    
    # Take the FFT of Z1 --> The peaks should be shifted, such that the v1 peak is now at the origin in k-space
    fftZ1 = npf.fftshift(npf.fft2(Z1))
    fftZ2 = npf.fftshift(npf.fft2(Z2))  # These peaks are shifted by v2 --> v2 peak at the center
    

    # GAUSSIAN LOW-PASS FILTERING
    # Create a Gaussian mask, of radius sigma 
    # The mask must have size (pix x pix), same as the image shape
    mask = np.zeros(Z1.shape) # Initialize a mask of zeros with the same shape as the image
    mask += np.exp(-(((X-ctr_pt)**2 + (Y-ctr_pt)**2)/(2*sigma**2))) 

    
    # Filter the image by convoluting the mask with the FFT of  Z1
    # Since the v1 peak gets shifted to the origin of fftZ1, 
    # and since the gaussian mask is centered at the origin in k-space (aka, the center pixel of the imaage) 
    # then, applying the mask on fftZ1 will fitler out all peaks except the one at the origin, that corresponds to the v1 freq
    fftZ1filt = mask * fftZ1 
    fftZ2filt = mask * fftZ2 
   
    # AMPLITUDE MAPS
    # Get back the filtered image (aka the amplitude maps) by taking the inverse FFT of the filtered fft
    Z1_filt = npf.ifft2(npf.ifftshift(fftZ1filt))
    Z2_filt = npf.ifft2(npf.ifftshift(fftZ2filt))

    
    Ztot = np.abs(Z1_filt) + np.abs(Z2_filt) 
    
    #print('sigma = ', sigma, '\nv1 = ', v1, '\nv2 = ', v2, '\nr0 = ', r0)
    
    
    return (Z1_filt)
