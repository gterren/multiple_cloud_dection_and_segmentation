import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.interpolate import interp1d
from scipy.stats import norm, multivariate_normal, mode, skew, kurtosis
from scipy.interpolate import griddata
from scipy.ndimage.filters import median_filter, gaussian_filter
from scipy.ndimage.morphology import binary_fill_holes
from skimage.transform import hough_circle, hough_circle_peaks

#from skimage.measure import label
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing

from utils import _save_file

# Normalize an infrared image of 16 bits to 8 bits
# https://www2.acom.ucar.edu/news/cloud-tops-and-tropopause
# There are 11.5km of atmosphere at 36 latitude north
# Albuquerque is at 1.6km above the sea level
# Temperature Decreases by 9.8 degrees each km
# Therefore the Maximum intesity of a cloud should be:
#           9.8 x (11.5 - 1.6) = aprox. 97 x 100 to intensity = aprox. 9700
def _normalize_infrared_image(I_, i_max = 9700.):
    return 255*(I_ - I_.min())/i_max

def _regularize_infrared_image(I_, i_max = 9700.):
    I_prime_ = 255*I_/i_max
    I_prime_[ I_prime_ < 0.] = 0.
    return I_prime_

def _normalization(I_, H_, tau, w = .0105, tau_n = 26., tau_h = 9000., verbose = True):
    I_norm_ = _normalize_infrared_image(I_)
    H_total_ = (H_ > tau_h).sum()
    I_total_ = (I_norm_ < tau_n).sum()
    diff = I_total_ - H_total_
    bias = np.absolute(w*diff)
    if verbose:
        print('>> No. threshold Normalized Intensity = {} No. threshold Height = {} Difference = {} offset = {}'.format(I_total_, H_total_, diff, bias))
    # If there is not offset return normalized image
    if np.absolute(diff) < tau:
        return I_norm_, I_norm_
    # Otherwise
    else:
        # Add offset
        I_prime_ = I_norm_ + bias
        I_prime_[ I_prime_ < 0.] = 0.
        return I_norm_, I_prime_
# Label Clouds by Connected Components labelling
def _cloud_pixels_labeling(I_): return label(I_, background = 0)

# Mathematical Dialatation for filling binary holes
def _fill_holes(I_): return binary_fill_holes(I_)

# Generating grid values and set them with reference to the Sun
# to denote their distance to the center of the image
def _polar_coordinates_transformation(x_sun_, X, Y):
    X_prime_ = X - x_sun_[0]
    Y_prime_ = Y - x_sun_[1]
    theta_   = np.sqrt( (X_prime_)**2 + (Y_prime_)**2 ) + 1e-25
    alpha_   = np.arccos(X_prime_/theta_)
    alpha_[:30, :] = np.pi - alpha_[:30, :] + np.pi
    alpha_ = np.fliplr(np.flipud(alpha_))
    return theta_, alpha_

# Interpolate the pixels on the circumsolar area for smoothing atmospheric effecs removal
def _sun_pixels_interpolation(I_, MA_, X_, Y_, N_y, N_x, radius):
    # Index of the circumsolar pixel
    idx_ = MA_[0] >= radius
    # Defining grid of pixels to interpolate by using the pixels selected and do interpolation
    xy = np.concatenate((X_[idx_][:, np.newaxis], Y_[idx_][:, np.newaxis]), axis = 1)
    I_prime_ = griddata(xy, I_[idx_], (X_.flatten(), Y_.flatten()), method = 'linear').reshape([N_y, N_x])
    # Removing remaining deformities: positives and negatives by appying a median filter
    #I_prime_[~idx_] = median_filter(I_prime_, size = 5, mode = 'reflect')[~idx_]
    I_prime_[~idx_] = gaussian_filter(I_prime_, sigma = .5, mode = 'reflect')[~idx_]
    return I_prime_

# Generate the atmospheric effect from the Sun frame position coordiantes and horizon angles
def _atmospheric_effect_v11(_tools, I_, x_sun_, m_, i_sun_max, tau):
    # Obtaining coefficients for the athmospheric models from a polynomial model
    def __polynomial_model(x_, w_, degree):
        x_poly_ = PolynomialFeatures(degree).fit_transform(x_)
        return np.squeeze( x_poly_ @ w_)
    # Applying atmospheric models to estimate effects of scatter radiation on the images
    def __F(I_, w_):
        # What is the highest Temperature in the IR image?
        i_max = I_.max()
        # Scatter radiation Model
        f_1_ = w_[0] * np.exp( (_tools.Y_ - x_sun_[1]) / w_[1])
        # Direct Radiation Model
        f_2_ = w_[2] * ( (w_[3]**2) / ( (_tools.X_ - x_sun_[0])**2 + (_tools.Y_- x_sun_[1])**2 + w_[3]**2 )**1.5 )
        # Regularization of the Maximum Direct Radiation in the Image
        f_2_[f_2_ > i_max] = i_max
        # Combine Both Models
        f_ = f_1_ + f_2_
        # Regularization of the Maximum Direct Radiation in the Image
        f_[f_ > i_max] = i_max

        fig = plt.figure(figsize = (7.5, 5))
        ax = fig.gca(projection = '3d')
        #ax.set_title(r'$\mathcal{S} (y_{i,j}; y_{0}, \theta_1, \theta_2) $', fontsize = 25)
        ax.set_title(r'Atmosphere Irradiance Model [K]',
        fontsize = 25)
        ax.plot_surface(_tools.X_, _tools.Y_, f_1_/100., linewidth = 1., antialiased = True,
        cmap = 'inferno')
        ax.set_xlabel(r'x-axis', fontsize = 15, labelpad = 10)
        ax.set_ylabel(r'y-axis', fontsize = 15, labelpad = 10)
        #ax.set_zlabel(r'Temp. $[^\circ K]$', fontsize = 15, labelpad = 10)
        ax.set_yticks(np.arange(0, 60, 15))
        ax.set_xticks(np.arange(0, 80, 15))
        ax.tick_params(labelsize = 15)
        plt.savefig('{}'.format(r'atmosphere_irradiance_model'), bbox_inches = 'tight', dpi = 500)
        plt.show()

        fig = plt.figure(figsize = (7.5, 5))
        ax = fig.gca(projection = '3d')
        #ax.set_title(r'$\mathcal{D} (\mathbf{x}_{i,j}; \mathbf{x}_{0}, \theta_3, \theta_4) $',
        #fontsize = 25)
        ax.set_title(r'Direct Irradiance Model [K]',
        fontsize = 25)
        ax.plot_surface(_tools.X_, _tools.Y_, f_2_/100., linewidth = 1., antialiased = True,
        cmap = 'inferno')
        ax.set_xlabel(r'x-axis', fontsize = 15, labelpad = 10)
        ax.set_ylabel(r'y-xis', fontsize = 15, labelpad = 10)
        #ax.set_zlabel(r'Temp. $[^\circ K]$', fontsize = 15, labelpad = 10)
        ax.set_yticks(np.arange(0, 60, 15))
        ax.set_xticks(np.arange(0, 80, 15))
        ax.tick_params(labelsize = 15)
        plt.savefig('{}'.format(r'direct_irradiance_model'), bbox_inches = 'tight', dpi = 500)
        plt.show()

        fig = plt.figure(figsize = (7.5, 5))
        ax = fig.gca(projection = '3d')
        #ax.set_title(r'$\mathcal{D} (\mathbf{x}_{i,j}; \mathbf{x}_{0}, \theta_3, \theta_4) $',
        #fontsize = 25)
        ax.set_title(r'Background Irradiance Model [K]',
        fontsize = 25)
        ax.plot_surface(_tools.X_, _tools.Y_, f_/100., linewidth = 1., antialiased = True,
        cmap = 'inferno')
        ax.set_xlabel(r'x-axis', fontsize = 15, labelpad = 10)
        ax.set_ylabel(r'y-xis', fontsize = 15, labelpad = 10)
        #ax.set_zlabel(r'Temp. $[^\circ K]$', fontsize = 15, labelpad = 10)
        ax.set_yticks(np.arange(0, 60, 15))
        ax.set_xticks(np.arange(0, 80, 15))
        ax.tick_params(labelsize = 15)
        plt.savefig('{}'.format(r'background_irradiance_model'), bbox_inches = 'tight', dpi = 500)
        plt.show()

        # When there is direct radiation apply direct radiation model and scatter radiation model
        if i_max > tau:
            return f_
        # When there is not direct radiation apply only the scatter model
        else:
            return f_1_
    # Get Scatter Radiation Parameters
    w_0 = __polynomial_model(_tools.x_0_, w_ = m_[0][0][0], degree = m_[0][0][1])
    w_1 = __polynomial_model(_tools.x_1_, w_ = m_[0][1][0], degree = m_[0][1][1])
    # Regularization of the Atmosphere Backgroud Radiation inclination
    if w_1 < 550: w_1 = 550.
    # Get Direct Radiation Parameters
    w_2 = m_[1][0]
    w_3 = m_[1][1]
    # Atmospheric models parameters
    return __F(I_, w_ = [w_0, w_1, w_2, w_3]), w_0


# Finding mass center for each segmented object on a frame
# https://stackoverflow.com/questions/37519238/python-find-center-of-object-in-an-image
def _clouds_mass_center(I_seg_, N_x, N_y):
    # Initialize positions matrix
    M_ = np.zeros((N_y, N_x), dtype = bool)
    # loop for each segmented object on an image
    for l in np.unique(I_seg_):
        # Selecting only the object to me analized
        I_bin_ = np.zeros((N_y, N_x), dtype = bool)
        I_bin_[I_seg_ == l] = True
        # Calculating object mass
        mass = I_bin_ / np.sum(I_bin_)
        # Finding the center of mass
        c_x = int( np.around(np.sum(np.sum(mass, 0) * np.arange(N_x))) )
        c_y = int( np.around(np.sum(np.sum(mass, 1) * np.arange(N_y))) )
        M_[c_y, c_x] = True
    return M_

def _perspective_transformation_v0(X_, Y_, N_x, N_y, x_sun_, A_sun_, height, altitude = 1630, FOV = 63.75, focal_length = 1.3669e-3, pixel_size = 17e-6):
    # Transform the Coordinates Plane to the Camera Plane
    def __camera_plane_coordinates(X_, Y_, dist, const):
        return X_ * dist * const, Y_ * dist * const
    # Project the Cross-Section plane in the Camera pLane
    def __cross_section_plane_coordinates(X_p_, Y_p_, X_pp_, Y_pp_, N_x):
        return X_p_ + np.cumsum(X_pp_/N_x, axis = 1), Y_pp_
    # Calculate the Cross-Section Coodinates System
    def __cross_seccion_coordinates(epsilon, dist, N_x, N_y, radians_per_pixel):
        # Transformation of the y-axis
        def ___y_axis(alpha_y_, height, epsilon):
            angle_upper_ = epsilon + alpha_y_
            angle_lower_ = epsilon - alpha_y_
            idx_ = angle_upper_ > np.pi/2
            y_ = height/np.tan(epsilon)
            y_upper = y_ - height/np.tan(angle_upper_)
            y_lower = height/np.tan(angle_lower_) - y_
            y_p_ = np.concatenate((-np.flip(y_upper), y_lower), axis = 0)
            y_p_+= np.absolute(y_p_[0])
            return y_p_
        # Transformation of the x-axis
        def ___x_axis(y_, alpha_y_, epsilon):
            return ( np.tan(alpha_y_[-1])*y_ )/( np.cos(alpha_y_[-1])*(np.tan(epsilon)*(1./np.tan(alpha_y_[-1])) + 1) )
        # Extend the FOV for each pixels along the y-axis
        alpha_y_ = np.linspace(radians_per_pixel/2, radians_per_pixel*N_y/2, N_y//2)
        # Get the Coordiantes transformation in the 4 Cuadrantes of the Coordinates system
        y_ = ___y_axis(alpha_y_, height, epsilon)
        x_ = ___x_axis(y_, alpha_y_, epsilon)
        # Repeat the vectors along the x-axis
        return np.tile(x_, (N_x, 1)).T, np.tile(y_, (N_x, 1)).T

    def __prespective_limits(X_ppp_, Y_ppp_, FOV_x, FOV_y, height, epsilon_max = 79):
        dist_x = 2*height*np.tan( FOV_x/2 )
        dist_y = 2*height*np.tan( FOV_y/2 )
        idx_x_ = np.absolute(X_ppp_) <= dist_x
        idx_y_ = np.absolute(Y_ppp_) <= dist_y
        return idx_x_ & idx_y_

    def __interpolation_grid(X_ppp_, Y_ppp_, index_, N_x, N_y):
        xx_= []
        Y_ = (N_y - 1)*( Y_ppp_[index_] - np.min(Y_ppp_[index_]))/(np.max(Y_ppp_[index_]) - np.min(Y_ppp_[index_]))
        for y in np.unique(Y_ppp_[index_]):
            idx_ = (Y_ppp_ == y) & index_
            x_ = (N_x - 1)*( X_ppp_[idx_] - np.min(X_ppp_[idx_]))/(np.max(X_ppp_[idx_]) - np.min(X_ppp_[idx_]))
            xx_.append(x_)
        X_ = np.concatenate(xx_, axis = 0)
        return [np.concatenate((X_[:, np.newaxis],  Y_[:, np.newaxis]), axis = 1), index_]

    epsilon = A_sun_[0, 0]
    #print(epsilon)
    # Camera Specifications
    angles_per_pixel  = FOV/np.sqrt(N_x**2 + N_y**2)
    radians_per_pixel = np.radians(angles_per_pixel)
    FOV_x = radians_per_pixel * N_x
    FOV_y = radians_per_pixel * N_y
    #print(epsilon, angles_per_pixel, radians_per_pixel, FOV_x, FOV_y)
    # Camera Constants
    const = pixel_size/focal_length
    # Sky-Parcel Geometric Constants
    height   = height #- altitude
    epsilon  = np.radians(epsilon)
    dist     = height/np.sin(epsilon)
    # Camera Plane Coordinates
    X_p_, Y_p_ = __camera_plane_coordinates(X_, Y_, dist, const)
    # Cross-Section Plane Coordinates
    X_pp_, Y_pp_ = __cross_seccion_coordinates(epsilon, dist, N_x, N_y, radians_per_pixel)
    # Cross-Section Coordinates Projected in the Plane
    X_ppp_, Y_ppp_ = __cross_section_plane_coordinates(X_p_, Y_p_, X_pp_, Y_pp_, N_x)
    # Calculate Increments per pixels in the x, y, and z axis
    dX_ppp_ = np.gradient(X_ppp_, axis = 1)
    dY_ppp_ = np.gradient(Y_ppp_, axis = 0)
    dZ_ppp_ = np.sqrt(dY_ppp_**2 + dY_ppp_**2)
    # Set the origen of the Coordinates plane in the Suns
    X_p_ -= X_p_[int(np.around(x_sun_[1])), int(np.around(x_sun_[0]))]
    Y_p_ -= Y_p_[int(np.around(x_sun_[1])), int(np.around(x_sun_[0]))]
    X_ppp_ -= np.tile(np.mean(X_ppp_, axis = 1), (N_x, 1)).T
    Y_ppp_ -= Y_ppp_[int(np.around(x_sun_[1])), int(np.around(x_sun_[0]))]
    # Calulate the Distance from any pixel to the sun
    Z_ppp_ = np.sqrt(X_ppp_**2 + Y_ppp_**2)
    # Stack together the Coordinates system and the incrementes in the coordinates grid
    XYZ_  = np.concatenate((X_ppp_[..., np.newaxis], Y_ppp_[..., np.newaxis], Z_ppp_[..., np.newaxis]), axis = 2)
    dXYZ_ = np.concatenate((dX_ppp_[..., np.newaxis], dY_ppp_[..., np.newaxis], dZ_ppp_[..., np.newaxis]), axis = 2)
    # Transformation of the camerea trjectory to pixels on the images
    x_ = A_sun_.copy()
    x_[0, :] = (x_[0, :] - x_[0, 0]) / angles_per_pixel
    x_[1, :] = (x_[1, :] - x_[1, 0]) / angles_per_pixel
    # Interpolation Pixels Indexes
    #index_ = __prespective_limits(X_ppp_, Y_ppp_, FOV_x, FOV_y, height, epsilon_max = 79)
    # Interpolation Grid for equidistant pixels
    #XY_interp_ = __interpolation_grid(X_ppp_, Y_ppp_, index_, N_x, N_y)
    return XYZ_, dXYZ_, x_

def _perspective_transformation_v1(X_, Y_, N_x, N_y, x_sun_, A_sun_, height,
                                   altitude = 1630, FOV = 63.75, focal_length = 1.3669e-3, pixel_size = 17e-6):

    # Transform the Coordinates Plane to the Camera Plane
    def __camera_plane_coordinates(X_, Y_, dist, const):
        return X_ * dist * const, Y_ * dist * const

    # Project the Cross-Section plane in the Camera pLane
    def __cross_section_plane_coordinates(X_p_, Y_p_, X_pp_, Y_pp_, N_x):
        return X_p_ + np.cumsum(X_pp_/N_x, axis = 1), Y_pp_

    # Calculate the Cross-Section Coodinates System
    def __cross_seccion_coordinates(epsilon, dist, N_x, N_y, radians_per_pixel):
        # Transformation of the y-axis
        def ___y_axis(alpha_y_, height, epsilon):
            angle_upper_ = epsilon + alpha_y_
            angle_lower_ = epsilon - alpha_y_
            idx_ = angle_upper_ > np.pi/2
            y_ = height/np.tan(epsilon)
            y_upper = y_ - height/np.tan(angle_upper_)
            y_lower = height/np.tan(angle_lower_) - y_
            y_p_ = np.concatenate((-np.flip(y_upper), y_lower), axis = 0)
            y_p_+= np.absolute(y_p_[0])
            return y_p_
        # Transformation of the x-axis
        def ___x_axis(y_, alpha_y_, epsilon):
            return ( np.tan(alpha_y_[-1])*y_ )/( np.cos(alpha_y_[-1])*(np.tan(epsilon)*(1./np.tan(alpha_y_[-1])) + 1) )
        # Extend the FOV for each pixels along the y-axis
        alpha_y_ = np.linspace(radians_per_pixel/2, radians_per_pixel*N_y/2, N_y//2)
        # Get the Coordiantes transformation in the 4 Cuadrantes of the Coordinates system
        y_ = ___y_axis(alpha_y_, height, epsilon)
        x_ = ___x_axis(y_, alpha_y_, epsilon)
        # Repeat the vectors along the x-axis
        return np.tile(x_, (N_x, 1)).T, np.tile(y_, (N_x, 1)).T

    # Frame distance projected in the camera plane
    def __get_distances(X_, Y_, height = 13000, epsilon = np.pi/2):
        dist = height/np.sin(epsilon)
        # Camera Plane Coordinates
        X_p_, Y_p_ = __camera_plane_coordinates(X_, Y_, dist, const)
        # Cross-Section Plane Coordinates
        X_pp_, Y_pp_ = __cross_seccion_coordinates(epsilon, dist, N_x, N_y, radians_per_pixel)
        return X_p_, Y_p_, X_pp_, Y_pp_

    # Perpendicular FOV as frame pixels limits
    def __prespective_limits(X_, Y_, X_ppp_, Y_ppp_):
        X_perp_, _, _, Y_perp_ = __get_distances(X_, Y_)
        X_perp_0_, Y_perp_0_, _ = __set_origen(X_perp_, Y_perp_)
        x_lim = X_perp_0_[-1, -1] + 1
        y_lim = Y_perp_0_[-1, -1] + 1
        idx_x_ = np.absolute(X_ppp_) <= x_lim
        idx_y_ = np.absolute(Y_ppp_) <= y_lim
        return idx_x_ & idx_y_

    # Calculate Increments per pixels in the x, y, and z axis
    def __get_increments(X_ppp_, Y_ppp_):
        dX_ppp_ = np.gradient(X_ppp_, axis = 1)
        dY_ppp_ = np.gradient(Y_ppp_, axis = 0)
        dZ_ppp_ = np.sqrt(dY_ppp_**2 + dY_ppp_**2)
        return dX_ppp_, dY_ppp_, dZ_ppp_

    # Interpolation Grid Coordiantes
    def __interpolation_grid(X_, Y_, X_ppp_, Y_ppp_, N_x, N_y):
        # Perpendicular FOV as frame pixels limits
        index_ = __prespective_limits(X_, Y_, X_ppp_, Y_ppp_)
        xx_= []
        Y_ = (N_y - 1)*( Y_ppp_[index_] - np.min(Y_ppp_[index_]))/(np.max(Y_ppp_[index_]) - np.min(Y_ppp_[index_]))
        for y in np.unique(Y_ppp_[index_]):
            idx_ = (Y_ppp_ == y) & index_
            x_ = (N_x - 1)*( X_ppp_[idx_] - np.min(X_ppp_[idx_]))/(np.max(X_ppp_[idx_]) - np.min(X_ppp_[idx_]))
            xx_.append(x_)
        X_ = np.concatenate(xx_, axis = 0)
        return [np.concatenate((X_[:, np.newaxis],  Y_[:, np.newaxis]), axis = 1), index_]

    # Set the origen of the Coordinates plane in the middle of the frame
    def __set_origen(X_ppp_, Y_ppp_, x_sun_, N_x):
        X_ppp_0_ = X_ppp_ - np.tile(np.mean(X_ppp_, axis = 1), (N_x, 1)).T
        Y_ppp_0_ = Y_ppp_ - Y_ppp_[int(np.around(x_sun_[1])), int(np.around(x_sun_[0]))]
        # Calulate the Distance from any pixel to the sun
        Z_ppp_0_ = np.sqrt(X_ppp_0_**2 + Y_ppp_0_**2)
        return X_ppp_0_, Y_ppp_0_, Z_ppp_0_

    # Elevation Angle from Degrees to Radiantes
    epsilon = A_sun_[0, 0]
    epsilon = np.radians(epsilon)
    # Camera Specifications
    angles_per_pixel  = FOV/np.sqrt(N_x**2 + N_y**2)
    radians_per_pixel = np.radians(angles_per_pixel)
    FOV_x = radians_per_pixel * N_x
    FOV_y = radians_per_pixel * N_y
    #print(epsilon, angles_per_pixel, radians_per_pixel, FOV_x, FOV_y)
    # Camera Constants
    const = pixel_size/focal_length
    # Sky-Parcel Geometric Constants
    height = height
    # Frame distance projected in the camera plane and the Cross-section plane y-axis
    X_p_, Y_p_, X_pp_, Y_pp_ = __get_distances(X_, Y_, height, epsilon)
    # Cross-Section Coordinates Projection in the x-axis
    X_ppp_, Y_ppp_ = __cross_section_plane_coordinates(X_p_, Y_p_, X_pp_, Y_pp_, N_x)
    # Calculate Increments per pixels in the x, y, and z axis
    dX_ppp_, dY_ppp_, dZ_ppp_ = __get_increments(X_ppp_, Y_ppp_)
    # Set the origen of the Coordinates plane in the Suns
    X_ppp_, Y_ppp_, Z_ppp_ = __set_origen(X_ppp_, Y_ppp_, x_sun_, N_x)
    # Stack together the Coordinates system and the incrementes in the coordinates grid
    XYZ_  = np.concatenate(( X_ppp_[..., np.newaxis],  Y_ppp_[..., np.newaxis],  Z_ppp_[..., np.newaxis]), axis = 2)
    dXYZ_ = np.concatenate((dX_ppp_[..., np.newaxis], dY_ppp_[..., np.newaxis], dZ_ppp_[..., np.newaxis]), axis = 2)
    # Transformation of the camerea trjectory to pixels on the images
    x_ = A_sun_.copy()
    x_[0, :] = (x_[0, :] - x_[0, 0]) / angles_per_pixel
    x_[1, :] = (x_[1, :] - x_[1, 0]) / angles_per_pixel
    #XY_interp_ = __interpolation_grid(X_, Y_, X_ppp_, Y_ppp_, N_x, N_y)
    return XYZ_, dXYZ_, x_

def _perspective_transformation_v2(X_, Y_, N_x, N_y, x_sun_, A_sun_, height,
                                   altitude = 1630, FOV = 63.75, focal_length = 1.3669e-3, pixel_size = 17e-6):
    # Solve Troposphere and Earth Surface Chords equation
    def __quadratic_solution(W_, e_, height):
        N = e_.shape[0]
        x_ = np.zeros(N)
        y_ = np.zeros(N)
        z_ = np.zeros(N)
        for i in range(N):
            x_[i] = np.roots(W_[:, i])[1]
            y_[i] = x_[i] * np.tan(e_[i])
            z_[i] = height / np.sin(e_[i])
        return x_, y_, z_

    # Compute coefficients of the Troposphere and Earth Surface Chords equation
    def __quadratic_coefficient(e_, r, height):
        N = e_.shape[0]
        a_ = 1 + np.tan(e_)**2
        b_ = 2. * r * np.tan(e_)
        c  = - height * (1. + 2.*r)
        c_ = np.ones(N) * c
        return np.stack((a_, b_, c_))

    # Extend Elevation from Sun's Elevation Angle for each pixel
    def __coordiantes_elevation(Y_, epsilon, FOV_y, radians_per_pixel):
        alpha = epsilon + FOV_y/2
        A_ = alpha - (Y_[:, 0] + 1)*radians_per_pixel
        return np.hstack((alpha, A_))

    # Transform the Coordinates Plane to the Camera Plane
    def __camera_plane_coordinates(X_, Y_, dist, const):
        return X_ * const * dist, Y_ * const * dist

    # Project the Cross-Section plane in the Camera pLane
    def __cross_section_plane_coordinates(X_p_, Y_p_, dz_, alpha_y, N_x):
        dx_ = np.arctan(alpha_y) * dz_
        dX_ = np.tile(dz_, (N_x, 1)).T
        return X_p_ + dX_, Y_p_

    # Vector Increments in each coordiante direction
    def __dxyz(x_, y_, z_):
        dx_ = np.diff(x_)
        dy_ = np.diff(y_)
        dz_ = np.diff(z_)
        x_ = x_[1:]
        y_ = y_[1:]
        z_ = z_[1:]
        return x_, y_, z_, dx_, dy_, dz_

    # Calculate Increments per pixels in the x, y, and z axis
    def __get_increments(X_ppp_, Y_ppp_):
        dX_ppp_ = np.gradient(X_ppp_, axis = 1)
        dY_ppp_ = np.gradient(Y_ppp_, axis = 0)
        dZ_ppp_ = np.sqrt(dY_ppp_**2 + dY_ppp_**2)
        return dX_ppp_, dY_ppp_, dZ_ppp_

    # Reshape Vectors to Matrix form
    def __reshape(x_, y_, z_, N_x):
        X_ = np.tile(x_, (N_x, 1)).T
        Y_ = np.tile(y_, (N_x, 1)).T
        Z_ = np.tile(z_, (N_x, 1)).T
        return X_, Y_, Z_

    # Frame distance projected in the camera plane
    def __get_distances(Y_, height = 13000, epsilon = np.pi/2):
        # Extend Elevation from Sun's Elevation Angle for each pixel
        e_ = __coordiantes_elevation(Y_, epsilon, FOV_y, radians_per_pixel)
        # Compute coefficients of the Troposphere and Earth Surface Chords equation
        W_ = __quadratic_coefficient(e_, r, height)
        # Solve Troposphere and Earth Surface Chords equation
        x_, y_, z_ = __quadratic_solution(W_, e_, height)
        # Vector Increments in each coordiante direction
        x_, y_, z_, dx_, dy_, dz_ = __dxyz(x_, y_, z_)
        # Reshape Vectors to Matrix form
        X_p_, Y_p_, Z_p_ = __reshape(x_, y_, z_, N_x)
        # Calculate Coordiantes projected in the Camera Plane
        X_pp_, Y_pp_ = __camera_plane_coordinates(X_, Y_, Z_p_, const)
        # Calculate Coordiantes projected in the Cross-Section Plane of the Troposphere
        return __cross_section_plane_coordinates(X_pp_, Y_pp_, dz_, alpha_y, N_x)

    # Perpendicular FOV as frame pixels limits
    def __prespective_limits(Y_, X_ppp_, Y_ppp_):
        X_perp_, Y_perp_ = __get_distances(Y_, height = 13000, epsilon = 1.39626)
        #X_perp_, Y_perp_ = __get_distances(Y_, height = 13000, epsilon = np.pi/2)
        X_perp_0_, Y_perp_0_, _ = __set_origen(X_perp_, Y_perp_)
        x_lim_ = X_perp_0_[:, -1] + 1
        y_lim_ = Y_perp_0_[:, -1]
        idx_x_ = np.absolute(X_ppp_) <= np.tile(x_lim_, (N_x, 1)).T
        idx_y_ = np.absolute(Y_ppp_) <= y_lim_[-1]
        return idx_x_ & idx_y_

    # Interpolation Grid Coordiantes
    def __interpolation_grid(X_, Y_, X_ppp_, Y_ppp_, N_x, N_y):
        # Perpendicular FOV as frame pixels limits
        index_ = __prespective_limits(Y_, X_ppp_, Y_ppp_)
        xx_= []
        Y_ = (N_y - 1)*( Y_ppp_[index_] - np.min(Y_ppp_[index_]))/(np.max(Y_ppp_[index_]) - np.min(Y_ppp_[index_]))
        for y in np.unique(Y_ppp_[index_]):
            idx_ = (Y_ppp_ == y) & index_
            x_ = (N_x - 1)*( X_ppp_[idx_] - np.min(X_ppp_[idx_]))/(np.max(X_ppp_[idx_]) - np.min(X_ppp_[idx_]))
            xx_.append(x_)
        X_ = np.concatenate(xx_, axis = 0)
        return [np.concatenate((X_[:, np.newaxis],  Y_[:, np.newaxis]), axis = 1), index_]

    # Set the origen of the Coordinates plane in the middle of the frame
    def __set_origen(X_ppp_, Y_ppp_, x_sun_, N_x):
        X_ppp_0_ = X_ppp_ - np.tile(np.mean(X_ppp_, axis = 1), (N_x, 1)).T
        Y_ppp_0_ = Y_ppp_ - Y_ppp_[int(np.around(x_sun_[1])), int(np.around(x_sun_[0]))]
        # Calulate the Distance from any pixel to the sun
        Z_ppp_0_ = np.sqrt(X_ppp_0_**2 + Y_ppp_0_**2)
        return X_ppp_0_, Y_ppp_0_, Z_ppp_0_

    # Elevation Angle from Degrees to Radiantes
    epsilon = A_sun_[0, 0]
    epsilon = np.radians(epsilon)
    # Camera Specifications
    angles_per_pixel  = FOV/np.sqrt(N_x**2 + N_y**2)
    radians_per_pixel = np.radians(angles_per_pixel)
    FOV_x = radians_per_pixel * N_x
    FOV_y = radians_per_pixel * N_y
    # Camera Constants
    const = pixel_size/focal_length
    alpha_y = radians_per_pixel*FOV_y/2
    # Sky-Parcel Geometric Constants
    r_earth  = 6371000.    # Average Earth radius
    r = r_earth + altitude # Earth radious in albuquerque
    dist = height/np.sin(epsilon)

    # Frame distance projected in the camera plane and the cross-section plane
    X_ppp_, Y_ppp_ = __get_distances(Y_, height, epsilon)
    # Calculate Increments per pixels in the x, y, and z axis
    dX_ppp_, dY_ppp_, dZ_ppp_ = __get_increments(X_ppp_, Y_ppp_)
    # Set the origen of the Coordinates plane in the middle of the frame
    #X_ppp_, Y_ppp_, Z_ppp_= __set_origen(X_ppp_, Y_ppp_)
    # Set the origen of the Coordinates plane in the Sun Position in the frame
    #X_ppp_, Y_ppp_, Z_ppp_ = __set_origen_in_sun(X_ppp_, Y_ppp_, Z_ppp_, x_sun_)

    # Set the origen of the Coordinates plane in the middle of the frame
    X_ppp_, Y_ppp_, Z_ppp_ = __set_origen(X_ppp_, Y_ppp_, x_sun_, N_x)

    # Stack together the Coordinates system and the incrementes in the coordinates grid
    XYZ_  = np.concatenate(( X_ppp_[..., np.newaxis],  Y_ppp_[..., np.newaxis],  Z_ppp_[..., np.newaxis]), axis = 2)
    dXYZ_ = np.concatenate((dX_ppp_[..., np.newaxis], dY_ppp_[..., np.newaxis], dZ_ppp_[..., np.newaxis]), axis = 2)
    # Transformation of the camerea trjectory to pixels on the images
    x_ = A_sun_.copy()
    x_[0, :] = (x_[0, :] - x_[0, 0]) / angles_per_pixel
    x_[1, :] = (x_[1, :] - x_[1, 0]) / angles_per_pixel
    #XY_interp_ = __interpolation_grid(X_, Y_, X_ppp_, Y_ppp_, N_x, N_y)
    return XYZ_, dXYZ_, x_

def _perspective_transformation_v3(X_, Y_, N_x, N_y, x_sun_, A_sun_, height,
                                   altitude = 1630, FOV = 63.75, focal_length = 1.3669e-3, pixel_size = 17e-6):
    # Solve Troposphere and Earth Surface Chords equation
    def __quadratic_solution(W_, e_, height):
        N = e_.shape[0]
        x_ = np.zeros(N)
        y_ = np.zeros(N)
        z_ = np.zeros(N)
        for i in range(N):
            x_[i] = np.roots(W_[:, i])[1]
            y_[i] = x_[i] * np.tan(e_[i])
            z_[i] = height / np.sin(e_[i])
        return x_, y_, z_

    # Compute coefficients of the Troposphere and Earth Surface Chords equation
    def __quadratic_coefficient(e_, r, height):
        N = e_.shape[0]
        a_ = 1 + np.tan(e_)**2
        b_ = 2. * r * np.tan(e_)
        c  = - height * (1. + 2.*r)
        c_ = np.ones(N) * c
        return np.stack((a_, b_, c_))

    # Extend Elevation from Sun's Elevation Angle for each pixel
    def __coordiantes_elevation(x_sun_, epsilon, N_y, radians_per_pixel):
        d_0 = int(np.around(x_sun_[1]))
        d_1 = N_y - d_0
        return np.linspace(epsilon + (d_0 + 1)*radians_per_pixel, epsilon - d_1*radians_per_pixel, N_y + 1)

    # Project the Cross-Section plane in the Camera pLane
    def __cross_section_plane_coordinates(x_, y_, z_, FOV_y, N_x):
        # Calculate Increments per pixels in the x axis in flatland
#         def __x_axis(z_, FOV_y):
#             alpha = FOV_y/2.
#             x_  = 2*np.tan(alpha)*z_[1:]
#             dx_ = x_/N_x
#             return x_, dx_
        # Calculate Increments per pixels in the x axis with curvature
        def __x_axis(z_, FOV_y):
            alpha = FOV_y/2.
            x_ = 2*np.tan(alpha)*z_[1:]
            for i in range(x_.shape[0]):
                # Coefficient
                a = - 4.
                b = 8.*r
                c = - x_[i]**2
                # Solve quadratic Equation
                h = np.roots([a, b, c])[1]
                # Get the arc length
                chi = h + x_[i]**2/(4.*h)
                x_[i] = np.arcsin(x_[i]/chi) * chi
            dx_ = x_/N_x
            return x_, dx_

        # Calculate Increments per pixels in the y axis
        def __y_axis(y_, x_):
            dx_ = np.diff(x_)
            dy_ = np.absolute(np.diff(y_))
            dz_ = dx_/np.cos(np.arctan(dy_/dx_))
            z_  = np.cumsum(dz_)
            return z_, dz_
        # Reshape Vectors to Matrix form
        def __grid(x_, dx_, N_x):
            X_  = np.tile(x_, (N_x, 1)).T
            dX_ = np.tile(dx_, (N_x, 1)).T
            return X_, dX_

        # Calculate Increments per pixels in the x and y axis
        y_, dy_ = __y_axis(y_, x_)
        x_, dx_ = __x_axis(z_, FOV_y)

        # Reshape Vectors to Matrix form
        X_p_, dX_p_ = __grid(x_, dx_, N_x)
        Y_p_, dY_p_ = __grid(y_, dy_, N_x)
        dZ_p_ = np.sqrt(dX_p_**2 + dY_p_**2)
        return X_p_, Y_p_, dX_p_, dY_p_, dZ_p_

    # Set the origen of the Coordinates plane in the Sun Position in the frame
    def __set_origen_in_sun(X_, Y_, x_sun_, N_x):
        x_0 = int(np.around(x_sun_[0]))
        y_0 = int(np.around(x_sun_[1]))

        x_0_ = X_[:, x_0]
        X_0_ = np.zeros(X_.shape)

        d_0 = N_x/x_0
        d_1 = N_x/(N_x - x_0)

        for i in range(x_0_.shape[0]):
            X_0_[i, :] = np.linspace(-x_0_[i]/d_0, x_0_[i]/d_1, N_x)

        Y_0_ = Y_ - Y_[y_0, x_0]
        Z_0_ = np.sqrt(X_0_**2 + Y_0_**2)
        return X_0_, Y_0_, Z_0_

    # Perpendicular FOV as frame pixels limits
    def __prespective_limits(X_ppp_, Y_ppp_):
        X_perp_, Y_perp_, _, _, _ = __get_distance(height = 13000, epsilon = 1.39626)
        #X_perp_, Y_perp_, _, _, _ = __get_distance(height = 13000, epsilon = np.pi/2)

        X_perp_0_, Y_perp_0_, Z_perp_0_ = __set_origen_in_sun(X_perp_, Y_perp_, x_sun_, N_x)
        x_lim_ = X_perp_0_[:, -1]
        y_lim_ = Y_perp_0_[:, -1]

        idx_x_ = X_ppp_ <= np.tile(x_lim_, (N_x, 1)).T + 1
        idx_y_ = ( Y_ppp_ < y_lim_.max() + 1) & ( Y_ppp_ > y_lim_.min() - 1 )

        return idx_x_ & idx_y_

    def __interpolation_grid(X_, Y_, X_ppp_, Y_ppp_, N_x, N_y):
        # Perpendicular FOV as frame pixels limits
        index_ = __prespective_limits(X_ppp_, Y_ppp_)
        xx_= []
        Y_ = (N_y - 1)*( Y_ppp_[index_] - np.min(Y_ppp_[index_]))/(np.max(Y_ppp_[index_]) - np.min(Y_ppp_[index_]))
        for y in np.unique(Y_ppp_[index_]):
            idx_ = (Y_ppp_ == y) & index_
            x_ = (N_x - 1)*( X_ppp_[idx_] - np.min(X_ppp_[idx_]))/(np.max(X_ppp_[idx_]) - np.min(X_ppp_[idx_]))
            xx_.append(x_)
        X_ = np.concatenate(xx_, axis = 0)
        return [np.concatenate((X_[:, np.newaxis],  Y_[:, np.newaxis]), axis = 1), index_]

    def __get_distance(epsilon, height):
        # Calculate Increments per pixels in the elevation
        e_ = __coordiantes_elevation(x_sun_, epsilon, N_y, radians_per_pixel)
        # Calculate Coefficient on the chord second order equation
        W_ = __quadratic_coefficient(e_, r, height)
        # Solution of the Quadratic Equation
        x_, y_, z_ = __quadratic_solution(W_, e_, height)
        # Frame distance projected in the cross-section plane
        return __cross_section_plane_coordinates(x_, y_, z_, FOV_y, N_x)

    # Elevation Angle from Degrees to Radiantes
    epsilon = A_sun_[0, 0]
    epsilon = np.radians(epsilon)
    # Camera Specifications
    angles_per_pixel  = FOV/np.sqrt(N_x**2 + N_y**2)
    radians_per_pixel = np.radians(angles_per_pixel)
    FOV_x = radians_per_pixel * N_x
    FOV_y = radians_per_pixel * N_y
    # Camera Constants
    const = pixel_size/focal_length
    alpha_y = radians_per_pixel*FOV_y/2
    # Sky-Parcel Geometric Constants
    r_earth  = 6371000.    # Average Earth radius
    r = r_earth + altitude # Earth radious in albuquerque
    # Frame distance projected in the camera plane and the cross-section plane
    X_p_, Y_p_, dX_p_, dY_p_, dZ_p_ = __get_distance(epsilon, height)
    # Set the origen of the Coordinates plane in the Sun Position in the frame
    X_0_, Y_0_, Z_0_ = __set_origen_in_sun(X_p_, Y_p_, x_sun_, N_x)

    # Stack together the Coordinates system and the incrementes in the coordinates grid
    XYZ_  = np.concatenate((X_0_[..., np.newaxis], Y_0_[..., np.newaxis], Z_0_[..., np.newaxis]), axis = 2)
    dXYZ_ = np.concatenate((dX_p_[..., np.newaxis], dY_p_[..., np.newaxis], dZ_p_[..., np.newaxis]), axis = 2)
    # Transformation of the camerea trjectory to pixels on the images
    x_ = A_sun_.copy()
    x_[0, :] = (x_[0, :] - x_[0, 0]) / angles_per_pixel
    x_[1, :] = (x_[1, :] - x_[1, 0]) / angles_per_pixel
    XY_interp_ = __interpolation_grid(X_, Y_, X_0_, Y_0_, N_x, N_y)
    return XYZ_, dXYZ_, x_

def _perspective_transformation_v4(X_, Y_, N_x, N_y, x_sun_, A_sun_, height,
                                   altitude = 1630, FOV = 63.75, focal_length = 1.3669e-3, pixel_size = 17e-6, display = False):
    # Solve Troposphere and Earth Surface Chords equation
    def __solve_quadratic_formula(W_, e_, height):
        N = e_.shape[0]
        x_ = np.zeros(N)
        y_ = np.zeros(N)
        z_ = np.zeros(N)
        # Solve quadratic formula for each elevation angle in a frame
        for i in range(N):
            x_[i] = np.roots(W_[:, i])[1]
            y_[i] = x_[i] * np.tan(e_[i])
            z_[i] = y_[i] / np.sin(e_[i])
        return x_, y_, z_
    # Compute coefficients of the Troposphere and Earth Surface Chords equation
    def __quadratic_coefficient(e_, r, height):
        N = e_.shape[0]
        # Compute quadratic formula coefficients for each elevation angle in a frame
        a_ = 1 + np.tan(e_)**2
        b_ = 2. * r * np.tan(e_)
        c  = - height * (1. + 2.*r)
        c_ = np.ones((N, 1)) * c
        return np.concatenate((a_, b_, c_), axis = 1).T
    # Extend Elevation from Sun's Elevation Angle for each pixel
    def __coordiantes_elevation(x_sun_, epsilon, N_y, radians_per_pixel):
        y_0 = x_sun_[1]
        d_0 = y_0
        d_1 = N_y - y_0
        return np.linspace(epsilon + (d_0 + 1)*radians_per_pixel, epsilon - d_1*radians_per_pixel, N_y + 1)
    # Extend azimuth from camera FOV Angle for each pixel
    def __coordiantes_azimuth(x_sun_, asimuth, N_x, radians_per_pixel):
        a_ = np.linspace(0, N_x + 1, N_x + 2)
        return np.absolute(a_ -  a_.mean())*radians_per_pixel
    # Project the Cross-Section plane in the Camera pLane in y-axis
    def __y_axis(x_, y_, z_, height):
        # Ciruclar Segment formula
        rho_ = height - y_
        k_ = rho_ + (x_**2)/rho_
        s_ = .5*np.arcsin(2.*x_/k_)*k_
        # compute y-axis absolute distance
        y_  = s_[1:] - s_[0]
        # Differencital distance between pixels
        dy_ = np.diff(s_)
        # Extend values to the frame dimensions grid
        dY_ = np.tile(dy_, (N_x, 1)).T
        Y_  = np.tile( y_, (N_x, 1)).T
        return Y_, dY_
    # Project the Cross-Section plane in the Camera pLane in x-axis
    def __x_axis(z_, azimuth_, r):
        # Quadratic formular coefficients
        def ___quadratic_coeff(dx, r):
            a = -1.
            b = 2. * r
            c = (dx/2.)**2
            return [a, b, c]
        # Solve Quadratic formular
        def ___quadratic_solver(w_, dx):
            # Find solution for given coefficients
            l = np.roots(w_)[1]
            # Ciruclar Segment formula
            k = l + (dx**2)/(4.*l)
            s = np.arcsin(dx/k) * k
            return l, k, s
        # Compute the increments in the x-axis using the FOV in the x-axis of the camera
        def ___inc(z_, azimuth_, i, j):
            alpha = np.absolute(azimuth_[j - 1]) + np.absolute(azimuth_[-j])
            x = 2.*z_[i]*np.tan(alpha/2.)
            return x
        # Variables initialization
        N = z_.shape[0]
        M = azimuth_.shape[0]
        X_  = np.zeros((N, M))
        dX_ = np.zeros((N, M))
        # Loop over pixels
        for i in range(N):
            for j in range(1, M//2 + 1):
                # Calculate incremental differnce in the x-axis
                dx = ___inc(z_, azimuth_, i, j)
                # Quadratic formula coeffcient
                w_ = ___quadratic_coeff(dx, r)
                # Solve for given coeffcient and estimate x-axis (i,j) coordiante value
                l, k, x = ___quadratic_solver(w_, dx)
                # x-axis simetric distance
                X_[i, j - 1] = -x/2.
                X_[i, -j]    =  x/2.
        # Compute x-axis differential pixels distance
        dx_ = np.diff(X_[:, :M//2], axis = 1)
        dX_ = np.concatenate((dx_, dx_[:, ::-1]), axis = 1)
        return X_[1:, 1:-1], dX_[1:, :]
    # Project the Cross-Section plane in the Camera pLane in distance from the Sun's potion
    def __z_axis(X_, Y_, dX_, dY_):
        # Geometric distance
        Z_  = np.sqrt( X_**2 +  Y_**2)
        # Differential ditance
        dZ_ = np.sqrt(dX_**2 + dY_**2)
        return Z_, dZ_
    # Set y-axis origen in the current position of the Sun
    def __set_origen_in_sun(X_0_, Y_0_, Z_0_, x_sun_):
        y_0 = int(np.around(x_sun_[1]))
        # Set y-axis
        Y_0_ -= Y_0_[y_0, :]
        # Corrent distance with newly computed centered grid of the y-axis
        Z_0_ = np.sqrt(X_0_**2 + Y_0_**2)
        return X_0_, Y_0_, Z_0_

    # Perpendicular FOV as frame pixels limits
    def __prespective_limits(X_, Y_, Z_):
        # Set origin in the current epsilon position
        X_, Y_, Z_ = __set_origen_in_sun(X_, Y_, Z_, x_sun_)
        # Get the axis in the 90 degrees epsilon position
        X_0_, Y_0_, Z_0_, dX_0_, dY_0_, dZ_0_ = __get_axes(x_sun_, epsilon = 1.39626, azimuth = 0, height = 1000)
        # Set origin in the 90 degrees epsilon position
        X_0_, Y_0_, Z_0_ = __set_origen_in_sun(X_0_, Y_0_, Z_0_, x_sun_)
        # Maximum distance in y-axis
        x_lim_ = X_0_[:, -1]
        # Index within the maximum distnace in x-axis
        idx_x_ = np.absolute(X_) <= np.tile(x_lim_, (N_x, 1)).T + 1
        # Maximum distance in y-axis
        y_lim_ = Y_0_[:, -1]
        # Index within the maximum distnace in y-axis
        idx_y_ = ( Y_ < y_lim_.max() + 1) & ( Y_ > y_lim_.min() - 1 )
        return idx_x_ & idx_y_

    def __interpolation_grid(X_p_, Y_p_, Z_p_, dX_, dY_, dZ_):
        # Perpendicular FOV as frame pixels limits
        index_ = __prespective_limits(X_p_, Y_p_, Z_p_)
        xx_= []
        Y_ = (N_y - 1)*( Y_p_[index_] - np.min(Y_p_[index_]))/(np.max(Y_p_[index_]) - np.min(Y_p_[index_]))
        for y in np.unique(Y_p_[index_]):
            idx_ = (Y_p_ == y) & index_
            x_ = (N_x - 1)*( X_p_[idx_] - np.min(X_p_[idx_]))/(np.max(X_p_[idx_]) - np.min(X_p_[idx_]))
            xx_.append(x_)
        X_ = np.concatenate(xx_, axis = 0)
        return [np.concatenate((X_[:, np.newaxis],  Y_[:, np.newaxis]), axis = 1), index_]

    def __get_axes(x_sun_, epsilon, azimuth, height):
        # Set the origen of the Coordinates plane in the Sun Position in the frame
        epsilon_ = __coordiantes_elevation(x_sun_, epsilon, N_y, radians_per_pixel)
        azimuth_ = __coordiantes_azimuth(x_sun_, azimuth, N_x, radians_per_pixel)
        # Get throposhere quadrative formulat solution coeffiencients
        W_ = __quadratic_coefficient(epsilon_, r, height)
        # Solve the quadrative formulate for the given coefficients
        x_, y_, z_ = __solve_quadratic_formula(W_, epsilon_, height)
        # Get the axis distance and differential distances
        Y_, dY_ = __y_axis(x_, y_, z_, height)
        X_, dX_ = __x_axis(z_, azimuth_, r)
        Z_, dZ_ = __z_axis(X_, Y_, dX_, dY_)
        return X_, Y_, Z_, dX_, dY_, dZ_

    # Elevation Angle from Degrees to Radiantes
    epsilon = A_sun_[0, 0]
    epsilon = np.radians(epsilon)
    azimuth = 0
    # Camera Specifications
    angles_per_pixel  = FOV/np.sqrt(N_x**2 + N_y**2)
    radians_per_pixel = np.radians(angles_per_pixel)
    FOV_x = radians_per_pixel * N_x
    FOV_y = radians_per_pixel * N_y
    # Camera Constants
    const = pixel_size/focal_length
    alpha_y = radians_per_pixel*FOV_y/2
    # Sky-Parcel Geometric Constants
    r_earth = 6371000.     # Average Earth radius
    r = r_earth + altitude # Earth radious in albuquerque
    # Get axis in the current degrees epsilon position
    X_, Y_, Z_, dX_, dY_, dZ_ = __get_axes(x_sun_, epsilon, azimuth, height)
    # Stack together the Coordinates system and the incrementes in the coordinates grid
    XYZ_  = np.concatenate(( X_[..., np.newaxis],  Y_[..., np.newaxis],  Z_[..., np.newaxis]), axis = 2)
    dXYZ_ = np.concatenate((dX_[..., np.newaxis], dY_[..., np.newaxis], dZ_[..., np.newaxis]), axis = 2)
    # Transformation of the camerea trjectory to pixels on the images
    x_ = A_sun_.copy()
    x_[0, :] = (x_[0, :] - x_[0, 0]) / angles_per_pixel
    x_[1, :] = (x_[1, :] - x_[1, 0]) / angles_per_pixel
    # Get index of the epsilon 90 grid projected in the current epsilon grid
    XY_interp_ = __interpolation_grid(X_, Y_, Z_, dX_, dY_, dZ_)

    if display:
        plt.figure(figsize = (20, 5))
        plt.scatter(XYZ_[..., 0].flatten(), XYZ_[..., 1].flatten(), s = 10)
        plt.scatter(XYZ_[XY_interp_[1], 0].flatten(), XYZ_[XY_interp_[1], 1].flatten(), s = 10)
        plt.grid()
        plt.gca().invert_yaxis()
        plt.show()

    return XYZ_, dXYZ_, x_


# Radiometry Functionality of the Infrared Camera with Shutter
def _infrared_radiometry(I_, ws_, i_sun_max, elevation = 1615, verbose = False):
    # Calculate the minumum height of cloud
    def __cloud_base(t_air, t_dew):
        return 304.8*(t_air - t_dew)/2.5
    # Transform Infrared Image Pixels Intensity to Kelvin
    def __intensity_to_kelvin(I_):
        return I_/100.0
    def __cloud_height(T_cloud_, t_air, alpha):
        return (t_air - T_cloud_) / alpha
    # Moist Adiabatic Lapse Rate
    def __MALR(T, T_d, p):
        # Earth's gravitational acceleration = 9.8076 m/s2
        g = 9.8076
        # heat of vaporization of water = 2501000 J/kg
        H_v = 2501000
        # specific gas constant of dry air = 287 J/kg·K
        R_sd = 287
        # specific gas constant of water vapour = 461.5 J/kg·K
        R_sw = 461.5
        # the dimensionless ratio of the specific gas constant of dry air to the specific gas constant for water vapour = 0.622
        epsilon = R_sd/R_sw
        # the water vapour pressure of the saturated air
        #e = epsilon * np.exp( (7.5 * T)/(273.3 + T) ) * 100 # Actual
        e = epsilon * np.exp( (7.5 * T_d)/(273.3 + T_d) ) * 100 # Saturates in hPa but tansform to Pa
        # the mixing ratio of the mass of water vapour to the mass of dry air
        r = (epsilon*e)/(p - e)
        # the specific heat of dry air at constant pressure, = 1003.5 J/kg·K
        c_pd = 1003.5
        return g * ( (R_sd * T**2) + (H_v * r * T) ) / ( (c_pd * R_sd * T**2) + (H_v**2 * r * epsilon) )
    # Celcius to Kelvin and mmHg to Pa
    def __convert_units(T, T_d, p):
        return T + 273.15, T_d + 273.15, p * 133.322
    # Pixels Intensity to Kelvin
    K_ = __intensity_to_kelvin(I_)
    # Air and Dew Temperature from Celcius to Kelvin
    t_air = ws_[0]
    t_dew = ws_[1]
    p_air = ws_[2]
    t_air, t_dew, p_air = __convert_units(t_air, t_dew, p_air)
    # Minimum hieght for physically feaseble cloud
    h_base = __cloud_base(t_air, t_dew)
    # Integrate Temperature Measurements to find average height of an object on the IR images
    gamma_elr  = .00649
    gamma_dalr = .00984
    gamma_malr = .005
    gamma      = __MALR(t_air, t_dew, p_air)
    H_ = __cloud_height(K_, t_air, alpha = gamma)
    # Regularized heigts of objects in the images
    tau = h_base - .5 / gamma
    tau = 500.
    H_[H_ < tau] = tau
    if verbose:
        print('>> T. Air: {} [C] T. Dew: {} [C] P. {} [mmHg] Laspe Rate: {} [C/m] '
              'Could Base: {} [m]'.format(ws_[0], ws_[1], ws_[2], gamma, h_base))
    return K_, H_

# Find Sun coordinates on a given frame
def _hough_sun_coordinates(I_, x_prev_, tau, num_peaks):
    # Keep static the Sun postions when it is occluded
    def __sun_occlusion_coordinates(x_prev_, x_now_):
        # It wasn't occuluded ...
        if x_prev_[0] != 40. and x_prev_[1] != 30.:
            # It's occuluded ...
            if x_now_[0] == 40. and x_now_[1] == 30.:
                x_now_ = x_prev_.copy()
        return x_now_

    def __hough_transform(image, tau, num_peaks):
        # Select the most prominent 3 circles, Detect two radii
        accums, cx, cy, radii = hough_circle_peaks(hough_circle(image > tau, 2), np.arange(2, 4, 2), total_num_peaks = num_peaks)
        try:
            return np.average(cx, weights = accums), np.average(cy, weights = accums)
        except:
            return np.array([]), np.array([])
    x_sun = 40.
    y_sun = 30.
    x_ = __hough_transform(I_, tau, num_peaks)
    if x_[0].size != 0 and x_[1].size != 0:
        x_sun = x_[0]
        y_sun = x_[1]
    x_now_ = np.array((x_sun, y_sun))[:, np.newaxis]
    # Keep previous position detecte in case of occlusion
    return __sun_occlusion_coordinates(x_prev_, x_now_)

# Window Artifacts Persistent model
def _window_persistent_model_v3(_tools, I_scatter_, label, n_samples, n_burnt, verbose = False):
    def __update_model(_tools, I_scatter_):
        _tools.W_lag_.append(I_scatter_[..., np.newaxis])
        # Add a new sample but forget the last sample that was added to the list
        if len(_tools.W_lag_) > n_samples:
            # If there are enough clear sky samples make a model of the artifacts in the window
            _tools.W_ = np.median(np.concatenate(_tools.W_lag_, axis = 2)[..., :-n_burnt], axis = 2)
        return _tools
    # Add a new sample if there is clear sky conditions
    #idx_ = __artifacts_segmentation_(I_norm_, tau)
    # Add a new sample if there is clear sky conditions
    if label == 0:
        tools_ = __update_model(_tools, I_scatter_)
    if verbose:
        print('>> Persistent Model Info: label = {} No. Samples = {}'.format(label, len(_tools.W_lag_)))
    return _tools

# Remove artifacts on the window with a persistent model of the bacground
def _remove_window_artifacts(_tools, I_, I_scatter_, tau, verbose = False):
    # Only Cloud in the IR Image
    def __detect_artifacts(I_, tau, min):
        I_bin_ = (I_ - min) > tau
        return I_bin_.sum()/48., I_.max() - min
    # Remove window Artifacts
    def __remove_window(I_aux_, W_aux_):
        return I_aux_ - (W_aux_ - W_aux_.min())

    # Copy varibles so they are not overwrite in the main function
    W_reflected_ = _tools.W_.copy()
    I_global_    = I_.copy()
    I_aux_       = I_scatter_.copy()

    # Calculate Abnormal Statistics
    I_percentage, I_delta = __detect_artifacts(I_aux_, tau, min = 0)
    W_percentage, W_delta = __detect_artifacts(W_reflected_, tau, min = W_reflected_.min())

    # Display stats
    if verbose: print('>> Obstruction Info: > Image Percentage = {} Delta = {} '
                      '> Window Percentage = {} Delta = {}'.format(I_percentage, I_delta, W_percentage, W_delta))

    # Normalize Only Clouds Image
    if I_percentage < 95. and W_delta < 1500. and W_percentage < 1.5:
        I_direct_  = __remove_window(I_global_, W_reflected_)
        I_diffuse_ = __remove_window(I_aux_, W_reflected_)
        return I_global_, I_direct_, I_diffuse_
    else:
        return I_global_, I_global_, I_aux_

# Classification of current frame Atmospheric Conditions
def _atmospheric_condition(_tools, i, K_0_, M_lk_, model_, tau = 0.05, n_samples = 50, verbose = False):
    # Compute Statistics
    def __get_stats(X_):
        # Mean, Std, Skew, and Kurtosis
        return np.array((np.mean(X_), np.std(X_), skew(X_), kurtosis(X_)))
    # Regression Output to Classification Label
    def __robust_classification(_SVC, csi_var_now, csi_var_past, X_):
        if csi_var_now != 1. and csi_var_now < tau and csi_var_past < tau:
            return 0
        else:
            return __classification(_SVC, X_)
    # For robustness... Select Most Frequent Label on a labels lagged-list
    def __lag_labels(y, labels_lag_, field_lag):
        # Keep the desire number of lags on the list by removing the last and aadding at the bigging
        if len(labels_lag_) == field_lag:
            labels_lag_.pop(0)
        labels_lag_.append(y)
        return mode(labels_lag_)[0][0].astype(int), labels_lag_
    # Predict
    def __classification(_SVC, X_):
        return _SVC.predict(PolynomialFeatures(degree).fit_transform(X_))[0]

    # Extract model Parameters
    _SVC   = model_[0]
    degree = model_[1]
    # How flat was the CSI?
    csi_var_past = np.absolute(np.sum(_tools.csi_[i - n_samples:i])/n_samples - 1.)
    csi_var_now  = np.absolute(_tools.csi_[i] - 1.)
    # Get weather features
    #w_ = np.array(_tools.pres_[i])[np.newaxis]
    w_ = np.array(_tools.csi_[i])[np.newaxis]
    # get stats of the images
    k_ = __get_stats(K_0_.flatten())
    # Get velocity vectors features
    #m_ = __get_stats(M_.flatten(), _skew = False)
    # Concatenate all selected features
    X_ = np.concatenate((w_, k_), axis = 0)[np.newaxis]
    # Robust SVM Multi-label Classification
    # Robust classification persistent_v7, threshold segmentation tau = 12.12
    y_hat = __robust_classification(_SVC, csi_var_now, csi_var_past, X_)
    # SVM Multi-label Classification
    #y_hat = __classification(_SVC, X_)
    # Return label for the segmentation type for this sky condition
    y_hat, _tools.labels_ =  __lag_labels(y_hat, labels_lag_ = _tools.labels_, field_lag = 3)
    if verbose:
        if y_hat == 0: print('>> Clear Sky ')
        if y_hat == 1: print('>> Cumulus Cloud')
        if y_hat == 2: print('>> Stratus Cloud')
        if y_hat == 3: print('>> Nimbus Cloud')
    return y_hat, _tools

# Find Sun coordinates on a given frame
def _sun_coordinates(I_, x_prev_, tau = 45057.):
    # Keep static the Sun postions when it is occluded
    def __sun_occlusion_coordinates(x_prev_, x_now_):
        # It wasn't occuluded ...
        if x_prev_[0] != 40. and x_prev_[1] != 30.:
            # It's occuluded ...
            if x_now_[0] == 40. and x_now_[1] == 30.:
                x_now_ = x_prev_.copy()
        return x_now_
    x_sun = 40.
    y_sun = 30.
    x_ = np.where(I_ >= tau)
    if x_[0].size != 0 and x_[1].size != 0:
        x_sun = np.mean(x_[1])
        y_sun = np.mean(x_[0])
    x_now_ = np.array((x_sun, y_sun))[:, np.newaxis]
    # Keep previous position detecte in case of occlusion
    return __sun_occlusion_coordinates(x_prev_, x_now_)

# Find Sun coordinates on a given frame
def _sun_coordinates_persistent(_tools, I_, x_prev_, tau = 45057.):
    # For robustness... Select Most Frequent Label on a labels lagged-list
    def __lag_position(x_, flag, position_lag_, field_lag):
        # Keep the desire number of lags on the list by removing the last and aadding at the bigging
        if flag:
            position_lag_.append(x_)
            if len(position_lag_) > field_lag:
                position_lag_.pop(0)
        return mode(position_lag_)[0][0], position_lag_
    x_ = np.where(I_ >= tau)
    # Keep static the Sun postions when it is occluded
    if x_[0].size != 0 and x_[1].size != 0:
        x_now_ = np.array((np.mean(x_[1]), np.mean(x_[0])))[:, np.newaxis]
        # Keep previous position detecte in case of occlusion
        x_now_, _tools.position_ = __lag_position(x_now_, flag = True, position_lag_ = _tools.position_, field_lag = 6)
    else:
        x_now_, _tools.position_ = __lag_position(None, flag = False, position_lag_ = _tools.position_, field_lag = 6)
    return x_now_


# Save Persistent Model and Lag of frames
def _save_window_persistent_model(_tools, name_, n_samples):
    if len(_tools.W_lag_) > n_samples:
        idx_ = np.random.permutation(len(_tools.W_lag_))[:n_samples]
        _tools.W_lag_ = [_tools.W_lag_[i] for i in idx_]
        _tools.W_ = np.median(np.concatenate(_tools.W_lag_, axis = 2), axis = 2)
    _save_file([_tools.W_, _tools.W_lag_], name_)

__all__ = ['_perspective_transformation_v0', '_normalize_infrared_image', '_sun_coordinates',
           '_remove_window_artifacts', '_atmospheric_effect_v11', '_perspective_transformation_v1',
           '_perspective_transformation_v4', '_window_persistent_model_v3', '_hough_sun_coordinates',
           '_save_window_persistent_model', '_sun_pixels_interpolation', '_perspective_transformation_v3',
           '_cloud_pixels_labeling', '_sun_coordinates_persistent', '_polar_coordinates_transformation',
           '_infrared_radiometry', '_fill_holes', '_regularize_infrared_image', '_normalization']
