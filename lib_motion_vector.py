import numpy as np
import matplotlib.pyplot as plt

#from cv2 import calcOpticalFlowFarneback, OPTFLOW_FARNEBACK_GAUSSIAN, OPTFLOW_USE_INITIAL_FLOW

from numpy.fft import fft, ifft, fft2, ifft2, fftshift

from scipy.signal import correlate2d as _correlation
from scipy.signal import convolve2d, correlate
from scipy.optimize import fmin
from scipy.stats import multivariate_normal
from scipy.interpolate import interp2d, RectBivariateSpline, griddata
from scipy.ndimage.filters import median_filter, gaussian_filter, uniform_filter

from sklearn import preprocessing
from sklearn.svm import SVR, LinearSVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import LeaveOneOut, KFold

from utils import *

# Gaussian Weighted Window
def _weighted_window(w, sigma):
    N = w*2 + 1
    if sigma > 0:
        # Define coodiantes of Window in x and y axis
        x_ = np.linspace(0., N - 1, N)
        y_ = np.linspace(0., N - 1, N)
        X_, Y_ = np.meshgrid(x_, y_)
        # Combine x and y axis in matrix form
        X_ = np.stack((X_.flatten(), Y_.flatten()), axis = 1)
        # Define Normal Distribution Paramters
        mu_ = np.ones((2))*w
        sigma_ = np.eye(2)*sigma
        # Define Normal Distribution
        _N = multivariate_normal(mu_, sigma_)
        # Evaluate Normal Distribution for the Window
        return _N.pdf(X_).reshape(N, N)
    else:
        return np.ones((N, N))


# Define the interpolation coordiantes for each pyramid level
def _interpolation_coordiantes(N, M, n, m):
    return np.linspace(0, M - 1, m), np.linspace(0, N - 1, n)

# Reobust Number of Levels in the pyramid, so window can be implemented
def _max_pyramid_level(no_pyramid, N, M, w):
    k = 2.*w + 1
    max_pyramid = int(np.min([N//k, M//k]))
    if no_pyramid > max_pyramid:
        no_pyramid = max_pyramid
        print('No. Levels Maximum through Window is: {}'.format(no_pyramid))
    if no_pyramid > 5:
        no_pyramid = 5
        print('No. Levels Maximum is: {}'.format(no_pyramid))
    return no_pyramid

# Reshape Image to the pyramid scale dimensions
def _reduce_image_scale(x_0_, y_0_, x_, y_, I_):
    _f = interp2d(x_0_, y_0_, I_, kind = "linear")
    return _f(x_, y_)

# Increase pyramid flow to the original image dimensions
def _increase_flow_scale(x_0_, y_0_, x_, y_, V_):
    # Velocity Vectors x-component
    _f_x = interp2d(x_, y_, V_[..., 0], kind = "linear")
    V_x_ = _f_x(x_0_, y_0_)
    # Velocity Vectors y-component
    _f_y = interp2d(x_, y_, V_[..., 1], kind = "linear")
    V_y_ = _f_y(x_0_, y_0_)
    return np.concatenate((V_x_[..., np.newaxis], V_y_[..., np.newaxis]), axis = 2)

def _pyramidal_posterior_weighted_lucas_kanade(I_1_, I_2_, W_, window_size, sigma, n_pyramid, tau = 1e-10, step_size = 1):
    """Parameters:
        I_1_ – first 8-bit single-channel input image.
        I_2_ – second input image of the same size and the same type as prev
        window_size – averaging window size; larger values increase the algorithm robustness to image noise
                  and give more chances for fast motion detection, but yield more blurred motion field.
        kernel_sigma - amplitude paramter for the time convolution kernel to adjusts the magnitude of the flow
        n_pyramid - No. of pyramids if pyramidal flow comoputation.
        wls_sigma - Gaussian Neirborhoog Weights standar ddeviation
        tau - Eigenvalues threshold to reduce the amount of noisy vectors on the flow
        step_size - Pixels jump in between windows."""

    # Type of variable definition
    window_size  = int(window_size)
    kernel_sigma = sigma
    n_pyramid = int(n_pyramid)
    tau       = tau
    step_size = int(step_size)
    #print(window_size, tau, kernel_sigma, n_pyramid, wls_sigma)

    # Update pixel coordinates
    def __check_corners(x_, x, w):
        # Cross-Correlation windows upper and lower interval
        x_1_, x_2_ = x_ - w, x_ + w + 1
        # Forze corners to stay on the image frame
        if x_1_ < 0:     x_1_ = 0
        if x_2_ > x - 1: x_2_ = x - 1
        return x_1_, x_2_

    def __sample_partial_derivations(f_x_, f_y_, f_t_, Q_, K_, i, j, d, n, w):
        # Adjust Pixel coordinates to have enough pixels on edges windows
        i_1, i_2 = __check_corners(i, d, w)
        j_1, j_2 = __check_corners(j, n, w)
        # Implement Lucas Kanade for each point, calculate I_x, I_y, I_t
        I_x = f_x_[i_1:i_2, j_1:j_2].flatten()[:, np.newaxis]
        I_y = f_y_[i_1:i_2, j_1:j_2].flatten()[:, np.newaxis]
        I_t = f_t_[i_1:i_2, j_1:j_2].flatten()[:, np.newaxis]
        # System Weights
        q_ = Q_[i_1:i_2, j_1:j_2].flatten()[:, np.newaxis]
        k_ = K_[i_1:i_2, j_1:j_2].flatten()[:, np.newaxis]
        # Matrix form Dataset
        X_ = np.concatenate((I_x, I_y), axis = 1)
        y_ = I_t
        return X_, y_, q_, k_

    # Solve Optical FLow for the defined grid
    def __solve_optical_flow(f_x_, f_y_, f_t_, Q_, K_, w, d, n, step_size):
        # Weighted Least-Squares Solution
        # def ___WLS(X_, y_, w_, tau):
        #     B_ = np.eye(w_.shape[0]) * w_
        #     return np.linalg.pinv(X_.T @ B_ @ X_ + np.eye(2)*tau) @ (X_.T @ B_ @ y_)
        # Multi-output Weighted Least-Squares Solution
        def ___MOWLS(X_, y_, q_, k_, tau):
            w_tilde_ = np.concatenate((q_, k_), axis = 0)
            y_tilde_ = np.concatenate((y_, y_), axis = 0)
            X_tilde_ = np.asarray(np.kron(np.eye(2), X_))
            W_tilde_ = np.eye(w_tilde_.shape[0]) * w_tilde_
            return np.linalg.pinv(X_tilde_.T @ W_tilde_ @ X_tilde_ + np.eye(4)*tau) @ (X_tilde_.T @ W_tilde_ @ y_tilde_)
        # Variables Initialization
        U_ = np.zeros((d, n, 2))
        V_ = np.zeros((d, n, 2))
        # Loop over pixels rows and coloumns
        for i in range(0, d, step_size):
            for j in range(0, n, step_size):
                # Implement Lucas Kanade for each point, calculate I_x, I_y, I_t
                X_, y_, q_, k_ = __sample_partial_derivations(f_x_, f_y_, f_t_, Q_, K_, i, j, d, n, w)
                # Solve weighted-Linear System or linear
                w_ = ___MOWLS(X_, y_, q_, k_, tau)
                # get velocity Components
                U_[i, j, 0] = w_[0]
                V_[i, j, 0] = w_[1]
                U_[i, j, 1] = w_[2]
                V_[i, j, 1] = w_[3]
        return U_, V_

    # Get the numerical 2D-differenciation of the images
    def __compute_numerical_derivatives(I_1_, I_2_, K_x_, K_y_, K_t_):
        # Numerical derivative convolutions
        f_x_ = convolve2d(I_1_, K_x_, boundary = 'symm', mode = 'same')
        f_y_ = convolve2d(I_1_, K_y_, boundary = 'symm', mode = 'same')
        f_t_ = convolve2d(I_2_, K_t_, boundary = 'symm', mode = 'same') + convolve2d(I_1_, -K_t_, boundary = 'symm', mode = 'same')
        return f_x_, f_y_, f_t_

    # All pixels within [-w, w] are in the window.
    w = int(window_size/2)
    # normalize pixels
    I_1_ = I_1_/255.
    I_2_ = I_2_/255.
    # Numerical derivative Kernels
    K_x_ = np.array([[-1.,  1.], [-1., 1.]])
    K_y_ = np.array([[-1., -1.], [ 1., 1.]])
    K_t_ = np.array([[ 1.,  1.], [ 1., 1.]]) * kernel_sigma
    # Variables Initialization
    d = I_1_.shape[0]//step_size
    n = I_1_.shape[1]//step_size
    # Get Pyramid Parameters
    #n_pyramid = _max_pyramid_level(n_pyramid, d, n, w)
    level_ = np.linspace(1, n_pyramid, n_pyramid, dtype = int)
    exp_   = np.linspace(0, n_pyramid - 1, n_pyramid, dtype = int)
    scale_ = 1./2**exp_
    U_ = np.zeros((d, n, 2, n_pyramid))
    V_ = np.zeros((d, n, 2, n_pyramid))
    # Define Initial Grid
    x_0_, y_0_ = _interpolation_coordiantes(d, n, d, n)
    # Loop over Pyramid Levels
    for level, scale in zip(level_, scale_):
        # Get Wieghted Windowns
        #W_ = _weighted_window(w, sigma = wls_sigma)
        # Calculate new pyramid image dimension
        m = int(d*scale)//step_size
        k = int(n*scale)//step_size
        if n_pyramid > 1:
            # Scale Coordinates System
            x_level_, y_level_ = _interpolation_coordiantes(d, n, m, k)
            # Rescale the images according to the pyramid level scale
            I_level_1_ = _reduce_image_scale(x_0_, y_0_, x_level_, y_level_, I_1_)
            I_level_2_ = _reduce_image_scale(x_0_, y_0_, x_level_, y_level_, I_2_)
            Q_ = _reduce_image_scale(x_0_, y_0_, x_level_, y_level_, W_[..., 0])
            K_ = _reduce_image_scale(x_0_, y_0_, x_level_, y_level_, W_[..., 1])
            # Get the numerical differencias of the images
            f_x_, f_y_, f_t_ = __compute_numerical_derivatives(I_level_1_, I_level_2_, K_x_, K_y_, K_t_)
            # Solver Optical Flow Equations
            U_level_, V_level_ = __solve_optical_flow(f_x_, f_y_, f_t_, Q_, K_, w, m, k, step_size)
            # Increase the scale of the flow to the first pyramid
            U_[..., level - 1] = _increase_flow_scale(x_0_, y_0_, x_level_, y_level_, U_level_)/scale
            V_[..., level - 1] = _increase_flow_scale(x_0_, y_0_, x_level_, y_level_, V_level_)/scale
        else:
            f_x_, f_y_, f_t_ = __compute_numerical_derivatives(I_1_, I_2_, K_x_, K_y_, K_t_)
            U_[..., level - 1] = __solve_optical_flow(f_x_, f_y_, f_t_, Q_, K_, w, m, k, step_size)
            V_[..., level - 1] = __solve_optical_flow(f_x_, f_y_, f_t_, Q_, K_, w, m, k, step_size)
    U_ = np.mean(U_, axis = -1)
    V_ = np.mean(V_, axis = -1)
    return np.moveaxis(np.stack((U_[..., 0], V_[..., 0])), 0, -1), np.moveaxis(np.stack((U_[..., 1], V_[..., 1])), 0, -1)

def _pyramidal_weighted_lucas_kanade(I_1_, I_2_, W_, window_size, sigma, n_pyramid, tau = 1e-10, step_size = 1):
    """Parameters:
        I_1_ – first 8-bit single-channel input image.
        I_2_ – second input image of the same size and the same type as prev
        window_size – averaging window size; larger values increase the algorithm robustness to image noise
                  and give more chances for fast motion detection, but yield more blurred motion field.
        kernel_sigma - amplitude paramter for the time convolution kernel to adjusts the magnitude of the flow
        n_pyramid - No. of pyramids if pyramidal flow comoputation.
        wls_sigma - Gaussian Neirborhoog Weights standar ddeviation
        tau - Eigenvalues threshold to reduce the amount of noisy vectors on the flow
        step_size - Pixels jump in between windows."""

    # Type of variable definition
    window_size  = int(window_size)
    kernel_sigma = sigma
    n_pyramid = int(n_pyramid)
    tau       = tau
    step_size = int(step_size)
    #print(window_size, tau, kernel_sigma, n_pyramid, wls_sigma)

    # Update pixel coordinates
    def __check_corners(x_, x, w):
        # Cross-Correlation windows upper and lower interval
        x_1_, x_2_ = x_ - w, x_ + w + 1
        # Forze corners to stay on the image frame
        if x_1_ < 0:     x_1_ = 0
        if x_2_ > x - 1: x_2_ = x - 1
        return x_1_, x_2_

    def __sample_partial_derivations(i, j, d, n, w):
        # Adjust Pixel coordinates to have enough pixels on edges windows
        i_1, i_2 = __check_corners(i, d, w)
        j_1, j_2 = __check_corners(j, n, w)
        # Implement Lucas Kanade for each point, calculate I_x, I_y, I_t
        I_x = f_x_[i_1:i_2, j_1:j_2].flatten()[:, np.newaxis]
        I_y = f_y_[i_1:i_2, j_1:j_2].flatten()[:, np.newaxis]
        I_t = f_t_[i_1:i_2, j_1:j_2].flatten()[:, np.newaxis]
        # System Weights
        w_ = Q_[i_1:i_2, j_1:j_2].flatten()[:, np.newaxis]
        # Matrix form Dataset
        X_ = np.concatenate((I_x, I_y), axis = 1)
        y_ = I_t
        return X_, y_, w_

    # Solve Optical FLow for the defined grid
    def __solve_optical_flow(w, d, n, step_size):
        # Weighted Least-Squares Solution
        def ___WLS(X_, y_, w_, tau):
            B_ = np.eye(w_.shape[0]) * w_
            return np.linalg.pinv(X_.T @ B_ @ X_ + np.eye(2)*tau) @ (X_.T @ B_ @ y_)
        # Variables Initialization
        v_x_ = np.zeros((d, n))
        v_y_ = np.zeros((d, n))
        # Loop over pixels rows and coloumns
        for i in range(0, d, step_size):
            for j in range(0, n, step_size):
                # Implement Lucas Kanade for each point, calculate I_x, I_y, I_t
                X_, y_, w_ = __sample_partial_derivations(i, j, d, n, w)
                # Solve weighted-Linear System or linear
                beta_ = ___WLS(X_, y_, w_, tau)
                # get velocity Components
                v_x_[i, j] = beta_[0]
                v_y_[i, j] = beta_[1]
        return np.dstack((v_x_, v_y_))

    # Get the numerical 2D-differenciation of the images
    def __compute_numerical_derivatives(I_1_, I_2_, K_x_, K_y_, K_t_):
        # Numerical derivative convolutions
        f_x_ = convolve2d(I_1_, K_x_, boundary = 'symm', mode = 'same')
        f_y_ = convolve2d(I_1_, K_y_, boundary = 'symm', mode = 'same')
        f_t_ = convolve2d(I_2_, K_t_, boundary = 'symm', mode = 'same') + convolve2d(I_1_, -K_t_, boundary = 'symm', mode = 'same')
        return f_x_, f_y_, f_t_

    # All pixels within [-w, w] are in the window.
    w = int(window_size/2)
    # normalize pixels
    I_1_ = I_1_/255.
    I_2_ = I_2_/255.
    # Numerical derivative Kernels
    K_x_ = np.array([[-1.,  1.], [-1., 1.]])
    K_y_ = np.array([[-1., -1.], [ 1., 1.]])
    K_t_ = np.array([[ 1.,  1.], [ 1., 1.]]) * kernel_sigma
    # Variables Initialization
    d = I_1_.shape[0]//step_size
    n = I_1_.shape[1]//step_size
    # Get Pyramid Parameters
    #n_pyramid = _max_pyramid_level(n_pyramid, d, n, w)
    level_ = np.linspace(1, n_pyramid, n_pyramid, dtype = int)
    exp_   = np.linspace(0, n_pyramid - 1, n_pyramid, dtype = int)
    scale_ = 1./2**exp_
    V_ = np.zeros((d, n, 2, n_pyramid))
    # Define Initial Grid
    x_0_, y_0_ = _interpolation_coordiantes(d, n, d, n)
    # Loop over Pyramid Levels
    for level, scale in zip(level_, scale_):
        # Get Wieghted Windowns
        #W_ = _weighted_window(w, sigma = wls_sigma)
        # Calculate new pyramid image dimension
        m = int(d*scale)//step_size
        k = int(n*scale)//step_size
        if n_pyramid > 1:
            # Scale Coordinates System
            x_level_, y_level_ = _interpolation_coordiantes(d, n, m, k)
            # Rescale the images according to the pyramid level scale
            I_level_1_ = _reduce_image_scale(x_0_, y_0_, x_level_, y_level_, I_1_)
            I_level_2_ = _reduce_image_scale(x_0_, y_0_, x_level_, y_level_, I_2_)
            Q_ = _reduce_image_scale(x_0_, y_0_, x_level_, y_level_, W_)
            # Get the numerical differencias of the images
            f_x_, f_y_, f_t_ = __compute_numerical_derivatives(I_level_1_, I_level_2_, K_x_, K_y_, K_t_)
            # Solver Optical Flow Equations
            V_level_ = __solve_optical_flow(w, m, k, step_size)
            # Increase the scale of the flow to the first pyramid
            V_[..., level - 1] = _increase_flow_scale(x_0_, y_0_, x_level_, y_level_, V_level_)/scale
        else:
            f_x_, f_y_, f_t_ = __compute_numerical_derivatives(I_1_, I_2_, K_x_, K_y_, K_t_)
            V_[..., level - 1] = __solve_optical_flow(w, m, k, step_size)

    return np.mean(V_, axis = -1)

# Lucas-Kanade Solved via Weighted Least-Squares
def _weighted_lucas_kanade(I_1_, I_2_, W_, window_size, sigma, tau = None, step_size = 1, filter_sigma = None):
    """Parameters:
        I_1_ – first 8-bit single-channel input image.
        I_2_ – second input image of the same size and the same type as prev.
        window_size – averaging window size; larger values increase the algorithm robustness to image noise
                  and give more chances for fast motion detection, but yield more blurred motion field.
        tau - Eigenvalues threshold to reduce the amount of noisy vectors on the flow
        sigma - amplitude paramter for the time convolution kernel to adjusts the magnitude of the flow
        step_size - Pixels jump in between windows."""
    # Weighted Least-Squares Solution
    def __WLS(X_, y_, w_, tau):
        B_ = np.eye(w_.shape[0]) * w_
        return np.linalg.pinv(X_.T @ B_ @ X_ + np.eye(2)*tau) @ (X_.T @ B_ @ y_)
    # Optical Flow partial derivatives in i,j neiborhood
    def __sample_partial_derivations(i, j):
        # Update pixel coordinates
        def ___check_corners(x_, x, w):
            # Cross-Correlation windows upper and lower interval
            x_1_, x_2_ = x_ - w, x_ + w + 1
            # Forze corners to stay on the image frame
            if x_1_ < 0:     x_1_ = 0
            if x_2_ > x - 1: x_2_ = x - 1
            return x_1_, x_2_
        # Adjust Pixel coordinates to have enough pixels on edges windows
        i_1, i_2 = ___check_corners(i, d, w)
        j_1, j_2 = ___check_corners(j, n, w)
        # Implement Lucas Kanade for each point, calculate I_x, I_y, I_t
        I_x = f_x_[i_1:i_2, j_1:j_2].flatten()[:, np.newaxis]
        I_y = f_y_[i_1:i_2, j_1:j_2].flatten()[:, np.newaxis]
        I_t = f_t_[i_1:i_2, j_1:j_2].flatten()[:, np.newaxis]
        # Matrix form Dataset
        X_ = np.concatenate((I_x, I_y), axis = 1)
        y_ = I_t
        w_ = W_[i_1:i_2, j_1:j_2].flatten()[:, np.newaxis]
        return X_, y_, w_
    # Type of variable definition
    window_size = int(window_size)
    tau       = tau
    sigma     = sigma
    step_size = int(step_size)
    # All pixels within [-w, w] are in the window.
    w = window_size//2
    d = I_1_.shape[0]//step_size
    n = I_1_.shape[1]//step_size
    # Variables Initialization
    v_x_ = np.zeros((d, n))
    v_y_ = np.zeros((d, n))
    # normalize pixels
    I_1_ = I_1_/255.
    I_2_ = I_2_/255.
    # Numerical derivative Kernels
    k_x = np.array([[-1.,  1.], [-1., 1.]])
    k_y = np.array([[-1., -1.], [ 1., 1.]])
    k_t = np.array([[ 1.,  1.], [ 1., 1.]]) * sigma
    # Numerical derivative convolutions
    f_x_ = convolve2d(I_1_, k_x, boundary = 'symm', mode = 'same')
    f_y_ = convolve2d(I_1_, k_y, boundary = 'symm', mode = 'same')
    f_t_ = convolve2d(I_2_, k_t, boundary = 'symm', mode = 'same') + convolve2d(I_1_, -k_t, boundary = 'symm', mode = 'same')
    # Loop over pixels rows and coloumns
    for i in range(0, d, step_size):
        for j in range(0, n, step_size):
            # Implement Lucas-Kanade for each point, calculate I_x, I_y, I_t
            X_, y_, w_ = __sample_partial_derivations(i, j)
            # Solve Linear System
            v_ = __WLS(X_, y_, w_, tau)
            # get velocity Components
            v_x_[i, j] = v_[0]
            v_y_[i, j] = v_[1]
    return np.dstack((v_x_, v_y_))

def _pyramidal_lucas_kanade(I_1_, I_2_, window_size, sigma, n_pyramid, tau = 1e-10, step_size = 1):
    """Parameters:
        I_1_ – first 8-bit single-channel input image.
        I_2_ – second input image of the same size and the same type as prev
        window_size – averaging window size; larger values increase the algorithm robustness to image noise
                  and give more chances for fast motion detection, but yield more blurred motion field.
        kernel_sigma - amplitude paramter for the time convolution kernel to adjusts the magnitude of the flow
        n_pyramid - No. of pyramids if pyramidal flow comoputation.
        wls_sigma - Gaussian Neirborhoog Weights standar ddeviation
        tau - Eigenvalues threshold to reduce the amount of noisy vectors on the flow
        step_size - Pixels jump in between windows."""

    # Type of variable definition
    window_size  = int(window_size)
    kernel_sigma = sigma
    n_pyramid = int(n_pyramid)
    tau       = tau
    step_size = int(step_size)
    #print(window_size, tau, kernel_sigma, n_pyramid, wls_sigma)

    # Update pixel coordinates
    def __check_corners(x_, x, w):
        # Cross-Correlation windows upper and lower interval
        x_1_, x_2_ = x_ - w, x_ + w + 1
        # Forze corners to stay on the image frame
        if x_1_ < 0:     x_1_ = 0
        if x_2_ > x - 1: x_2_ = x - 1
        return x_1_, x_2_

    def __sample_partial_derivations(i, j, d, n, w):
        # Adjust Pixel coordinates to have enough pixels on edges windows
        i_1, i_2 = __check_corners(i, d, w)
        j_1, j_2 = __check_corners(j, n, w)
        # Implement Lucas Kanade for each point, calculate I_x, I_y, I_t
        I_x = f_x_[i_1:i_2, j_1:j_2].flatten()[:, np.newaxis]
        I_y = f_y_[i_1:i_2, j_1:j_2].flatten()[:, np.newaxis]
        I_t = f_t_[i_1:i_2, j_1:j_2].flatten()[:, np.newaxis]
        # Matrix form Dataset
        X_ = np.concatenate((I_x, I_y), axis = 1)
        y_ = I_t
        return X_, y_

    # Solve Optical FLow for the defined grid
    def __solve_optical_flow(w, d, n, step_size):
        # Weighted Least-Squares Solution
        def ___LS(X_, y_, tau):
            return np.linalg.pinv(X_.T @ X_ + np.eye(2)*tau) @ (X_.T @ y_)
        # Variables Initialization
        v_x_ = np.zeros((d, n))
        v_y_ = np.zeros((d, n))
        # Loop over pixels rows and coloumns
        for i in range(0, d, step_size):
            for j in range(0, n, step_size):
                # Implement Lucas Kanade for each point, calculate I_x, I_y, I_t
                X_, y_ = __sample_partial_derivations(i, j, d, n, w)
                # Solve weighted-Linear System or linear
                beta_ = ___LS(X_, y_, tau)
                # get velocity Components
                v_x_[i, j] = beta_[0]
                v_y_[i, j] = beta_[1]
        return np.dstack((v_x_, v_y_))

    # Get the numerical 2D-differenciation of the images
    def __compute_numerical_derivatives(I_1_, I_2_, K_x_, K_y_, K_t_):
        # Numerical derivative convolutions
        f_x_ = convolve2d(I_1_, K_x_, boundary = 'symm', mode = 'same')
        f_y_ = convolve2d(I_1_, K_y_, boundary = 'symm', mode = 'same')
        f_t_ = convolve2d(I_2_, K_t_, boundary = 'symm', mode = 'same') + convolve2d(I_1_, -K_t_, boundary = 'symm', mode = 'same')
        return f_x_, f_y_, f_t_

    # All pixels within [-w, w] are in the window.
    w = int(window_size/2)
    # normalize pixels
    I_1_ = I_1_/255.
    I_2_ = I_2_/255.
    # Numerical derivative Kernels
    K_x_ = np.array([[-1.,  1.], [-1., 1.]])
    K_y_ = np.array([[-1., -1.], [ 1., 1.]])
    K_t_ = np.array([[ 1.,  1.], [ 1., 1.]]) * kernel_sigma
    # Variables Initialization
    d = I_1_.shape[0]//step_size
    n = I_1_.shape[1]//step_size
    # Get Pyramid Parameters
    #n_pyramid = _max_pyramid_level(n_pyramid, d, n, w)
    level_ = np.linspace(1, n_pyramid, n_pyramid, dtype = int)
    exp_   = np.linspace(0, n_pyramid - 1, n_pyramid, dtype = int)
    scale_ = 1./2**exp_
    V_ = np.zeros((d, n, 2, n_pyramid))
    # Define Initial Grid
    x_0_, y_0_ = _interpolation_coordiantes(d, n, d, n)
    # Loop over Pyramid Levels
    for level, scale in zip(level_, scale_):
        # Get Wieghted Windowns
        #W_ = _weighted_window(w, sigma = wls_sigma)
        # Calculate new pyramid image dimension
        m = int(d*scale)//step_size
        k = int(n*scale)//step_size
        if n_pyramid > 1:
            # Scale Coordinates System
            x_level_, y_level_ = _interpolation_coordiantes(d, n, m, k)
            # Rescale the images according to the pyramid level scale
            I_level_1_ = _reduce_image_scale(x_0_, y_0_, x_level_, y_level_, I_1_)
            I_level_2_ = _reduce_image_scale(x_0_, y_0_, x_level_, y_level_, I_2_)
            # Get the numerical differencias of the images
            f_x_, f_y_, f_t_ = __compute_numerical_derivatives(I_level_1_, I_level_2_, K_x_, K_y_, K_t_)
            # Solver Optical Flow Equations
            V_level_ = __solve_optical_flow(w, m, k, step_size)
            # Increase the scale of the flow to the first pyramid
            V_[..., level - 1] = _increase_flow_scale(x_0_, y_0_, x_level_, y_level_, V_level_)/scale
        else:
            f_x_, f_y_, f_t_ = __compute_numerical_derivatives(I_1_, I_2_, K_x_, K_y_, K_t_)
            V_[..., level - 1] = __solve_optical_flow(w, m, k, step_size)

    return np.mean(V_, axis = -1)


def _pyramidal_weighted_lucas_kanade(I_1_, I_2_, W_, window_size, sigma, n_pyramid, tau = 1e-10, step_size = 1):
    """Parameters:
        I_1_ – first 8-bit single-channel input image.
        I_2_ – second input image of the same size and the same type as prev
        window_size – averaging window size; larger values increase the algorithm robustness to image noise
                  and give more chances for fast motion detection, but yield more blurred motion field.
        kernel_sigma - amplitude paramter for the time convolution kernel to adjusts the magnitude of the flow
        n_pyramid - No. of pyramids if pyramidal flow comoputation.
        wls_sigma - Gaussian Neirborhoog Weights standar ddeviation
        tau - Eigenvalues threshold to reduce the amount of noisy vectors on the flow
        step_size - Pixels jump in between windows."""

    # Type of variable definition
    window_size  = int(window_size)
    kernel_sigma = sigma
    n_pyramid = int(n_pyramid)
    tau       = tau
    step_size = int(step_size)
    #print(window_size, tau, kernel_sigma, n_pyramid, wls_sigma)

    # Update pixel coordinates
    def __check_corners(x_, x, w):
        # Cross-Correlation windows upper and lower interval
        x_1_, x_2_ = x_ - w, x_ + w + 1
        # Forze corners to stay on the image frame
        if x_1_ < 0:     x_1_ = 0
        if x_2_ > x - 1: x_2_ = x - 1
        return x_1_, x_2_

    def __sample_partial_derivations(i, j, d, n, w):
        # Adjust Pixel coordinates to have enough pixels on edges windows
        i_1, i_2 = __check_corners(i, d, w)
        j_1, j_2 = __check_corners(j, n, w)
        # Implement Lucas Kanade for each point, calculate I_x, I_y, I_t
        I_x = f_x_[i_1:i_2, j_1:j_2].flatten()[:, np.newaxis]
        I_y = f_y_[i_1:i_2, j_1:j_2].flatten()[:, np.newaxis]
        I_t = f_t_[i_1:i_2, j_1:j_2].flatten()[:, np.newaxis]
        # System Weights
        w_ = Q_[i_1:i_2, j_1:j_2].flatten()[:, np.newaxis]
        # Matrix form Dataset
        X_ = np.concatenate((I_x, I_y), axis = 1)
        y_ = I_t
        return X_, y_, w_

    # Solve Optical FLow for the defined grid
    def __solve_optical_flow(w, d, n, step_size):
        # Weighted Least-Squares Solution
        def ___WLS(X_, y_, w_, tau):
            B_ = np.eye(w_.shape[0]) * w_
            return np.linalg.pinv(X_.T @ B_ @ X_ + np.eye(2)*tau) @ (X_.T @ B_ @ y_)
        # Variables Initialization
        v_x_ = np.zeros((d, n))
        v_y_ = np.zeros((d, n))
        # Loop over pixels rows and coloumns
        for i in range(0, d, step_size):
            for j in range(0, n, step_size):
                # Implement Lucas Kanade for each point, calculate I_x, I_y, I_t
                X_, y_, w_ = __sample_partial_derivations(i, j, d, n, w)
                # Solve weighted-Linear System or linear
                beta_ = ___WLS(X_, y_, w_, tau)
                # get velocity Components
                v_x_[i, j] = beta_[0]
                v_y_[i, j] = beta_[1]
        return np.dstack((v_x_, v_y_))

    # Get the numerical 2D-differenciation of the images
    def __compute_numerical_derivatives(I_1_, I_2_, K_x_, K_y_, K_t_):
        # Numerical derivative convolutions
        f_x_ = convolve2d(I_1_, K_x_, boundary = 'symm', mode = 'same')
        f_y_ = convolve2d(I_1_, K_y_, boundary = 'symm', mode = 'same')
        f_t_ = convolve2d(I_2_, K_t_, boundary = 'symm', mode = 'same') + convolve2d(I_1_, -K_t_, boundary = 'symm', mode = 'same')
        return f_x_, f_y_, f_t_

    # All pixels within [-w, w] are in the window.
    w = int(window_size/2)
    # normalize pixels
    I_1_ = I_1_/255.
    I_2_ = I_2_/255.
    # Numerical derivative Kernels
    K_x_ = np.array([[-1.,  1.], [-1., 1.]])
    K_y_ = np.array([[-1., -1.], [ 1., 1.]])
    K_t_ = np.array([[ 1.,  1.], [ 1., 1.]]) * kernel_sigma
    # Variables Initialization
    d = I_1_.shape[0]//step_size
    n = I_1_.shape[1]//step_size
    # Get Pyramid Parameters
    #n_pyramid = _max_pyramid_level(n_pyramid, d, n, w)
    level_ = np.linspace(1, n_pyramid, n_pyramid, dtype = int)
    exp_   = np.linspace(0, n_pyramid - 1, n_pyramid, dtype = int)
    scale_ = 1./2**exp_
    V_ = np.zeros((d, n, 2, n_pyramid))
    # Define Initial Grid
    x_0_, y_0_ = _interpolation_coordiantes(d, n, d, n)
    # Loop over Pyramid Levels
    for level, scale in zip(level_, scale_):
        # Get Wieghted Windowns
        #W_ = _weighted_window(w, sigma = wls_sigma)
        # Calculate new pyramid image dimension
        m = int(d*scale)//step_size
        k = int(n*scale)//step_size
        if n_pyramid > 1:
            # Scale Coordinates System
            x_level_, y_level_ = _interpolation_coordiantes(d, n, m, k)
            # Rescale the images according to the pyramid level scale
            I_level_1_ = _reduce_image_scale(x_0_, y_0_, x_level_, y_level_, I_1_)
            I_level_2_ = _reduce_image_scale(x_0_, y_0_, x_level_, y_level_, I_2_)
            Q_ = _reduce_image_scale(x_0_, y_0_, x_level_, y_level_, W_)
            # Get the numerical differencias of the images
            f_x_, f_y_, f_t_ = __compute_numerical_derivatives(I_level_1_, I_level_2_, K_x_, K_y_, K_t_)
            # Solver Optical Flow Equations
            V_level_ = __solve_optical_flow(w, m, k, step_size)
            # Increase the scale of the flow to the first pyramid
            V_[..., level - 1] = _increase_flow_scale(x_0_, y_0_, x_level_, y_level_, V_level_)/scale
        else:
            f_x_, f_y_, f_t_ = __compute_numerical_derivatives(I_1_, I_2_, K_x_, K_y_, K_t_)
            Q_ = W_.copy()
            V_[..., level - 1] = __solve_optical_flow(w, m, k, step_size)

    return np.mean(V_, axis = -1)

# Lucas-Kanade Solved via Weighted Least-Squares
def _weighted_lucas_kanade(I_1_, I_2_, W_, window_size, sigma, tau = None, step_size = 1, filter_sigma = None):
    """Parameters:
        I_1_ – first 8-bit single-channel input image.
        I_2_ – second input image of the same size and the same type as prev.
        window_size – averaging window size; larger values increase the algorithm robustness to image noise
                  and give more chances for fast motion detection, but yield more blurred motion field.
        tau - Eigenvalues threshold to reduce the amount of noisy vectors on the flow
        sigma - amplitude paramter for the time convolution kernel to adjusts the magnitude of the flow
        step_size - Pixels jump in between windows."""
    # Weighted Least-Squares Solution
    def __WLS(X_, y_, w_, tau):
        B_ = np.eye(w_.shape[0]) * w_
        return np.linalg.pinv(X_.T @ B_ @ X_ + np.eye(2)*tau) @ (X_.T @ B_ @ y_)
    # Optical Flow partial derivatives in i,j neiborhood
    def __sample_partial_derivations(i, j):
        # Update pixel coordinates
        def ___check_corners(x_, x, w):
            # Cross-Correlation windows upper and lower interval
            x_1_, x_2_ = x_ - w, x_ + w + 1
            # Forze corners to stay on the image frame
            if x_1_ < 0:     x_1_ = 0
            if x_2_ > x - 1: x_2_ = x - 1
            return x_1_, x_2_
        # Adjust Pixel coordinates to have enough pixels on edges windows
        i_1, i_2 = ___check_corners(i, d, w)
        j_1, j_2 = ___check_corners(j, n, w)
        # Implement Lucas Kanade for each point, calculate I_x, I_y, I_t
        I_x = f_x_[i_1:i_2, j_1:j_2].flatten()[:, np.newaxis]
        I_y = f_y_[i_1:i_2, j_1:j_2].flatten()[:, np.newaxis]
        I_t = f_t_[i_1:i_2, j_1:j_2].flatten()[:, np.newaxis]
        # Matrix form Dataset
        X_ = np.concatenate((I_x, I_y), axis = 1)
        y_ = I_t
        w_ = W_[i_1:i_2, j_1:j_2].flatten()[:, np.newaxis]
        return X_, y_, w_
    # Type of variable definition
    window_size = int(window_size)
    tau       = tau
    sigma     = sigma
    step_size = int(step_size)
    # All pixels within [-w, w] are in the window.
    w = window_size//2
    d = I_1_.shape[0]//step_size
    n = I_1_.shape[1]//step_size
    # Variables Initialization
    v_x_ = np.zeros((d, n))
    v_y_ = np.zeros((d, n))
    # normalize pixels
    I_1_ = I_1_/255.
    I_2_ = I_2_/255.
    # Numerical derivative Kernels
    k_x = np.array([[-1.,  1.], [-1., 1.]])
    k_y = np.array([[-1., -1.], [ 1., 1.]])
    k_t = np.array([[ 1.,  1.], [ 1., 1.]]) * sigma
    # Numerical derivative convolutions
    f_x_ = convolve2d(I_1_, k_x, boundary = 'symm', mode = 'same')
    f_y_ = convolve2d(I_1_, k_y, boundary = 'symm', mode = 'same')
    f_t_ = convolve2d(I_2_, k_t, boundary = 'symm', mode = 'same') + convolve2d(I_1_, -k_t, boundary = 'symm', mode = 'same')
    # Loop over pixels rows and coloumns
    for i in range(0, d, step_size):
        for j in range(0, n, step_size):
            # Implement Lucas-Kanade for each point, calculate I_x, I_y, I_t
            X_, y_, w_ = __sample_partial_derivations(i, j)
            # Solve Linear System
            v_ = __WLS(X_, y_, w_, tau)
            # get velocity Components
            v_x_[i, j] = v_[0]
            v_y_[i, j] = v_[1]
    return np.dstack((v_x_, v_y_))



def _LS_SVM_MV(I_1_, I_2_, window_size = 5, sigma = .25, step_size = 1, n_init = 25, GS = False):
    """Parameters:
        I_1_ – first 8-bit single-channel input image.
        I_2_ – second input image of the same size and the same type as prev.
        window_size – averaging window size; larger values increase the algorithm robustness to image noise
                  and give more chances for fast motion detection, but yield more blurred motion field.
        sigma - amplitude paramter for the time convolution kernel to adjusts the magnitude of the flow
        n_init – Number of random selection of pixels windows to cross-validate the value of C.
        GS - boolean variable. In case of Grid-Search intesive Cross-Validation
        step_size - Pixels jump in between windows.
        """
    # Solver SVM as a linear system
    def __LS_SVM(X_, y_, c):
        # Variable Initialization
        A = X_.T
        o_ = np.ones((A.shape[1], 1))
        # Compute A Matrix of Ax = b
        a_00_ = np.matmul(A, A.T)*c + np.eye(A.shape[0])
        a_01_ = np.matmul(A, o_)*c
        a_10_ = np.matmul(o_.T, A.T)
        a_11_ = np.matmul(o_.T, o_)
        A_    =  np.concatenate((np.concatenate((a_00_, a_01_), axis = 1), np.concatenate((a_10_, a_11_), axis = 1)), axis = 0)
        # Compute b vector from Ax = b
        b_0_ = np.matmul(A, y_)*c
        b_1_ = np.matmul(o_.T, y_)
        b_ = np.concatenate((b_0_, b_1_), axis = 0)
        # Computer x from x = b A^-1
        x_ = np.matmul(np.linalg.pinv(A_), b_)
        #eig = np.linalg.eigvals(A_)
        return x_[-1:, :], x_[:2, :]

    # Make a prediction
    def __prediction(x_, w_, b):
        return np.matmul(x_, w_) + b

    # Solver LS-SVM, and evaluate Error
    def __solve_LS_SVM(X_, y_, c):
        b, w_  = __LS_SVM(X_, y_, c)
        y_hat_ = __prediction(X_, w_, b)
        e = np.mean(np.sqrt((y_hat_ - y_)**2))
        return w_, e, c

    # Grid-Search Optimization of the regulariazation term
    def __grid_search(X_, y_, C, K):
        # K-fold Cross-Validation Variables
        N    = X_.shape[0]
        idx_ = np.random.permutation(N)
        n_samples_fold = N//K
        # Variables Initialization
        c_ = np.logspace(-10, 10, C) # Complexity
        e_ = np.zeros((C))
        # Loop over Complexity values
        for c, i in zip(c_, range(c_.shape[0])):
            # Save SVM K-fold Accurary
            e_[i] = __cross_validation(c, X_, y_, K, n_samples_fold, idx_)
        # Validation Results and Model Fit
        i = np.where(e_ == np.min(e_))[0]
        return __solve_LS_SVM(X_, y_, c = c_[i][0])

    # k-Fold Cross-Validation
    def __cross_validation(c, X_, y_, K, n_samples_fold, idx_):
        e_ = np.zeros((K))
        # Loop over K-Fold Cross-Calidation
        for k in range(K):
            # Cross-Validation Index
            idx_val_ = idx_[k*n_samples_fold:(k + 1)*n_samples_fold]
            idx_tr_  = np.setxor1d(idx_, idx_[k*n_samples_fold:(k + 1)*n_samples_fold])
            # LS-SVM Fit
            e_[k] = __solve_LS_SVM(X_[idx_tr_, :], y_[idx_tr_, :], c)[1]
        return np.mean(e_)

    # Update pixel coordinates
    def __check_corners(x_, x, w):
        # Cross-Correlation windows upper and lower interval
        x_1_, x_2_ = x_ - w, x_ + w + 1
        # Forze corners to stay on the image frame
        if x_1_ < 0:     x_1_ = 0
        if x_2_ > x - 1: x_2_ = x - 1
        return x_1_, x_2_

    def __random_sample_cross_validation(d, n, n_init = 5):
        c_ = 0
        for _ in range(n_init):
            i = np.random.randint(0, d)
            j = np.random.randint(0, n)
            X_, y_ = __sample_partial_derivations(i, j)
            _, _, c = __grid_search(X_, y_, C = 100, K = 5)
            c_ += c
        return c_/n_init

    def __sample_partial_derivations(i, j):
        # Adjust Pixel coordinates to have enough pixels on edges windows
        i_1, i_2 = __check_corners(i, d, w)
        j_1, j_2 = __check_corners(j, n, w)
        # Implement Lucas Kanade for each point, calculate I_x, I_y, I_t
        I_x = f_x[i_1:i_2, j_1:j_2].flatten()[:, np.newaxis]
        I_y = f_y[i_1:i_2, j_1:j_2].flatten()[:, np.newaxis]
        I_t = f_t[i_1:i_2, j_1:j_2].flatten()[:, np.newaxis]
        # Matrix form Dataset
        X_ = np.concatenate((I_x, I_y), axis = 1)
        y_ = I_t
        return X_, y_

    # Type of variable definition
    window_size = int(window_size)
    sigma       = sigma
    step_size   = int(step_size)
    # All pixels within [-w, w] are in the window.
    w = window_size//2
    d = I_1_.shape[0]//step_size
    n = I_1_.shape[1]//step_size
    # normalize pixels
    I_1_ = I_1_/255.
    I_2_ = I_2_/255.
    # normalize pixels
    v_x_ = np.zeros((d, n))
    v_y_ = np.zeros((d, n))
    # Numerical derivative Kernels
    k_x = np.array([[-1.,  1.], [-1., 1.]])
    k_y = np.array([[-1., -1.], [ 1., 1.]])
    k_t = np.array([[ 1.,  1.], [ 1., 1.]]) * sigma
    # Numerical derivative convolutions
    f_x = convolve2d(I_1_, k_x, boundary = 'symm', mode = 'same')
    f_y = convolve2d(I_1_, k_y, boundary = 'symm', mode = 'same')
    f_t = convolve2d(I_2_, k_t, boundary = 'symm', mode = 'same') + convolve2d(I_1_, -k_t, boundary = 'symm', mode = 'same')
    # Random Grid Search Cross-Validation
    if not GS: c = __random_sample_cross_validation(d, n, n_init)
    e_ = 0.
    # Loop over pixels rows and coloumns
    for i in range(0, d, step_size):
        for j in range(0, n, step_size):
            X_, y_ = __sample_partial_derivations(i, j)
            # Solve LS-SVM System
            if GS: w_, e, _ = __grid_search(X_, y_, C = 100, K = 5)
            else:  w_, e, _ = __solve_LS_SVM(X_, y_, c)
            # Get Velocity Components
            v_x_[i, j] = w_[0]
            v_y_[i, j] = w_[1]
    return np.dstack((v_x_, v_y_))


def _lucas_kanade(I_1_, I_2_, window_size, sigma, tau = None, step_size = 1, filter_sigma = None, filter_size = None):
    """Parameters:
        I_1_ – first 8-bit single-channel input image.
        I_2_ – second input image of the same size and the same type as prev.
        window_size – averaging window size; larger values increase the algorithm robustness to image noise
                  and give more chances for fast motion detection, but yield more blurred motion field.
        tau - Eigenvalues threshold to reduce the amount of noisy vectors on the flow
        sigma - amplitude paramter for the time convolution kernel to adjusts the magnitude of the flow
        step_size - Pixels jump in between windows."""

    # Type of variable definition
    window_size = int(window_size)
    tau         = tau
    sigma       = sigma
    step_size   = int(step_size)
    #print(window_size, tau, sigma, step_size)

    def __LS(X_, y_, tau):
        def ___error(w_, X_, y_):
            y_hat = np.matmul(X_, w_)
            return np.square(np.mean((y_ - y_hat)**2))
        A_ = np.matmul(X_.T, X_) + np.eye(2)*tau
        w_ = np.matmul(np.linalg.pinv(A_), np.matmul(X_.T, y_))
        e_ = ___error(w_, X_, y_)
        return w_, e_

   # Grid-Search Optimization of the regulariazation term
    def __grid_search(X_, y_, C, K):
        # k-Fold Cross-Validation
        def ___cross_validation(c, X_, y_, K, n_samples_fold, idx_):
            e_ = np.zeros((K))
            # Loop over K-Fold Cross-Calidation
            for k in range(K):
                # Cross-Validation Index
                idx_val_ = idx_[k*n_samples_fold:(k + 1)*n_samples_fold]
                idx_tr_  = np.setxor1d(idx_, idx_[k*n_samples_fold:(k + 1)*n_samples_fold])
                # LS-SVM Fit
                e_[k] = __LS(X_[idx_tr_, :], y_[idx_tr_, :], c)[1]
            return np.mean(e_)
        # K-fold Cross-Validation Variables
        N    = X_.shape[0]
        idx_ = np.random.permutation(N)
        n_samples_fold = N//K
        # Variables Initialization
        c_ = np.logspace(-20, 20, C) # Complexity
        e_ = np.zeros((C))
        # Loop over Complexity values
        for c, i in zip(c_, range(c_.shape[0])):
            # Save SVM K-fold Accurary
            try: e_[i] = ___cross_validation(c, X_, y_, K, n_samples_fold, idx_)
            except: e_[i] = np.inf
        # Validation Results and Model Fit
        i = np.where(e_ == np.min(e_))[0]
        w_, e_ = __LS(X_, y_, c_[i][0])
        return w_, e_, c_[i][0]

    # Update pixel coordinates
    def __check_corners(x_, x, w):
        # Cross-Correlation windows upper and lower interval
        x_1_, x_2_ = x_ - w, x_ + w + 1
        # Forze corners to stay on the image frame
        if x_1_ < 0:     x_1_ = 0
        if x_2_ > x - 1: x_2_ = x - 1
        return x_1_, x_2_

    def __random_sample_cross_validation(d, n, n_init = 5):
        c_ = 0
        for _ in range(n_init):
            i = np.random.randint(0, d)
            j = np.random.randint(0, n)
            X_, y_ = __sample_partial_derivations(i, j)
            _, _, c = __grid_search(X_, y_, C = 100, K = 5)
            c_ += c
        return c_/n_init

    # Update pixel coordinates
    def __check_corners(x_, x, w):
        # Cross-Correlation windows upper and lower interval
        x_1_, x_2_ = x_ - w, x_ + w + 1
        # Forze corners to stay on the image frame
        if x_1_ < 0:     x_1_ = 0
        if x_2_ > x - 1: x_2_ = x - 1
        return x_1_, x_2_

    def __sample_partial_derivations(i, j):
        # Adjust Pixel coordinates to have enough pixels on edges windows
        i_1, i_2 = __check_corners(i, d, w)
        j_1, j_2 = __check_corners(j, n, w)
        # Implement Lucas Kanade for each point, calculate I_x, I_y, I_t
        I_x = f_x[i_1:i_2, j_1:j_2].flatten()[:, np.newaxis]
        I_y = f_y[i_1:i_2, j_1:j_2].flatten()[:, np.newaxis]
        I_t = f_t[i_1:i_2, j_1:j_2].flatten()[:, np.newaxis]
        # Matrix form Dataset
        X_ = np.concatenate((I_x, I_y), axis = 1)
        y_ = I_t
        return X_, y_

    # All pixels within [-w, w] are in the window.
    w = window_size//2
    d = I_1_.shape[0]//step_size
    n = I_1_.shape[1]//step_size
    # Variables Initialization
    v_x_ = np.zeros((d, n))
    v_y_ = np.zeros((d, n))
    # normalize pixels
    I_1_ = I_1_/255.
    I_2_ = I_2_/255.
    # Numerical derivative Kernels
    k_x = np.array([[-1.,  1.], [-1., 1.]])
    k_y = np.array([[-1., -1.], [ 1., 1.]])
    k_t = np.array([[ 1.,  1.], [ 1., 1.]]) * sigma
    # Numerical derivative convolutions
    f_x = convolve2d(I_1_, k_x, boundary = 'symm', mode = 'same')
    f_y = convolve2d(I_1_, k_y, boundary = 'symm', mode = 'same')
    f_t = convolve2d(I_2_, k_t, boundary = 'symm', mode = 'same') + convolve2d(I_1_, -k_t, boundary = 'symm', mode = 'same')
    if tau is None: tau = __random_sample_cross_validation(d, n, n_init = 5)
    e_ = 0.
    # Loop over pixels rows and coloumns
    for i in range(0, d, step_size):
        for j in range(0, n, step_size):
            # Implement Lucas Kanade for each point, calculate I_x, I_y, I_t
            X_, y_ = __sample_partial_derivations(i, j)
            # Solve Linear System
            w_, e = __LS(X_, y_, tau)
            # get velocity Components
            v_x_[i, j] = w_[0]
            v_y_[i, j] = w_[1]
            e_ += e

    if filter_sigma is not None:
        v_x_ = gaussian_filter(v_x_, sigma = filter_sigma, mode = 'reflect')
        v_y_ = gaussian_filter(v_y_, sigma = filter_sigma, mode = 'reflect')
    if filter_size is not None:
        v_x_ = uniform_filter(v_x_, size = filter_size)
        v_y_ = uniform_filter(v_y_, size = filter_size)

    return np.dstack((v_x_, v_y_))


def _farneback(I_prev_, I_next_, pyr_scale, n_levels, window_size, n_iter, n_poly, sigma, flow):
    """Parameters:
        prev – first 8-bit single-channel input image.
        next – second input image of the same size and the same type as prev.
        flow – computed flow image that has the same size as prev and type CV_32FC2.
        pyr_scale – parameter, specifying the image scale (<1) to build pyramids for each image;
                    pyr_scale=0.5 means a classical pyramid, where each next layer is twice smaller than the previous one.
        levels – number of pyramid layers including the initial image;
                 levels=1 means that no extra layers are created and only the original images are used.
        winsize – averaging window size; larger values increase the algorithm robustness to image noise
                  and give more chances for fast motion detection, but yield more blurred motion field.
        iterations – number of iterations the algorithm does at each pyramid level.
        poly_n – size of the pixel neighborhood used to find polynomial expansion in each pixel;
                 larger values mean that the image will be approximated with smoother surfaces,
                 yielding more robust algorithm and more blurred motion field, typically poly_n =5 or 7.
        poly_sigma – standard deviation of the Gaussian that is used to smooth derivatives used as a basis for the
                     polynomial expansion; for poly_n=5, you can set poly_sigma=1.1,
                     for poly_n=7, a good value would be poly_sigma=1.5.
        flags - Not used... """

    # Type of variable definition
    flow        = flow
    pyr_scale   = pyr_scale
    n_levels    = int(n_levels)
    window_size = int(window_size)
    n_iter      = int(n_iter)
    n_poly      = int(n_poly)
    sigma       = sigma
    #print(pyr_scale, n_levels, window_size, n_iter, n_poly, sigma)

    # Implementationf from OpenCV library (C++)
    if flow is None:
        return calcOpticalFlowFarneback(I_prev_, I_next_, flow, pyr_scale, n_levels, window_size, n_iter, n_poly, sigma, 0)
    else:
        return calcOpticalFlowFarneback(I_prev_, I_next_, flow, pyr_scale, n_levels, window_size, n_iter, n_poly, sigma, OPTFLOW_USE_INITIAL_FLOW)

def _pyramidal_weighted_horn_schunck(I_1_, I_2_, window_size, n_pyramid, alpha = 1e-5, kernel_sigma = 0.25, whs_sigma = 1., tol = 1e-3, max_iter = 100):
    """Parameters:
        I_1_ – first 8-bit single-channel input image.
        I_2_ – second input image of the same size and the same type as prev.
        window_size – averaging window size; larger values increase the algorithm robustness to image noise
                  and give more chances for fast motion detection, but yield more blurred motion field.
        n_pyramid - No. of pyramids if pyramidal flow comoputation.
        alpha - magnitude parameter to adjust the updateing increments magnitude during the optimization.
        kernel_sigma - amplitude paramter for the time convolution kernel to adjusts the magnitude of the flow
        wls_sigma - Gaussian Neirborhoog Weights standar ddeviation
        tol - tolerance allow on the updating increments as stopping creteria,
              so that is decided when the increment on the optimization achive is too small to continue.
        max_iter - Maximum number of iteratios for the recursive optimization algorithm."""

    # Type of variable definition
    window_size  = int(window_size)
    alpha        = alpha
    kernel_sigma = kernel_sigma
    n_pyramid    = int(n_pyramid)
    whs_sigma    = whs_sigma
    tol          = tol
    max_iter     = int(max_iter)
    #print(window_size, alpha, kernel_sigma, n_pyramid, wls_sigma, tol, max_iter)

    # Solver Constrained Optical Flow Equation via iterative numerical optimization
    def __solve_optical_flow(f_x, f_y, f_t, K_hs_, d, n):
        # Variables Initializaiton
        v_x_ = np.zeros((d, n))
        v_y_ = np.zeros((d, n))
        d_0_ = 0.
        # Loop Iterative Optimization
        for i in range(max_iter):
            # Horn Schunk average window via 2d-convolution step
            bar_v_x_ = convolve2d(v_x_, K_hs_, boundary = 'symm', mode = 'same')
            bar_v_y_ = convolve2d(v_y_, K_hs_, boundary = 'symm', mode = 'same')
            # Iterative incremental numerical optimization
            d_1_ = (f_x * bar_v_x_ + f_y * bar_v_y_ + f_t) / (alpha**2 + f_x**2 + f_y**2)
            v_x_ = bar_v_x_ - f_x * d_1_
            v_y_ = bar_v_y_ - f_y * d_1_
            # Stop optimization when algorithm converges
            if abs(d_1_.sum() - d_0_) < tol:
                break
            else:
                d_0_ = d_1_.sum()
        return np.dstack((v_x_, v_y_))

    # Get the numerical 2D-differenciation of the images
    def __compute_numerical_derivatives(I_1_, I_2_, K_x_, K_y_, K_t_):
        # Convolution numerical derivative
        f_x_ = convolve2d(I_1_, K_x_, boundary = 'symm', mode = 'same') + convolve2d(I_2_,  K_x_, boundary = 'symm', mode = 'same')
        f_y_ = convolve2d(I_1_, K_y_, boundary = 'symm', mode = 'same') + convolve2d(I_2_,  K_y_, boundary = 'symm', mode = 'same')
        f_t_ = convolve2d(I_1_, K_t_, boundary = 'symm', mode = 'same') + convolve2d(I_2_, -K_t_, boundary = 'symm', mode = 'same')
        return f_x_, f_y_, f_t_

    # All pixels within [-w, w] are in the window.
    w = window_size//2
    # Normalize pixels
    I_1_ = I_1_/255.
    I_2_ = I_2_/255.
    # Get Wieghted Windowns
    #K_hs_ = _weighted_window(w, sigma = whs_sigma)
    # Horn Schunk Convolution Kernel
    K_hs_ = np.array([[1./12., 1./6., 1./12.],
                      [1./6. ,    0., 1./6. ],
                      [1./12., 1./6., 1./12.]])
    # Implement Lucas Kanade for each point, calculate I_x, I_y, I_t
    # Numerial Derivatives Kernel
    K_x_ = np.array([[-1., 1.], [-1., 1.]]) * kernel_sigma
    K_y_ = np.array([[-1.,-1.], [ 1., 1.]]) * kernel_sigma
    K_t_ = np.array([[ 1., 1.], [ 1., 1.]]) * kernel_sigma
    # Variables Initialization
    d = I_1_.shape[0]
    n = I_1_.shape[1]
    # Get Pyramid Parameters
    n_pyramid = _max_pyramid_level(n_pyramid, d, n, w)
    level_ = np.linspace(1, n_pyramid, n_pyramid, dtype = int)
    exp_   = np.linspace(0, n_pyramid - 1, n_pyramid, dtype = int)
    scale_ = 1./2**exp_
    # Flow Variable Initialization
    V_ = np.zeros((d, n, 2, n_pyramid))
    # Define Initial Grid
    x_0_, y_0_ = _interpolation_coordiantes(d, n, d, n)
    # Loop over Pyramid Levels
    for level, scale in zip(level_, scale_):
        # Calculate new pyramid image dimension
        m = int(d*scale)
        k = int(n*scale)
        if level > 1:
            # Scale Coordinates System
            x_level_, y_level_ = _interpolation_coordiantes(d, n, m, k)
            # Rescale the images according to the pyramid level scale
            I_level_1_ = _reduce_image_scale(x_0_, y_0_, x_level_, y_level_, I_1_)
            I_level_2_ = _reduce_image_scale(x_0_, y_0_, x_level_, y_level_, I_2_)
            # Get the numerical differencias of the images
            f_x_, f_y_, f_t_ = __compute_numerical_derivatives(I_level_1_, I_level_2_, K_x_, K_y_, K_t_)
            # Solver Optical Flow Equations
            V_level_ = __solve_optical_flow(f_x_, f_y_, f_t_, K_hs_, m, k)
            # Increase the scale of the flow to the first pyramid
            V_[..., level - 1] = _increase_flow_scale(x_0_, y_0_, x_level_, y_level_, V_level_)
        else:
            f_x_, f_y_, f_t_ = __compute_numerical_derivatives(I_1_, I_2_, K_x_, K_y_, K_t_)
            V_[..., level - 1] = __solve_optical_flow(f_x_, f_y_, f_t_, K_hs_, m, k)
    return np.mean(V_, axis = -1)

def _horn_schunck(I_1_, I_2_, tol = 1e-3, alpha = 1e-5, max_iter = 5, sigma = 0.25):
    """Parameters:
        I_1_ – first 8-bit single-channel input image.
        I_2_ – second input image of the same size and the same type as prev.
        tol - tolerance allow on the updating increments as stopping creteria,
              so that is decided when the increment on the optimization achive is too small to continue.
        alpha - magnitude parameter to adjust the updateing increments magnitude during the optimization.
        max_iter - Maximum number of iteratios for the recursive optimization algorithm.
        sigma - amplitude paramter for the time convolution kernel to adjusts the magnitude of the flow."""

    # Type of variable definition
    tol      = tol
    alpha    = alpha
    max_iter = int(max_iter)
    sigma    = sigma
    #print(tol, alpha, max_iter, step_size)

    # Normalize pixels
    I_1_ = I_1_/255.
    I_2_ = I_2_/255.

    # Horn Schunk Convolution Kernel
    K_hs_ = np.array([[1./12., 1./6., 1./12.],
                      [1./6. ,    0., 1./6. ],
                      [1./12., 1./6., 1./12.]])

    # Implement Lucas Kanade for each point, calculate I_x, I_y, I_t
    # Numerial Derivatives Kernel
    K_x_ = np.array([[-1., 1.], [-1., 1.]]) * sigma
    K_y_ = np.array([[-1.,-1.], [ 1., 1.]]) * sigma
    K_t_ = np.array([[ 1., 1.], [ 1., 1.]]) * sigma

    # Convolution numerical derivative
    f_x = convolve2d(I_1_, K_x_, boundary = 'symm', mode = 'same') + convolve2d(I_2_,  K_x_, boundary = 'symm', mode = 'same')
    f_y = convolve2d(I_1_, K_y_, boundary = 'symm', mode = 'same') + convolve2d(I_2_,  K_y_, boundary = 'symm', mode = 'same')
    f_t = convolve2d(I_1_, K_t_, boundary = 'symm', mode = 'same') + convolve2d(I_2_, -K_t_, boundary = 'symm', mode = 'same')

    d, n = I_1_.shape
    v_x_ = np.zeros((d, n))
    v_y_ = np.zeros((d, n))
    d_0_ = 0.

    for i in range(max_iter):

        # Horn Schunk convolution step
        bar_v_x_ = convolve2d(v_x_, K_hs_, boundary = 'symm', mode = 'same')
        bar_v_y_ = convolve2d(v_y_, K_hs_, boundary = 'symm', mode = 'same')

        # Iterative incremental numerical optimization
        d_1_ = (f_x * bar_v_x_ + f_y * bar_v_y_ + f_t) / (alpha**2 + f_x**2 + f_y**2)
        v_x_ = bar_v_x_ - f_x * d_1_
        v_y_ = bar_v_y_ - f_y * d_1_

        # Stop optimization when algorithm converges
        if abs(d_1_.sum() - d_0_) < tol:
            break
        else:
            d_0_ = d_1_.sum()

    return np.dstack((v_x_, v_y_))


def _spatial_cross_correlation(I_1_, I_2_, window_size = 5, degree = 9, step_size = 1):
    """Parameters:
        I_1_ – first 8-bit single-channel input image.
        I_2_ – second input image of the same size and the same type as prev.
        window_size – averaging window size; larger values increase the algorithm robustness to image noise
                  and give more chances for fast motion detection, but yield more blurred motion field.
        step_size - Pixels jump in between windows.
        degree - polynomial degree for the approximation of a non-integer motion"""

    # Type of variable definition
    window_size = int(window_size)
    step_size   = int(step_size)
    degree      = int(degree)
    print(window_size, step_size, degree)

    def __fast_cross_correlation(x_1, x_2):

        d, n = x_1.shape

        x_pad_1 = np.zeros((d*2, n*2))
        x_pad_2 = np.zeros((d*2, n*2))

        x_pad_1[:d,:n] = x_1
        x_pad_2[:d,:n] = x_2

        x_fft_1 = fft2(x_pad_1)
        x_fft_2 = fft2(x_pad_2).conj()
        x_fft   = x_fft_1 * x_fft_2

        x_ifft  = ifft2(x_fft)
        x_shift = np.abs(fftshift(x_ifft))

        return x_shift[(d//2 + 1):(-d//2 + 1), (n//2 + 1):(-n//2 + 1)]

    # Calculate Cross-Correlation
    def __cross_correlation(W_1_, W_2_):    return np.nan_to_num(__fast_cross_correlation(W_1_, W_2_))
    def __polynomial_fit(x, x_cc_, degree): return np.poly1d(np.polyfit(x, x_cc_, deg = degree))
    def __find_maxima(_f, x0):              return fmin(_f, x0, disp = 0)

    # All pixels within [-w, w] are in the window.
    w = window_size//2
    x = np.linspace(0, 2*w, 2*w + 1)

    d, n = I_1_.shape
    v_x_ = np.zeros((d, n))
    v_y_ = np.zeros((d, n))

    # Zero-padded images to have enough pixels on edges windows
    P_1_ = np.zeros((d + 2*w, n + 2*w))
    P_2_ = np.zeros((d + 2*w, n + 2*w))
    P_1_[w:-w, w:-w] = I_1_
    P_2_[w:-w, w:-w] = I_2_

    for i in range(w, d + w, step_size):
        for j in range(w, n + w, step_size):

            # Select window on images
            W_1_ = P_1_[(i - w):(i + w + 1), (j - w):(j + w + 1)]
            W_2_ = P_2_[(i - w):(i + w + 1), (j - w):(j + w + 1)]

            cc_ = __cross_correlation(W_1_, W_2_)

            # Integrate Cross-Correlation
            x_cc_ = np.sum(cc_, axis = 0)
            y_cc_ = np.sum(cc_, axis = 1)

            _f_x = __polynomial_fit(x, - x_cc_, degree)
            _f_y = __polynomial_fit(x, - y_cc_, degree)

            # Find sub-pixel maximum Cross-Correlation
            dx = __find_maxima(_f_x, w)
            dy = __find_maxima(_f_y, w)

            # Get the motion with respect to the window center
            v_x_[i - w, j - w] = w - dx
            v_y_[i - w, j - w] = w - dy

    return np.dstack((v_x_, v_y_))


def _normalized_cross_correlation(I_1_, I_2_, window_size = 5, degree = 9, step_size = 1):
    """Parameters:
        I_1_ – first 8-bit single-channel input image.
        I_2_ – second input image of the same size and the same type as prev.
        window_size – averaging window size; larger values increase algorithms robustness against image noise,
                        and gives more chances for fast motion detection, but it yields to more blurred motion field.
        step_size - Pixels jump in between windows.
        degree - polynomial degree for the approximation of a non-integer motion"""

    # Input variable definition in the correct format
    window_size = int(window_size)
    degree      = int(degree)
    step_size   = int(step_size)
    print(window_size, degree, step_size)
    # Normalize the cross-correlation
    def __normalize(a, b):
        a = (a - np.mean(a)) / (np.std(a) * len(a))
        b = (b - np.mean(b)) / (np.std(b))
        return a, b
    # Cross-Correlation in Frequency domain
    def __fast_cross_correlation(x_1, x_2):
        # Window Dimension
        d, n = x_1.shape
        # Zero-pad Image-Window
        x_pad_1 = np.zeros((d*2, n*2))
        x_pad_2 = np.zeros((d*2, n*2))
        x_pad_1[:d,:n] = x_1
        x_pad_2[:d,:n] = x_2
        # Fast Fourier Transform for cross-correlation in frequency domain
        x_fft_1 = fft2(x_pad_1)
        x_fft_2 = fft2(x_pad_2).conj()
        x_fft   = x_fft_1 * x_fft_2
        # Inverse Fourir trainsform to get back to time domain and fix image shifting
        x_ifft  = ifft2(x_fft)
        x_shift = np.abs(fftshift(x_ifft))
        return x_shift[(d//2 + 1):(-d//2 + 1), (n//2 + 1):(-n//2 + 1)]
    # Calculate Cross-Correlation
    def __cross_correlation(W_1_, W_2_):
        W_1_, W_2_ = __normalize(W_1_, W_2_)
        return np.nan_to_num(__fast_cross_correlation(W_1_, W_2_))
    # Adjust polynomial to estimate cross-correlation function for each velocity component
    def __polynomial_fit(x, x_cc_, degree): return np.poly1d(np.polyfit(x, x_cc_, deg = degree))
    # Find maxima in the polynomal function
    def __find_maxima(_f, x0): return fmin(_f, x0, disp = 0)
    # Update pixel coordinates
    def __check_corners(x_, x, w):
        # Cross-Correlation windows upper and lower interval
        x_1_, x_2_ = x_ - w, x_ + w + 1
        # Forze corners to stay on the image frame
        if x_1_ < 0:
            x_1_ = 0
        if x_2_ > x - 1:
            x_2 = x - 1
        return x_1_, x_2_

    # All pixels within [-w, w] are in the window.
    w = window_size//2
    # Variables initialization
    d, n = I_1_.shape
    v_x_ = np.zeros((d, n))
    v_y_ = np.zeros((d, n))
    # Loop over all pixels ...
    for i_ in range(0, d, step_size):
        for j_ in range(0, n, step_size):
            # Adjust Pixel coordinates to have enough pixels on edges windows
            i_1_, i_2_ = __check_corners(i_, d, w)
            j_1_, j_2_ = __check_corners(j_, n, w)
            # Select window on images
            W_1_ = I_1_[i_1_:i_2_, j_1_:j_2_]
            W_2_ = I_2_[i_1_:i_2_, j_1_:j_2_]
            # Calculate normalize cross-correlation
            cc_ = __cross_correlation(W_1_, W_2_)
            # Integrate Cross-Correlation
            x_cc_ = np.sum(cc_, axis = 0)
            y_cc_ = np.sum(cc_, axis = 1)
            # X-axis coordinates for the 1D cross-correlation function obtained after integration
            x_ = np.linspace(0, x_cc_.shape[0] - 1, x_cc_.shape[0])
            y_ = np.linspace(0, y_cc_.shape[0] - 1, y_cc_.shape[0])
            # Find Middle point images
            x_mean = np.mean(x_)//1
            y_mean = np.mean(y_)//1
            #print(x_cc_.shape, y_cc_.shape, x_.shape, x_mean, w)
            _f_x = __polynomial_fit(x_, - x_cc_, degree)
            _f_y = __polynomial_fit(y_, - y_cc_, degree)
            # Find sub-pixel maximum Cross-Correlation
            dx = __find_maxima(_f_x, x0 = x_mean)
            dy = __find_maxima(_f_y, x0 = y_mean)
            # Get the motion with respect to the window center
            v_x_[i_, j_] = x_mean - dx
            v_y_[i_, j_] = y_mean - dy
    # Return Veloctiy Vectors for each pixels on the image
    return np.dstack((v_x_, v_y_))


__all__ = ['_lucas_kanade', '_horn_schunck', '_pyramidal_weighted_lucas_kanade', '_pyramidal_weighted_horn_schunck',
           '_farneback', '_spatial_cross_correlation', '_normalized_cross_correlation',
           '_pyramidal_lucas_kanade', '_pyramidal_posterior_weighted_lucas_kanade']
