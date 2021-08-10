import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from scipy import optimize
from scipy.ndimage import median_filter
from scipy.stats import vonmises, norm, beta, uniform, multivariate_normal
from scipy.special import logsumexp, iv, beta, gamma, digamma

from utils import *
from detection_clustering_utils import *
from lib_motion_vector import _farneback, _horn_schunck, _lucas_kanade, _pyramidal_weighted_lucas_kanade
from feature_extraction_utils import _magnitude

from lib_motion_vector import _farneback, _lucas_kanade, _pyramidal_weighted_lucas_kanade
from lib_motion_vector import _weighted_lucas_kanade, _pyramidal_lucas_kanade

# Calculate the motion of the clouds between two consecutive frames
def _LK_cloud_velocity_vector(I_1_norm_, I_2_norm_, opt_):
    # Motion vectors approximation method can always change, output has to remain the same
    F_lk_ = _lucas_kanade(I_1_norm_, I_2_norm_,
                          window_size = opt_[0], tau = opt_[1], sigma = opt_[2])
    # Velocity magnitude
    M_lk_ = _magnitude(F_lk_[..., 1], F_lk_[..., 0])
    return np.nan_to_num(F_lk_), np.nan_to_num(M_lk_)

# Redece the dimensions of the clouds velocity field
def _cloud_velocity_vector_average(F_, M_, Q_, X_, Y_, U_, V_, W_, Z_, x_, y_, step_size, lag, tau, display = False):
    # Applying an average windown over a vector field
    def __average_velocity_field(F_, Q_, step_size):
        # Average only between velotiy vectors
        def ___average_reshape(X_, new_shape):
            shape = (new_shape[0], X_.shape[0] // new_shape[0],
                     new_shape[1], X_.shape[1] // new_shape[1])

            #return X_.reshape(shape).mean(-1).mean(1)
            return np.mean(np.mean(X_.reshape(shape), axis = -1), axis = 1)
            #return np.median(np.median(X_.reshape(shape), axis = -1), axis = 1)

        # Slidding Windown Average
        def ___slidding_window(X_, step_size, M, N, K, dim = 2):
            # Variable Initialization
            m = np.divmod(M, step_size)[0]
            n = np.divmod(N, step_size)[0]
            x_ = np.zeros((m, n, dim))
            # Loop over field component
            for i in np.arange(K):
                x_[..., i] = ___average_reshape(X_[..., i], (m, n))
            return x_

        # Average Sldding Window for Velocity Vectors
        M, N, D = F_.shape
        f_ = ___slidding_window(F_, step_size, M, N, D)

        # Average Sldding Window for Velocity Vector Probabilities
        M, N, N_layers = Q_.shape
        q_ = ___slidding_window(Q_, step_size, M, N, N_layers)

        if display:
            plt.figure(figsize = (20, 5))
            plt.subplot(131)
            plt.title('Avg. Velocity Vector Magnitude')
            plt.imshow(_magnitude(f_[..., 0], f_[..., 1]))
            plt.colorbar()
            plt.subplot(132)
            plt.title('Avg. Probability Layer 1')
            plt.imshow(q_[..., 0])
            plt.colorbar()
            plt.subplot(133)
            plt.title('Avg. Probability Layer 2')
            plt.imshow(q_[..., 1])
            plt.colorbar()
            plt.show()

        # Return Average Velocity in vector form
        return q_[..., 0].flatten(), q_[..., 1].flatten(), f_[..., 0].flatten(), f_[..., 1].flatten()

    # Lagged list of consecutive vectors
    def __lag_data(x_lag, y_lag, u_lag, v_lag, w_lag, z_lag, wz_, xy_, uv_, lag):
        # Keep the desire number of lags on the list by removing the last and aadding at the bigging
        if len(x_lag) == lag:
            x_lag.pop(0)
            y_lag.pop(0)
            u_lag.pop(0)
            v_lag.pop(0)
            z_lag.pop(0)
            w_lag.pop(0)
        # Keep adding until we have the desired number of lag time stamps
        x_lag.append(xy_[0])
        y_lag.append(xy_[1])
        u_lag.append(uv_[0])
        v_lag.append(uv_[1])
        w_lag.append(wz_[0])
        z_lag.append(wz_[1])
        return x_lag, y_lag, u_lag, v_lag, w_lag, z_lag

    #I_ = M_ > tau
    # Applying mean window to reduce velocity field dimensions
    w_, z_, u_, v_ = __average_velocity_field(F_, Q_, step_size)
    # Index of thresholding velocity vectors to remove noisy vectors
    i_ = _magnitude(u_, v_) > tau
    # Lagging data for wind velocity field estimation
    xy_ = [x_[i_], y_[i_]]
    wz_ = [w_[i_], z_[i_]]
    uv_ = [u_[i_], v_[i_]]
    # Lag Data
    return __lag_data(X_, Y_, U_, V_, W_, Z_, wz_, xy_, uv_, lag)

# Redece the dimensions of the clouds velocity field
def _cloud_velocity_vector_average_v2(F_, M_, Q_, X_, Y_, U_, V_, W_, Z_, x_, y_, step_size, tau, display = False):
    # Applying an average windown over a vector field
    def __average_velocity_field(F_, Q_, step_size):
        # Average only between velotiy vectors
        def ___average_reshape(X_, new_shape):
            shape = (new_shape[0], X_.shape[0] // new_shape[0],
                     new_shape[1], X_.shape[1] // new_shape[1])

            #return X_.reshape(shape).mean(-1).mean(1)
            return np.mean(np.mean(X_.reshape(shape), axis = -1), axis = 1)
            #return np.median(np.median(X_.reshape(shape), axis = -1), axis = 1)

        # Slidding Windown Average
        def ___slidding_window(X_, step_size, M, N, K, dim = 2):
            # Variable Initialization
            m = np.divmod(M, step_size)[0]
            n = np.divmod(N, step_size)[0]
            x_ = np.zeros((m, n, dim))
            # Loop over field component
            for i in np.arange(K):
                x_[..., i] = ___average_reshape(X_[..., i], (m, n))
            return x_

        # Average Sldding Window for Velocity Vectors
        M, N, D = F_.shape
        f_ = ___slidding_window(F_, step_size, M, N, D)

        # Average Sldding Window for Velocity Vector Probabilities
        M, N, N_layers = Q_.shape
        q_ = ___slidding_window(Q_, step_size, M, N, N_layers)

        if display:
            plt.figure(figsize = (20, 5))
            plt.subplot(131)
            plt.title('Avg. Velocity Vector Magnitude')
            plt.imshow(_magnitude(f_[..., 0], f_[..., 1]))
            plt.colorbar()
            plt.subplot(132)
            plt.title('Avg. Probability Layer 1')
            plt.imshow(q_[..., 0])
            plt.colorbar()
            plt.subplot(133)
            plt.title('Avg. Probability Layer 2')
            plt.imshow(q_[..., 1])
            plt.colorbar()
            plt.show()

        # Return Average Velocity in vector form
        return q_[..., 0].flatten(), q_[..., 1].flatten(), f_[..., 0].flatten(), f_[..., 1].flatten()
    #I_ = M_ > tau
    # Applying mean window to reduce velocity field dimensions
    w_, z_, u_, v_ = __average_velocity_field(F_, Q_, step_size)
    # Index of thresholding velocity vectors to remove noisy vectors
    i_ = _magnitude(u_, v_) > tau
    # Lagging data for wind velocity field estimation
    return [x_[i_], y_[i_]], [w_[i_], z_[i_]], [u_[i_], v_[i_]]

def _autoregressive_dataset(X_, Y_, U_, V_, W_, Z_, wz_, xy_, uv_, lag):
    # Lagged list of consecutive vectors
    def __lag_data(x_lag, y_lag, u_lag, v_lag, w_lag, z_lag, wz_, xy_, uv_, lag):
        # Keep the desire number of lags on the list by removing the last and aadding at the bigging
        if len(x_lag) == lag:
            x_lag.pop(0)
            y_lag.pop(0)
            u_lag.pop(0)
            v_lag.pop(0)
            z_lag.pop(0)
            w_lag.pop(0)
        # Keep adding until we have the desired number of lag time stamps
        x_lag.append(xy_[0])
        y_lag.append(xy_[1])
        u_lag.append(uv_[0])
        v_lag.append(uv_[1])
        w_lag.append(wz_[0])
        z_lag.append(wz_[1])
        return x_lag, y_lag, u_lag, v_lag, w_lag, z_lag
    # Lag Data
    return __lag_data(X_, Y_, U_, V_, W_, Z_, wz_, xy_, uv_, lag)

# Select the most probable set of vectors from the clouds velocity field
def _cloud_velocity_vector_selection_v1(X_lag, Y_lag, U_lag, V_lag, W_lag, Z_lag, N_sel, reg_sigma, percentage = 0.5):
    # Vector Selection main function
    def __vector_selection(XY_, UV_, WZ_, n_sel):
        N, D = WZ_.shape
        # Normal Probability for Regularization
        m_ = np.sqrt(UV_[..., 0]**2 + UV_[..., 1]**2)
        z_ = norm(0, reg_sigma).pdf(m_)
        WZ_ = WZ_ * np.tile(z_[..., np.newaxis], (1, D))
        # Normalize Probabilities to add to 1.
        wz_ = np.sum(WZ_, axis = 1) + 1e-5
        wz_ /=np.sum(wz_)
        # Importance Sample Selection
        idx_ = np.arange(N)
        return np.random.choice(idx_, size = n_sel, p = wz_, replace = False)
    # Select per row the velocity vectors
    def __select_vectors_per_row(XY_, UV_, WZ_):
        # Variables Initialization
        index_ = []
        # Find out number of rows and vector per row
        y_, N_ = np.unique(XY_[:, 1], return_counts = True)
        # Calculate the number of vector to be selected per row proportionally
        N = np.sum(N_)
        if N < N_sel:
            n_sel = N
        else:
            n_sel = N_sel
        n_ = np.around(n_sel * N_ / N).astype(int)
        # Loop over distance data
        for y, n in zip(y_, n_):
            idx_ = np.where(XY_[:, 1] == y)[0]
            index_.append(idx_[__vector_selection(XY_[idx_, :], UV_[idx_, :], WZ_[idx_, :], n)])
        return np.hstack(index_)
    # Divide features sources sets in training and test
    def __divide_dataset(XY_, UV_, WZ_, index_):
        # Number of samples for training and test
        N = index_.shape[0]
        N_tr = int(N * percentage)
        N_ts = N - N_tr
        np.random.shuffle(index_)
        return [XY_[index_[:N_tr], :], UV_[index_[:N_tr], :], WZ_[index_[:N_tr], :]], \
               [XY_[index_[-N_ts:], :], UV_[index_[-N_ts:], :], WZ_[index_[-N_ts:], :]]
    # Converting from list of vectors to matrix form the samples of clouds velocity field
    XY_ = np.concatenate((np.vstack(X_lag), np.vstack(Y_lag)), axis = 1)
    UV_ = np.concatenate((np.hstack(U_lag)[..., np.newaxis], np.hstack(V_lag)[..., np.newaxis]), axis = 1)
    WZ_ = np.concatenate((np.hstack(W_lag)[..., np.newaxis], np.hstack(Z_lag)[..., np.newaxis]), axis = 1)
    # Find number of layers
    wind_flow_indicator_ = np.sum(WZ_, axis = 0) > 0.
    # Selecting the most likely set of vectors recording over a period of lags
    index_ = __select_vectors_per_row(XY_, UV_, WZ_)
    # Divide features sources sets in training and test
    X_tr_, X_ts_ = __divide_dataset(XY_, UV_, WZ_, index_)
    return X_tr_, X_ts_, wind_flow_indicator_

# Select the most probable set of vectors from the clouds velocity field
def _cloud_velocity_vector_selection_v2(X_lag, Y_lag, U_lag, V_lag, W_lag, Z_lag, N_sel, reg_sigma, percentage = 0.5):
    def __infer_dist(X_, WZ_, wind_flow_indicator_):
        # Variable Initialization
        N, D = WZ_.shape
        N_ = [[], []]
        # Loop over wind layers
        for i in np.array([0, 1])[wind_flow_indicator_]:
            # Variables in Vetor and Matrix from
            w_ = WZ_[..., i][:, np.newaxis]
            o_ = np.ones(w_.shape)
            W_ = np.identity(N) * w_
            # Compute Mean
            mu_ = (X_.T @ w_) / (o_.T @ w_)
            # Compute Covariance
            MU_ = np.tile(mu_.T, (N, 1))
            sigma_ = ( ( X_ - MU_ ).T @ W_ @ ( X_ - MU_ ) ) / (o_.T @ w_)
            N_[i] = multivariate_normal(np.squeeze(mu_), sigma_)
        return N_
    # Evaluate probabilities of the velocity vectors
    def __eval_proba(UV_, WZ_, N_, wind_flow_indicator_):
        # Variables Initialization
        N, D = WZ_.shape
        Z_ = np.zeros((N, D))
        # Normal Probability for Regularization
        m_ = np.sqrt(UV_[..., 0]**2 + UV_[..., 1]**2)
        w_ = norm(0, reg_sigma).pdf(m_)
        # loop over wind layer Distritubiton
        for i in np.array([0, 1])[wind_flow_indicator_]:
            Z_[..., i] = N_[i].pdf(UV_) * w_
        return Z_
    # Vector Selection main function
    def __vector_selection(XY_, UV_, WZ_, Z_, n_sel):
        # Normalize Probabilities to add to 1.
        N, D = Z_.shape
        z_ = np.sum(Z_, axis = 1) + 1e-5
        z_ /=np.sum(z_)
        # Importance Sample Selection
        idx_ = np.arange(N)
        return np.random.choice(idx_, size = n_sel, p = z_, replace = False)
    # Select per row the velocity vectors
    def __select_vectors_per_row(XY_, UV_, WZ_, Z_):
        # Variables Initialization
        index_ = []
        # Find out number of rows and vector per row
        y_, N_ = np.unique(XY_[:, 1], return_counts = True)
        # Calculate the number of vector to be selected per row proportionally
        N = np.sum(N_)
        if N < N_sel:
            n_sel = N
        else:
            n_sel = N_sel
        n_ = np.around(n_sel * N_ / N).astype(int)
        # Loop over distance data
        for y, n in zip(y_, n_):
            idx_ = np.where(XY_[:, 1] == y)[0]
            index_.append(idx_[__vector_selection(XY_[idx_, :], UV_[idx_, :], WZ_[idx_, :], Z_[idx_, :], n)])
        return np.hstack(index_)
    # Divide features sources sets in training and test
    def __divide_dataset(XY_, UV_, WZ_, index_):
        # Number of samples for training and test
        N = index_.shape[0]
        N_tr = int(N * percentage)
        N_ts = N - N_tr
        np.random.shuffle(index_)
        return [XY_[index_[:N_tr], :], UV_[index_[:N_tr], :], WZ_[index_[:N_tr], :]], \
               [XY_[index_[-N_ts:], :], UV_[index_[-N_ts:], :], WZ_[index_[-N_ts:], :]]
    # Converting from list of vectors to matrix form the samples of clouds velocity field
    XY_ = np.concatenate((np.vstack(X_lag), np.vstack(Y_lag)), axis = 1)
    UV_ = np.concatenate((np.hstack(U_lag)[..., np.newaxis], np.hstack(V_lag)[..., np.newaxis]), axis = 1)
    WZ_ = np.concatenate((np.hstack(W_lag)[..., np.newaxis], np.hstack(Z_lag)[..., np.newaxis]), axis = 1)
    # Find number of layers
    wind_flow_indicator_ = np.sum(WZ_, axis = 0) > 0.
    # Infer Velocity Vector Distribution per wind velocity layer
    N_ =  __infer_dist(UV_, WZ_, wind_flow_indicator_)
    # Evaluate Velocity Vector Probabilities
    Z_ = __eval_proba(UV_, WZ_, N_, wind_flow_indicator_)
    # Selecting the most likely set of vectors recording over a period of lags
    index_ = __select_vectors_per_row(XY_, UV_, WZ_, Z_)
    # Divide features sources sets in training and test
    X_tr_, X_ts_ = __divide_dataset(XY_, UV_, WZ_, index_)
    return X_tr_, X_ts_, wind_flow_indicator_

# Select the most probable set of vectors from the clouds velocity field
def _cloud_velocity_vector_selection_v3(X_lag, Y_lag, U_lag, V_lag, W_lag, Z_lag, N_sel, reg_mean, reg_sigma, percentage = 0.5):
    def __infer_dist(X_, WZ_, wind_flow_indicator_):
        # Variable Initialization
        N, D = WZ_.shape
        N_ = [[], []]
        # Loop over wind layers
        for i in np.array([0, 1])[wind_flow_indicator_]:
            # Variables in Vetor and Matrix from
            w_ = WZ_[..., i][:, np.newaxis]
            o_ = np.ones(w_.shape)
            W_ = np.identity(N) * w_
            # Compute Mean
            mu_ = (X_.T @ w_) / (o_.T @ w_)
            # Compute Covariance
            MU_ = np.tile(mu_.T, (N, 1))
            sigma_ = ( ( X_ - MU_ ).T @ W_ @ ( X_ - MU_ ) ) / (o_.T @ w_)
            N_[i] = multivariate_normal(np.squeeze(mu_), sigma_)
        return N_
    # Evaluate probabilities of the velocity vectors
    def __eval_proba(UV_, WZ_, N_, wind_flow_indicator_):
        # Variables Initialization
        N, D = WZ_.shape
        Z_ = np.zeros((N, D))
        # Normal Probability for Regularization
        m_ = np.sqrt(UV_[..., 0]**2 + UV_[..., 1]**2)
        w_ = norm(reg_mean, reg_sigma).pdf(m_)
        # loop over wind layer Distritubiton
        for i in np.array([0, 1])[wind_flow_indicator_]:
            Z_[..., i] = N_[i].pdf(UV_) * w_
        return Z_
    # Vector Selection main function
    def __vector_selection(XY_, UV_, WZ_, Z_, N_sel):
        # Normalize Probabilities to add to 1.
        N, D = Z_.shape
        z_ = np.sum(Z_, axis = 1) + 1e-5
        z_ /=np.sum(z_)
        # Importance Sample Selection
        idx_ = np.arange(N)
        if N < N_sel:
            n_sel = N
        else:
            n_sel = N_sel
        return np.random.choice(idx_, size = n_sel, p = z_, replace = False)

    # Divide features sources sets in training and test
    def __divide_dataset(XY_, UV_, WZ_, index_):
        # Number of samples for training and test
        N = index_.shape[0]
        N_tr = int(N * percentage)
        N_ts = N - N_tr
        np.random.shuffle(index_)
        return [XY_[index_[:N_tr], :], UV_[index_[:N_tr], :], WZ_[index_[:N_tr], :]], \
               [XY_[index_[-N_ts:], :], UV_[index_[-N_ts:], :], WZ_[index_[-N_ts:], :]]

    # Converting from list of vectors to matrix form the samples of clouds velocity field
    XY_ = np.concatenate((np.vstack(X_lag), np.vstack(Y_lag)), axis = 1)
    UV_ = np.concatenate((np.hstack(U_lag)[..., np.newaxis], np.hstack(V_lag)[..., np.newaxis]), axis = 1)
    WZ_ = np.concatenate((np.hstack(W_lag)[..., np.newaxis], np.hstack(Z_lag)[..., np.newaxis]), axis = 1)
    # Find number of layers
    wind_flow_indicator_ = np.sum(WZ_, axis = 0) > 0.
    # Infer Velocity Vector Distribution per wind velocity layer
    N_ =  __infer_dist(UV_, WZ_, wind_flow_indicator_)
    # Evaluate Velocity Vector Probabilities
    Z_ = __eval_proba(UV_, WZ_, N_, wind_flow_indicator_)
    # Selecting the most likely set of vectors recording over a period of lags
    index_ = __vector_selection(XY_, UV_, WZ_, Z_, N_sel)
    # Divide features sources sets in training and test
    X_tr_, X_ts_ = __divide_dataset(XY_, UV_, WZ_, index_)
    return X_tr_, X_ts_, wind_flow_indicator_


# Calculate the motion of the clouds between two consecutive frames
def _cloud_velocity_vector_fb(I_1_norm_, I_2_norm_, opt_, F_1_ = None):
    # Motion vectors approximation method can always change, output has to remain the same
    F_2_ = _farneback(I_1_norm_, I_2_norm_, pyr_scale = opt_[0], n_levels = opt_[1],
                      window_size = opt_[2], n_iter = opt_[3], n_poly = opt_[4], sigma = opt_[5], flow = F_1_)
    M_2_ = _magnitude(F_2_[..., 1], F_2_[..., 0])
    return np.nan_to_num(F_2_), np.nan_to_num(M_2_)

# Calculate the motion of the clouds between two consecutive frames
def _cloud_velocity_vector_lk(I_1_norm_, I_2_norm_, opt_):
    # Motion vectors approximation method can always change, output has to remain the same
    F_2_ = _lucas_kanade(I_1_norm_, I_2_norm_, window_size = opt_[0], tau = opt_[1], sigma = opt_[2])
    M_2_ = _magnitude(F_2_[..., 1], F_2_[..., 0])
    return np.nan_to_num(F_2_), np.nan_to_num(M_2_)

# Calculate the motion of the clouds between two consecutive frames
def _cloud_velocity_vector_hs(I_1_norm_, I_2_norm_, opt_):
    # Motion vectors approximation method can always change, output has to remain the same
    F_2_ = _horn_schunck(I_1_norm_, I_2_norm_, tol = 1e-5, alpha = opt_[0], max_iter = 1e3, sigma = opt_[1])
    M_2_ = _magnitude(F_2_[..., 1], F_2_[..., 0])
    return np.nan_to_num(F_2_), np.nan_to_num(M_2_)

# Calculate the motion of the clouds between two consecutive frames
def _cloud_velocity_vector_wlk(I_1_norm_, I_2_norm_, XYZ_, opt_):
    # Motion vectors approximation method can always change, output has to remain the same
    F_2_ = _weighted_lucas_kanade(I_1_norm_, I_2_norm_, W_ = XYZ_[1][..., 2], window_size = opt_[0], tau = opt_[1], sigma = opt_[2])
    M_2_ = _magnitude(F_2_[..., 1], F_2_[..., 0])
    return np.nan_to_num(F_2_), np.nan_to_num(M_2_)

# Calculate the motion of the clouds between two consecutive frames
def _cloud_velocity_vector_comparison(I_1_norm_, I_2_norm_, opt_, F_1_ = None, display = False):
    lk_opt_, hs_opt_, fb_opt_ = opt_
    # Motion vectors approximation method can always change, output has to remain the same
    F_lk_ = _lukas_kanade(I_1_norm_, I_2_norm_, window_size = lk_opt_[0], tau = lk_opt_[1], sigma = lk_opt_[2])
    F_hs_ = _horn_schunck(I_1_norm_, I_2_norm_, tol = 1e-5, alpha = hs_opt_[0], max_iter = 1e3, sigma = hs_opt_[1])
    F_fb_ = _farneback(I_1_norm_, I_2_norm_, pyr_scale = fb_opt_[0], n_levels = fb_opt_[1],
                       window_size = fb_opt_[2], n_iter = fb_opt_[3], n_poly = fb_opt_[4], sigma = fb_opt_[5], flow = F_1_)
    # Velocity magnitude
    M_lk_ = _magnitude(F_lk_[..., 1], F_lk_[..., 0])
    M_hs_ = _magnitude(F_hs_[..., 1], F_hs_[..., 0])
    M_fb_ = _magnitude(F_fb_[..., 1], F_fb_[..., 0])

    if display:

        plt.figure(figsize = (5, 20))
        plt.subplot(511)
        plt.title('Previous 8bits IR Image', fontsize = 15)
        plt.imshow(I_1_norm_, cmap = 'inferno')
        plt.xlabel('x-axis', fontsize = 15)
        plt.ylabel('y-axis', fontsize = 15)
        plt.colorbar()
        plt.subplot(512)
        plt.title('Current 8bits IR Image', fontsize = 15)
        plt.imshow(I_2_norm_, cmap = 'inferno')
        plt.xlabel('x-axis', fontsize = 15)
        plt.ylabel('y-axis', fontsize = 15)
        plt.colorbar()
        plt.subplot(513)
        plt.title('Lucas-Kanade', fontsize = 15)
        plt.imshow(M_lk_, cmap = 'jet')
        plt.xlabel('x-axis', fontsize = 15)
        plt.ylabel('y-axis', fontsize = 5)
        plt.colorbar()
        plt.subplot(514)
        plt.title('Horn-Schunck', fontsize = 15)
        plt.xlabel('x-axis', fontsize = 15)
        plt.ylabel('y-axis', fontsize = 15)
        plt.imshow(M_hs_, cmap = 'jet')
        plt.colorbar()
        plt.subplot(515)
        plt.title('Farneback', fontsize = 15)
        plt.imshow(M_fb_, cmap = 'jet')
        plt.xlabel('x-axis', fontsize = 15)
        plt.ylabel('y-axis', fontsize = 15)
        plt.colorbar()

    return [np.nan_to_num(F_lk_), np.nan_to_num(F_hs_), np.nan_to_num(F_fb_)], \
           [np.nan_to_num(M_lk_), np.nan_to_num(M_hs_), np.nan_to_num(M_fb_)]

# Calculate the average, meadian, and standard deviation of the height of a labelled object
def _cloud_label_height(H_, I_labels_, verbose = False):
    m_to_km = 1000.
    # Find Labels
    idx_class_ = np.unique(I_labels_)
    # Variables Initialization
    h_mean_ = np.zeros(idx_class_.shape[0])
    h_medi_ = np.zeros(idx_class_.shape[0])
    h_std_  = np.zeros(idx_class_.shape[0])
    # Loop over labels
    for i, j in zip(idx_class_, range(idx_class_.shape[0])):
        # Calculate Statistics
        h_mean_[j] = np.mean(H_[I_labels_ == i])/m_to_km
        h_medi_[j] = np.median(H_[I_labels_ == i])/m_to_km
        h_std_[j]  = np.std(H_[I_labels_ == i])/m_to_km
        if verbose:
            print('>> Label: {} Avg. Height: {} km Median Height: {} km Std. Height: {} km'.format(i, h_mean_[j], h_medi_[j], h_std_[j]))
    return h_mean_, h_medi_, h_std_

# Random Initialization of VM mixture model Paramters
def __init_Ga(_tools, X_, n_clusters):
    def __init(n_clusters):
        # Variable Initialization
        bounds_ = []
        a_init_ = np.empty((n_clusters))
        b_init_ = np.empty((n_clusters))
        k_init_ = np.empty((n_clusters))
        # Randon Initialization VM Distributions weights and Noise Distribution
        w_init_  = np.random.uniform(0, 1, n_clusters)[:, np.newaxis]
        #w_init_ /= w_init_.sum()
        # Gamma Distribution Parameters Initialization
        for k in range(n_clusters):
            # Define Variable Boundaries for Optimization
            bounds_.insert(0, (.1, 100.))
            bounds_.append((.1, 100.))
            bounds_.append((.1, 100.))
            # Random Initialization of the Gamma Parameters
            a_init_[k] = np.random.uniform(1., 5., 1)[0]
            b_init_[k] = np.random.uniform(1., 5., 1)[0]
            k_init_[k] = np.random.uniform(1., 5., 1)[0]
        return a_init_.copy(), b_init_.copy(), k_init_.copy(), w_init_.copy(), bounds_

    # Seperate Theta in Clusters Parameter lists
    def __get_parameters(_tools):
        n_clusters = _tools.Ga_[2]
        theta_  = _tools.Ga_[3]
        bounds_ = []
        a_init_ = np.empty((n_clusters))
        b_init_ = np.empty((n_clusters))
        k_init_ = np.empty((n_clusters))
        w_init_ = np.empty((n_clusters))
        for k in range(n_clusters):
            # Define Variable Boundaries for Optimization
            bounds_.insert(0, (.1, 100.))
            bounds_.append((.1, 100.))
            bounds_.append((.1, 100.))
            a_init_[k] = theta_[k::n_clusters][0]
            b_init_[k] = theta_[k::n_clusters][1]
            k_init_[k] = theta_[k::n_clusters][2]
            w_init_[k] = theta_[k::n_clusters][3]
        return a_init_, b_init_, k_init_, w_init_, bounds_

    # Constants Definition
    N, dim  = X_.shape
    ll_k    = - np.inf
    n_prev  = _tools.Ga_[2]
    # Initiliacation According number of clusters
    if n_clusters == n_prev:
        a_, b_, k_, w_, bounds_ = __get_parameters(_tools)
    if n_clusters == 2 & n_prev == 1:
        a_1, b_1, k_1, w_1, bounds_1 = __get_parameters(_tools)
        a_2, b_2, k_2, w_2, bounds_2 = __init(n_clusters = 1)
        a_ = np.hstack((a_1, a_2))
        b_ = np.hstack((b_1, b_2))
        k_ = np.hstack((k_1, k_2))
        w_ = np.hstack((w_1, w_2))
        bounds_ = bounds_1 + bounds_2
    if n_clusters > 0 & n_prev == 0:
        a_, b_, k_, w_, bounds_ = __init(n_clusters = n_clusters)
    if n_clusters < n_prev:
        i = np.argmax(_tools.Ga_[4])
        a_, b_, k_, w_, bounds_ = __get_parameters(_tools)
        a_ = a_[i][np.newaxis]
        b_ = b_[i][np.newaxis]
        k_ = k_[i][np.newaxis]
        w_ = w_[i][np.newaxis]
        bounds_ = bounds_[i*3: i*3 + 3]
    # Weights Should Add to 1
    w_/= w_.sum()
    return a_, b_, k_, w_, bounds_, ll_k, dim

# Random Initialization of VM mixture model Paramters
def __init_VM(_tools, X_, n_clusters):
    def __init(n_clusters):
        # Variable Initialization
        bounds_ = []
        m_init_ = np.empty((n_clusters, dim))
        k_init_ = np.empty((n_clusters, dim))
        # Randon Initialization VM Distributions weights and Noise Distribution
        w_init_ = np.random.uniform(0, 1, n_clusters + 1)[:, np.newaxis]
        #w_init_/= w_init_.sum()
        # Von Mises Distribution Parameters Initialization
        mu = np.pi/2. + np.random.uniform(-.5, .5, 1)[0]
        for k in range(n_clusters):
            # Define Variable Boundaries for Optimization
            bounds_.insert(0, (0., 2*np.pi))
            bounds_.append((0., 25.))
            # Random Initialization of the VM Parameters
            m_init_[k] = mu
            k_init_[k] = np.random.uniform(0, 25., 1)[0]
            mu = np.pi + np.random.uniform(-.5, .5, 1)[0]
        return m_init_.copy(), k_init_.copy(), w_init_.copy(), bounds_

    # Seperate Theta in Clusters Parameter lists
    def __get_parameters(_tools, noise = False):
        n_clusters = _tools.VM_[2]
        theta_  = _tools.VM_[3]
        bounds_ = []
        m_init_ = np.empty((n_clusters))
        k_init_ = np.empty((n_clusters))
        if noise: w_init_ = np.empty((n_clusters + 1))
        else:     w_init_ = np.empty((n_clusters))
        for k in range(n_clusters):
            # Define Variable Boundaries for Optimization
            bounds_.insert(0, (.1, 100.))
            bounds_.append((.1, 100.))
            bounds_.append((.1, 100.))
            m_init_[k] = theta_[k::n_clusters][0]
            k_init_[k] = theta_[k::n_clusters][1]
            w_init_[k] = theta_[k::n_clusters][2]
        if noise: w_init_[-1] = theta_[-1]
        return m_init_, k_init_, w_init_, bounds_

    # Constants Definition
    N, dim  = X_.shape
    ll_k    = - np.inf
    n_prev  = _tools.VM_[2]
    # Initiliacation According number of clusters
    if n_clusters == n_prev:
        m_, k_, w_, bounds_ = __get_parameters(_tools, noise = True)
    if n_clusters == 2 & n_prev == 1:
        m_1, k_1, w_1, bounds_1 = __get_parameters(_tools, noise = False)
        m_1, k_1, w_1, bounds_1 = __init(n_clusters = 1)
        m_ = np.hstack((m_1, m_2))
        k_ = np.hstack((k_1, k_2))
        w_ = np.hstack((w_1, w_2))
        bounds_ = bounds_1 + bounds_2
    if n_clusters > 0 & n_prev == 0:
        m_, k_, w_, bounds_ = __init(n_clusters = n_clusters)
    if n_clusters < n_prev:
        i = np.argmax(_tools.VM_[4])
        m_, k_, w_, bounds_ = __get_parameters(_tools, noise = True)
        m_ = m_[i][np.newaxis]
        k_ = k_[i][np.newaxis]
        w_ = np.array((w_[i], w_[-1]))
        bounds_ = bounds_[i*2: i*2 + 2]
    # Weights Should Add to 1
    w_/= w_.sum()
    return m_, k_, w_, bounds_, ll_k, dim


# Calculate Euclidian distance between means
def __euclidian_distane(theta_0_, theta_1_):
    A_ = np.zeros((theta_0_.shape[0], theta_1_.shape[0]))
    for i in range(theta_0_.shape[0]):
        for j in range(theta_1_.shape[0]):
            A_[i, j] = np.linalg.norm(theta_0_[i, :] - theta_1_[j, :])
    return A_

# Set the list adequately
def __mixture_model_list(X_1_, i, j):
    # Initialization
    X_ = [[], [], 0, X_1_[3], X_1_[4], [0, 1]]
    # Assigning correctly the clusters means
    X_[i]     = X_1_[j]
    X_[1 - i] = X_1_[1 - j]
    # Specify the number of clusters
    X_[2] = X_1_[2]
    return X_

# Shall I invert the assgined Labels?
def __invert_labels(i, j):
    # Only if there was a swaping in the means
    if i != j:
        return [1, 0]
    # Otherwise not to do it
    else:
        return [0, 1]

# Reshape vectors to matrix form
def __reshape(theta_, dim):
    # For a vector input
    if theta_.shape[0] == dim:
        return theta_[np.newaxis, :]
    # For a matrix input
    else:
        return theta_.reshape(2, dim)

def _persistent_GaMM(_tools, theta_, z_, X_, labels_, n_clusters, idx_):
    # Get parameters of each infer gamma distribution to identify cluster labels
    def __Ga_model_list(theta_, z_, X_, labels_, n_clusters):
        index_ = [0, 1]
        # if there was detected only a cluster
        if n_clusters == 1:
            mu_0_ = np.mean(X_, axis = 0)
            mu_1_ = []
        # if there were detected two clusters
        if n_clusters == 2:
            idx_0_ = labels_ == 0
            idx_1_ = labels_ == 1
            if idx_0_.sum() != 0. and idx_1_.sum() != 0:
                mu_0_ = np.mean(X_[idx_0_, :], axis = 0)
                mu_1_ = np.mean(X_[idx_1_ , :], axis = 0)
            else:
                mu_1_ = []
                n_clusters = 1
                if idx_0_.sum() == 0:
                    mu_0_ = np.mean(X_[idx_1_, :], axis = 0)
                    index_ = [1, 0]
                if idx_1_.sum() == 0:
                    mu_0_ = np.mean(X_[idx_0_, :], axis = 0)
        return [mu_0_, mu_1_, n_clusters, theta_, z_, index_]
    # Persitent cluster label indentification
    def __assign_GaMM_labels(Ga_0_, Ga_1_):
        # Index of the minama in the euclidiand distance matrix
        def ___find_index(A_):
            i_, j_ = np.where(A_ == A_.min())
            return i_[0], j_[0]
        # Where any cluster indentied previous frame?
        if Ga_0_[2] != 0:
            # Find Which Means are closser
            theta_0_ = __reshape(theta_ = np.hstack((Ga_0_[0], Ga_0_[1])), dim = 2)
            theta_1_ = __reshape(theta_ = np.hstack((Ga_1_[0], Ga_1_[1])), dim = 2)
            # Get Corresponding Index
            i, j = ___find_index(A_ = __euclidian_distane(theta_0_, theta_1_))
            # Fix index in case single cluster in previous list in index 1
            if len(Ga_0_[0]) == 0:
                i += 1
            # Set thetas in List format
            Ga_1_ = __mixture_model_list(Ga_1_, i, j)
            # Invert labels?
            Ga_1_[-1] = __invert_labels(i, j)
        # Return Muxture List and where to invert labels or not
        return Ga_1_, bool(Ga_1_[-1][0])
    # Identidy the label assign to each Ga cluster
    Ga_0_ = _tools.Ga_
    Ga_1_ = __Ga_model_list(theta_, z_, X_[idx_, :], labels_[idx_], n_clusters)
    _tools.Ga_, invert_labels = __assign_GaMM_labels(Ga_0_, Ga_1_)

    # Invert Ga-Mixture-model Labels?
    if invert_labels: labels_ = 1 - labels_
    return _tools, labels_

def _persistent_VMMM(_tools, theta_, z_, labels_, n_clusters):
    # Get parameters of each infer gamma distribution to identify cluster labels
    def __VM_model_list(theta_, z_, n_clusters):
        if n_clusters == 1:
            return [np.asarray(theta_[0])[np.newaxis], [], n_clusters, theta_, z_, [0, 1]]
        if n_clusters == 2:
            return [np.asarray(theta_[0])[np.newaxis], np.asarray(theta_[2])[np.newaxis], n_clusters, theta_, z_, [0, 1]]
    # Persitent cluster label indentification
    def __assign_VMMM_labels(VM_0_, VM_1_):
        # Rotate Means 180 degrees
        def ___rotate(theta_):
            theta_p_ = theta_.copy()
            for i in range(theta_.shape[0]):
                if theta_[i, :] < np.pi:
                    theta_p_[i, :] += 2*np.pi
            return theta_p_
        # Find Closses Mean by ratating them
        def ___find_index(A_, B_):
            # User rotate means or not
            if A_.min() <= B_.min():
                A_ = A_.copy()
            else:
                A_ = B_.copy()
            # Index of the minama in the euclidiand distance matrix
            i_, j_ = np.where(A_ == A_.min())
            return i_[0], j_[0]
        # Where any cluster indentied previous frame?
        if VM_0_[2] != 0:
            # Find Which Means are closser
            theta_0_ = __reshape(theta_ = np.hstack((VM_0_[0], VM_0_[1])), dim = 1)
            theta_1_ = __reshape(theta_ = np.hstack((VM_1_[0], VM_1_[1])), dim = 1)
            # Get Corresponding Index
            i, j = ___find_index(A_ = __euclidian_distane(theta_0_, theta_1_),
                                 B_ = __euclidian_distane(___rotate(theta_0_), ___rotate(theta_1_)))
            # Fix index in case single cluster in previous list in index 1
            if len(VM_0_[0]) == 0:
                i += 1
            # Set thetas in List format
            VM_1_ = __mixture_model_list(VM_1_, i, j)
            # Invert labels?
            VM_1_[-1] = __invert_labels(i, j)
        # Return Muxture List and where to invert labels or not
        return VM_1_, bool(VM_1_[-1][0])

    # Identidy the label assign to each VM cluster
    VM_0_ = _tools.VM_
    VM_1_ = __VM_model_list(theta_, z_, n_clusters)
    _tools.VM_, invert_labels = __assign_VMMM_labels(VM_0_, VM_1_)

    # Invert VM-Mixture-model Labels?
    if invert_labels: labels_ = 1 - labels_
    return _tools, labels_

# Filter Cloud Velocity Vectors to remove noise approximation the wind velocity field
def _cloud_velocity_vector_clustering(_tools, H_, F_, M_, I_norm_, I_segm_, beta, tau, display = False, verbose = False):
  # This Clustering Functions is temporal
    def __von_mises_mixture_model(_tools, X_, tau = 100., verbose = False):
        # Fit Von Mises Mixture model
        theta_1_, log_z_1 = _EM_NVmMM(X_, n_clusters = 1, n_init = 3, tol = .1, n_interations = 1000,
                                      w_init_ = __init_VM(_tools, X_, n_clusters = 1), verbose = verbose)
        theta_2_, log_z_2 = _EM_NVmMM(X_, n_clusters = 2, n_init = 3, tol = .1, n_interations = 1000,
                                      w_init_ = __init_VM(_tools, X_, n_clusters = 2), verbose = verbose)
        # Calculate BIC
        bic_1 = 2*log_z_1 - len(theta_1_)*np.log(X_.shape[0])
        bic_2 = 2*log_z_2 - len(theta_2_)*np.log(X_.shape[0])
        if True:
            print('>> Von Mises BIC: 1 Cluster = {} 2 Clusters = {}'.format(bic_1, bic_2))
        # Calculate first and second order differential
        diff = bic_1 - bic_2
        # Identify best Model by complete data log-likelihood
        if diff > - tau:
            return theta_1_, 1, [bic_1, bic_2]
        else:
            return theta_2_, 2, [bic_1, bic_2]

    # Fit 2D Gamma Distribution to the velocity vectors magnitudes
    def __2D_gamma_mixture_model(_tools, X_, n_clusters, verbose = False):
        # 2D-Gamma Misture Model fitting Parameters
        theta_1_, log_z_1 = _EM_2D_GaMM(X_, n_clusters = 1, n_init = 3, tol = .1, n_interations = 1000,
                                    w_init_ = __init_Ga(_tools, X_, n_clusters = 1), verbose = verbose)
        # 2D-Gamma Misture Model fitting Parameters
        theta_2_, log_z_2 = _EM_2D_GaMM(X_, n_clusters = 2, n_init = 3, tol = .1, n_interations = 1000,
                                    w_init_ = __init_Ga(_tools, X_, n_clusters = 2), verbose = verbose)
        # Calculate BIC
        bic_1 = 2*log_z_1 - len(theta_1_)*np.log(X_.shape[0])
        bic_2 = 2*log_z_2 - len(theta_2_)*np.log(X_.shape[0])
        if True:
            print('>> Gamma BIC: 1 Cluster = {} 2 Clusters = {}'.format(bic_1, bic_2))
        # if verbose:
        #     print('>> Ga. Likelihood = {} No. Clusters = {} BIC = {}'.format(log_z, n_clusters, bic))
        if n_clusters == 2 and bic_1 < bic_2:
            return theta_2_, 2, [bic_1, bic_2]
        else:
            return theta_1_, 1, [bic_1, bic_2]

    # Plot Mixture Models Density Function
    def _plot_density(X_, Y_, vm_theta_, ga_theta_, ga_labels_, n = 100, display = False):
        # Generate Grid of Data to sample the Density function of the Von Mises Mixture Model
        x_ = np.linspace(0., 2.*np.pi, n)
        # Sample the UVm-MM Density Function
        w_ = np.sum(_NVmMM(vm_theta_, x_)[0], axis = 1)
        if display:
            # Display UVm-MM Density Function
            plt.figure(figsize = (20, 5))
            plt.subplot(131)
            plt.title('Velocity Angle 1D-Von Mises Clustering with Noise', fontsize = 10)
            plt.hist(Y_, 25, color = 'blue', density = True, alpha = 0.5, label = 'Hist.')
            plt.plot(x_, w_, '--', color = 'red', label = 'PDF')
            plt.ylabel(r'$Prob.$', fontsize = 15)
            plt.xlabel(r'$ang( \vec v )$ [rad.]', fontsize = 15)
            plt.legend(loc = 'upper right', prop = {'size': 15})
            plt.grid()
        # Generate Grid of Data to sample the Density function of the 2D Gamma Mixture Model
        xx_, yy_ = np.meshgrid(np.linspace(1., 13., n), np.linspace(0., 2.5, n))
        xx_ = np.concatenate((xx_.flatten()[:, np.newaxis], yy_.flatten()[:, np.newaxis]), axis = 1)
        # Sample the 2D-Ga-MM Density Function
        ww_ = np.sum(_2D_GaMM(ga_theta_, xx_)[0], axis = 1)
        color_ = [['red', 'blue'][i] for i in ga_labels_]
        if display:
            # Display 2D GaMM Density Function
            plt.subplot(132)
            plt.title('Velocity Magnitude and Cloud Height 2D-Gamma Clustering', fontsize = 10)
            plt.scatter(X_[:, 0], X_[:, 1], c = color_, s = 5, label = 'Pixel')
            plt.contour(xx_[:, 0].reshape(n, n), xx_[:, 1].reshape(n, n), ww_.reshape(n, n), 100, alpha = .25, label = 'PDF')
            plt.xlabel(r'Height [km]', fontsize = 15)
            plt.ylabel(r'$mag( \vec v )$ [pixels/frame]', fontsize = 15)
            plt.legend(loc = 'upper right', prop = {'size': 15})
            plt.grid()
        if display:
            # Display 2D GaMM Density Function
            plt.subplot(133)
            plt.title('Velocity Angle and Cloud Height 2D-Gamma Clustering', fontsize = 10)
            plt.scatter(Y_[:, 0], X_[:, 0], c = color_, s = 5, label = 'Pixel')
            plt.xlabel(r'$ang( \vec v )$ [rad.]', fontsize = 15)
            plt.ylabel(r'Height [km]', fontsize = 15)
            plt.legend(loc = 'upper right', prop = {'size': 15})
            plt.grid()
        plt.show()
        return x_, w_, xx_, ww_, n

    # Get Datasets for the models
    def __get_datasets(H_, F_, I_segm_):
        I_bin_ = I_segm_.astype(bool)
        # Remove velocity Vectors where there are no clouds
        x_ = H_[I_bin_]
        # Velocity Vectors from Matrix to Vector
        v_ = np.concatenate((F_[I_bin_, 0].flatten()[:, np.newaxis], F_[I_bin_, 1].flatten()[:, np.newaxis]), axis = 1)
        # Trainsform Velocity Vectors to Polar Coordiantes
        m_, a_ = _cart_to_polar(v_[:, 0], v_[:, 1])
        # Get non zero common elements in each vector
        idx_x_ = x_ <= 0.
        idx_m_ = m_ == 0.
        # Get all non zero that are at least in one vector
        idx_zero_ = np.invert(idx_x_ | idx_m_)
        # Dataset For Height and Vectors Magnitude Clustering
        X_ = np.concatenate((x_[:, np.newaxis]/1000, m_[:, np.newaxis]), axis = 1)
        # Dataset For Vectors Angles Clustering
        Y_ = a_[:, np.newaxis] - 2*np.pi
        # Dataset for Markov Random Field Multi-Object Segmentation
        Z_ = np.concatenate((H_[..., np.newaxis]/1000., M_[..., np.newaxis]), axis = 2)
        return X_, Y_, Z_, idx_zero_

    # Assign Labels by 2D Mixture distribution
    def __assign_labels(labels_, I_segm_):
         # Variable Initialization
        Z_ = np.zeros((I_segm_.shape))
        # Assign labels to segmented index
        Z_[I_segm_] = labels_ + 1
        return Z_

    # Von Mises Mixture Model with noise
    def __NVmMM(theta_, X_, n_dist_theta = 3):
        n_clusters = int(theta_.shape[0]/n_dist_theta)
        # Sample Probabilities Initialization
        Z_ = np.zeros((X_.shape[0], n_clusters))
        # (1) Von Mises Distribution Probabilities
        for k in range(n_clusters):
            # Unpuck Cluster Parameters
            mu     = theta_[k:k + 1]
            kappa  = theta_[k + 1*n_clusters:k + 1*n_clusters + 1]
            weight = theta_[k + 2*n_clusters:k + 2*n_clusters + 1]
            # Evaluate Cluster Probabilities
            Z_[:, k] = weight * _VM(X_, mu, kappa)
        # Cluster label
        k_ = np.argmax(Z_, axis = 1)
        z_ = np.sum(Z_, axis = 0)
        return Z_, z_, k_

    # Gamma Mixture Model
    def __2D_GaMM(theta_, X_, n_dist_theta = 4):
        n_clusters = int(theta_.shape[0]/n_dist_theta)
        # Sample Probabilities Initialization
        Z_ = np.zeros((X_.shape[0], n_clusters))
        # (1) Gamma Distribution Probabilities
        for k in range(n_clusters):
            # Unpuck Cluster Parameters
            alpha  = theta_[k:k + 1]
            beta   = theta_[k + 1*n_clusters:k + 1*n_clusters + 1]
            kappa  = theta_[k + 2*n_clusters:k + 2*n_clusters + 1]
            weight = theta_[k + 3*n_clusters:k + 3*n_clusters + 1]
            # Evaluate Cluster Probabilities
            Z_[:, k] = weight * _2D_G(X_, alpha, beta, kappa)
        # Cluster label
        k_ = np.argmax(Z_, axis = 1)
        z_ = np.sum(Z_, axis = 0)
        return Z_, z_, k_

    # Multi-Class Markov Random Field Model with Mixture-Model Likelihood function
    def __MRF(Z_, W_init_, beta, idx_clique = 1, n_eval = 10, verbose = False):
        # Cliques Order
        cliques_0_ = [[0, 0]]
        cliques_1_ = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        cliques_2_ = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
        cliques_   = [cliques_0_, cliques_1_, cliques_2_]
        # Evaluate Prior Distribution
        def ___eval_prior_proba(W_, x_, y_, cliques_, M, N, beta = 0, gamma = 0, alpha = 0):
            # Prior based on neigborhood class
            def ____neigborhood(w, W_, i, j, cliques_, beta):
                M, N = W_.shape
                prior = 0
                # Loop over neigbors
                for clique_ in cliques_:
                    k = i + clique_[0]
                    m = j + clique_[1]
                    if k < 0 or m < 0 or k >= M or m >= N:
                        pass
                    else:
                        if 0 == W_[k, m]:
                            pass
                        else:
                            if w == W_[k, m]:
                                prior += beta
                            else:
                                prior -= beta
                return prior
            # Current Evaluation Weights Initialization
            prior_ = np.zeros((M, N, 2))
            # Loop over Pixels in an Image
            for i, j in zip(x_, y_):
                for k in range(2):
                    # Prior Probability
                    prior_[i, j, k] = ____neigborhood(k + 1, W_, i, j, cliques_, beta)
            return prior_[x_, y_, :]
        # Energy Potential Function
        def __eval_energy(lik_, prior_):
            return lik_ + prior_
        # Classification and Energy of the system
        def __pixel_classification(U_):
            return np.argmax(U_, axis = 1)
        # Compute the Total Energy of current Weights
        def __total_energy(U_, W_):
            return U_[W_ == 0, 0].sum() + U_[W_ == 1, 1].sum()

        # Model Parameters Definitio
        cliques_ = cliques_[idx_clique]
        # Constants Initialization
        M, N, N_layers = Z_.shape
        # Select only Pixels not labeled as background
        x_, y_ = np.where(W_init_ != 0)
        lik_ = Z_[x_, y_, :]
        # Stopping criteria Initialization
        u_k  = - np.inf
        # Initialization of output variables
        U_hat_ = np.zeros((M, N, 2))
        W_hat_ = np.zeros((M, N))
        # loop over evaluations
        for k in range(n_eval):
            # Evaluate Neiborhood Cliques
            prior_ = ___eval_prior_proba(W_init_, x_, y_, cliques_, M, N, beta)
            #print(prior_.shape)
            # Current Evaluation Total Energy
            U_ = __eval_energy(lik_, prior_)
            # Update Pixels Labels by maximum energy
            W_ = __pixel_classification(U_)
            # Calculate Total Energy of the pixels
            u_k_1 = __total_energy(U_, W_)
            # Stop if it is a minima
            if (u_k >= u_k_1) or np.isnan(u_k_1):
                break
            if verbose:
                print('>>> No iter.: {} Energy: {}'.format(k, u_k_1))
            # If not keep optimizing
            u_k = u_k_1.copy()
            W_hat_[x_, y_] = W_ + 1
            U_hat_[x_, y_] = U_
            W_init_ = W_hat_.copy()
        # Return Pixels labels and Energy
        return W_hat_, U_hat_
    # Compute softmax values for each sets of scores in x
    def __softmax(x_):
        z_ = np.exp(x_)
        return z_ / np.tile(np.sum(z_, axis = 1)[:, np.newaxis], (1, 2))
    # Compute the sum of the values to add to 1
    def __hardmax(x_):
        x_ = x_ - x_.min()
        return np.nan_to_num(x_ / np.tile(np.sum(x_, axis = 1)[:, np.newaxis], (1, 2)))
    # Get Velocity Vector probabilites per clsuter
    def __get_proba(X_, labels_, I_segm_, idx_, beta, verbose):
        I_bin_ = I_segm_.astype(bool)
        # Variables Initialization
        M, N = I_segm_.shape
        Z_ = np.zeros((M, N, 2))
        # Computer Log-Probabilities
        n, N_layers = X_.shape
        # Case that 1 wind flow layer is detected
        if N_layers == 1:
            Z_[I_bin_, idx_[0]] = 1.
            # Labels by Mixture distribution
            I_labels_ = __assign_labels(labels_, I_bin_)
            I_probs_  = Z_.copy()
        # Case that 2 wind flow layers are detected
        if N_layers == 2:
            for i, j in zip(idx_, [0, 1]):
                Z_[I_bin_, i] = np.nan_to_num(np.log(X_[..., j]))
            # MRF get probabilities by uniformization of th neiborhood
            I_labels_, I_probs_ = __MRF(Z_, I_segm_, beta, idx_clique = 1, n_eval = 25, verbose = verbose)
            # Probabilities Normalization
            #I_probs_ = __softmax(I_probs_.reshape(M*N, 2)).reshape(M, N, 2)
            I_probs_ = __hardmax(I_probs_.reshape(M*N, 2)).reshape(M, N, 2)
        return I_labels_, I_probs_

    # Variables Initialization
    M, N = I_segm_.shape
    I_labels_ = np.zeros((M, N))
    I_probs_  = np.zeros((M, N, 2))

    # Compute how many cloud pixels are in the image
    segmented_pixels = I_segm_.sum() > tau

    # If there are enough cloudy pixels on an image
    if segmented_pixels:

        # Get Datasets for the models
        X_, Y_, Z_, idx_zero_ = __get_datasets(H_, F_, I_segm_)

        # Inference of a Von Misen Mixture Model with Uniform Noisey
        vm_theta_, vm_n_clusters, vm_bic_ = __von_mises_mixture_model(_tools, Y_[idx_zero_, :], tau, verbose = verbose)

        # Evaluate Uniform Von-Mises Clustering Probabilities
        W_, w_, vm_labels_ = __NVmMM(vm_theta_, np.squeeze(Y_))

        # Inference of a 2D Gamma Mixture Model
        ga_theta_, ga_n_clusters, ga_bic_ = __2D_gamma_mixture_model(_tools, X_[idx_zero_, :],
                                                                     n_clusters = vm_n_clusters, verbose = verbose)

        # Evaluate Gamma Clustering Probabilities
        Q_, q_, ga_labels_ = __2D_GaMM(ga_theta_, X_)

        # Persistent Clustering Centroids Assignation
        _tools, ga_labels_ = _persistent_GaMM(_tools, ga_theta_, q_, X_, ga_labels_, ga_n_clusters, idx_zero_)
        #_tools, vm_labels_ = _persistent_VMMM(_tools, vm_theta_, w_, vm_labels_, vm_n_clusters)

        # Display the inferred Density Models
        display_ = _plot_density(X_[idx_zero_, :], Y_[idx_zero_, :], vm_theta_,
                                            ga_theta_, ga_labels_[idx_zero_], n = 100, display = display)

        # # Display Classification by unsupervise clustering
        # I_vm_labels_ = np.zeros(I_segm_.shape)
        # I_ga_labels_ = np.zeros(I_segm_.shape)
        # I_vm_labels_[I_segm_.astype(bool)] = vm_labels_ + 1
        # I_ga_labels_[I_segm_.astype(bool)] = ga_labels_ + 1

        # Get Probabilities by Mixture Model or MRF
        I_labels_, I_probs_ = __get_proba(Q_, ga_labels_, I_segm_, idx_ = _tools.Ga_[-1], beta = beta, verbose = verbose)

    # * if not segmented pixels remove clusters
    else:
        _tools.Ga_ = [[], [], 0, None, None, [0, 1]]
        _tools.VM_ = [[], [], 0, None, None, [0, 1]]
        vm_bic_  = [0., 0.]
        ga_bic_  = [0., 0.]
        display_ = [None, None, None, None, None]
    return _tools, I_labels_, I_probs_, [vm_bic_, ga_bic_], display_

# Filter Cloud Velocity Vectors to remove noise approximation the wind velocity field
def _cloud_velocity_vector_clustering_v2(_tools, H_, F_, M_, Z_, I_norm_, I_segm_, beta, tau, display = False, verbose = False):
  # This Clustering Functions is temporal
    def __von_mises_mixture_model(_tools, X_, tau = 100., verbose = False):
        # Fit Von Mises Mixture model
        theta_1_, log_z_1, scores_1_ = _EM_NVmMM(X_, n_clusters = 1, n_init = 4, tol = .01, n_interations = 1000, verbose = verbose)
        #print(scores_1_)
        theta_2_, log_z_2, scores_2_ = _EM_NVmMM(X_, n_clusters = 2, n_init = 4, tol = .01, n_interations = 1000, verbose = verbose)
        #print(scores_2_)
        # Calculate BIC
        #bic_1 = 2*log_z_1 - len(theta_1_)*np.log(X_.shape[0])
        #bic_2 = 2*log_z_2 - len(theta_2_)*np.log(X_.shape[0])
        score_1 = scores_1_[2]
        score_2 = scores_2_[2]
        if verbose:
            print('>> Von Mises Score: 1 Cluster = {} 2 Clusters = {} Rate = {}'.format(score_1, score_2, score_2/score_1))
        # Calculate first and second order differential
        #diff = bic_1 - bic_2
        # Identify best Model by complete data log-likelihood
        if score_1 > score_2:
            return theta_2_, [scores_1_, scores_2_], 2
        else:
            return theta_1_, [scores_1_, scores_2_], 1

    # Fit 2D Gamma Distribution to the velocity vectors magnitudes
    def __2D_gamma_mixture_model(_tools, X_, n_clusters, verbose = False):
        # 2D-Gamma Misture Model fitting Parameters
        theta_1_, log_z_1, scores_1_ = _EM_2D_GaMM(X_, n_clusters = 1, n_init = 4, tol = .01, n_interations = 1000, verbose = verbose)
        #print(scores_1_)
        # 2D-Gamma Misture Model fitting Parameters
        theta_2_, log_z_2, scores_2_ = _EM_2D_GaMM(X_, n_clusters = 2, n_init = 4, tol = .01, n_interations = 1000, verbose = verbose)
        #print(scores_2_)
        # Calculate BIC
        #bic_1 = 2*log_z_1 - len(theta_1_)*np.log(X_.shape[0])
        #bic_2 = 2*log_z_2 - len(theta_2_)*np.log(X_.shape[0])
        score_1 = scores_1_[2]
        score_2 = scores_2_[2]
        if verbose:
            print('>> 2D-Gamma Score: 1 Cluster = {} 2 Clusters = {} Rate = {}'.format(score_1, score_2, score_2/score_1))
        if score_1 > score_2:
            return theta_2_, [scores_1_, scores_2_], 2
        else:
            return theta_1_, [scores_1_, scores_2_], 1

    # Fit 2D Gamma Distribution to the velocity vectors magnitudes
    def __2DT_gamma_mixture_model(_tools, X_, Z_, n_clusters, verbose = False):
        # 2D-Gamma Misture Model fitting Parameters
        theta_1_, X_p_, log_z_1, scores_1_ = _EM_2DT_GaMM(X_, Z_, n_clusters = 1, n_init = 4, tol = .01, n_interations = 1000, verbose = False)
        #print(scores_1_)
        # 2D-Gamma Misture Model fitting Parameters
        theta_2_, X_p_, log_z_2, scores_2_ = _EM_2DT_GaMM(X_, Z_, n_clusters = 2, n_init = 4, tol = .01, n_interations = 1000, verbose = False)
        #print(scores_2_)
        # Calculate BIC
        #bic_1 = 2*log_z_1 - len(theta_1_)*np.log(X_.shape[0])
        #bic_2 = 2*log_z_2 - len(theta_2_)*np.log(X_.shape[0])
        score_1 = scores_1_[2]
        score_2 = scores_2_[2]
        if verbose:
            print('>> 2DT-Gamma BIC: 1 Cluster = {} 2 Clusters = {} Rate = {}'.format(score_1, score_2, score_2/score_1))
        if score_1 > score_2:
            return X_p_, theta_2_, [scores_1_, scores_2_], 2
        else:
            return X_p_, theta_1_, [scores_1_, scores_2_], 1

    # Fit 2D Gamma Distribution to the velocity vectors magnitudes
    def __beta_mixture_model(_tools, X_, n_clusters, verbose = False):
        # Gamma Misture Model fitting Parameters
        theta_1_, log_z_1, scores_1_ = _EM_BeMM(X_[:, np.newaxis], n_clusters = 1, n_init = 4, tol = .01, verbose = False)
        #print(scores_1_)
        # Gamma Misture Model fitting Parameters
        theta_2_, log_z_2, scores_2_= _EM_BeMM(X_[:, np.newaxis], n_clusters = 2, n_init = 4, tol = .01, verbose = False)
        #print(scores_2_)
        # Calculate BIC
        #bic_1 = 2*log_z_1 - len(theta_1_)*np.log(X_.shape[0])
        #bic_2 = 2*log_z_2 - len(theta_2_)*np.log(X_.shape[0])
        score_1 = scores_1_[2]
        score_2 = scores_2_[2]
        if verbose:
            print('>> Beta BIC: 1 Cluster = {} 2 Clusters = {} Rate = {}'.format(score_1, score_2, score_2/score_1))
        if score_1 > score_2:
            return theta_2_, [scores_1_, scores_2_], 2
        else:
            return theta_1_, [scores_1_, scores_2_], 1

    # Fit 2D Gamma Distribution to the velocity vectors magnitudes
    def __gamma_mixture_model(_tools, X_, n_clusters, verbose = False):
        # Gamma Misture Model fitting Parameters
        theta_1_, log_z_1, scores_1_ = _EM_GaMM(X_[:, np.newaxis], n_clusters = 1, n_init = 4, tol = .01, verbose = False)
        #print(scores_1_)
        # Gamma Misture Model fitting Parameters
        theta_2_, log_z_2, scores_2_ = _EM_GaMM(X_[:, np.newaxis], n_clusters = 2, n_init = 4, tol = .01, verbose = False)
        #print(scores_2_)
        # Calculate BIC
        #bic_1 = 2*log_z_1 - len(theta_1_)*np.log(X_.shape[0])
        #bic_2 = 2*log_z_2 - len(theta_2_)*np.log(X_.shape[0])
        score_1 = scores_1_[2]
        score_2 = scores_2_[2]
        if verbose:
            print('>> Gamma BIC: 1 Cluster = {} 2 Clusters = {} Rate = {}'.format(score_1, score_2, score_2/score_1))
        if score_1 > score_2:
            return theta_2_, [scores_1_, scores_2_], 2
        else:
            return theta_1_, [scores_1_, scores_2_], 1

    # Plot Mixture Models Density Function
    def _plot_density(X_, Y_, vm_theta_, ga_theta_, ga_labels_, q_, n = 100, display = False):
        # Generate Grid of Data to sample the Density function of the Von Mises Mixture Model
        x_ = np.linspace(0., 2.*np.pi, n)
        # Sample the UVm-MM Density Function
        w_ = np.sum(_NVmMM(vm_theta_, x_)[0], axis = 1)
        if display:
            # Display UVm-MM Density Function
            plt.figure(figsize = (20, 2.5))
            plt.subplot(131)
            plt.title('Velocity Angle 1D-Von Mises Clustering with Noise', fontsize = 10)
            plt.hist(Y_, 25, color = 'blue', density = True, alpha = 0.5, label = 'Hist.')
            plt.ylim(0, 1.2)
            plt.plot(x_, w_, '--', color = 'red', label = 'PDF')
            plt.ylabel(r'$Prob.$', fontsize = 15)
            plt.xlabel(r'$ang( \vec v )$ [rad.]', fontsize = 15)
            plt.legend(loc = 'upper right', prop = {'size': 15})
            plt.grid()
        # Generate Grid of Data to sample the Density function of the 2D Gamma Mixture Model
        xx_, yy_ = np.meshgrid(np.linspace(1., 12., n), np.linspace(0., 4, n))
        xx_ = np.concatenate((xx_.flatten()[:, np.newaxis], yy_.flatten()[:, np.newaxis]), axis = 1)
        # Sample the 2D-Ga-MM Density Function
        ww_ = np.sum(_2D_GaMM(ga_theta_, xx_)[0], axis = 1)
        color_ = [['red', 'blue'][i] for i in ga_labels_]
        if display:
            # Display 2D GaMM Density Function
            plt.subplot(132)
            plt.title('Velocity Magnitude and Cloud Height 2D-Gamma Clustering', fontsize = 10)
            plt.scatter(X_[:, 0], X_[:, 1], c = color_, s = 5, label = 'Pixel')
            #plt.colorbar()
            plt.contour(xx_[:, 0].reshape(n, n), xx_[:, 1].reshape(n, n), ww_.reshape(n, n), 100, alpha = .25, label = 'PDF')
            plt.xlabel(r'Height [km]', fontsize = 15)
            plt.ylabel(r'$mag( \vec v )$ [pixels/frame]', fontsize = 15)
            plt.legend(loc = 'upper right', prop = {'size': 15})
            plt.grid()
        if display:
            # Display 2D GaMM Density Function
            plt.subplot(133)
            plt.title('Velocity Angle and Cloud Height 2D-Gamma Clustering', fontsize = 10)
            plt.scatter(Y_[:, 0], X_[:, 0], c = color_, s = 5, label = 'Pixel')
            plt.xlabel(r'$ang( \vec v )$ [rad.]', fontsize = 15)
            plt.ylabel(r'Height [km]', fontsize = 15)
            plt.legend(loc = 'upper right', prop = {'size': 15})
            plt.grid()
        plt.show()
        return x_, w_, xx_, ww_, n

        # Plot Mixture Models Density Function
    def _plot_density_v1(X_, X_p_, be_theta_, ga_2d_theta_, ga_2dt_theta_, n = 100, display = False):

        # Generate Grid of Data to sample the Density function of the Von Mises Mixture Model
        x_ = np.linspace(0., 13., n)/14.
        # Sample the UVm-MM Density Function
        w_ = np.sum(__BeMM(be_theta_, x_)[0], axis = 1)

        # Display UVm-MM Density Function
        plt.figure(figsize = (20, 2.5))
        plt.subplot(131)
        plt.title('Height 1D-Beta Clustering', fontsize = 10)
        plt.hist(X_[:, 0]/14., 25, range = (0., 1.), color = 'blue', density = True, alpha = 0.5, label = 'Hist.')
        plt.ylim(0, 7.5)
        plt.plot(x_, w_, '--', color = 'red', label = 'PDF')
        plt.ylabel(r'$Prob.$', fontsize = 15)
        plt.xlabel(r'Height [km]', fontsize = 15)
        plt.legend(loc = 'upper right', prop = {'size': 15})
        plt.grid()

        # Generate Grid of Data to sample the Density function of the 2D Gamma Mixture Model
        xx_, yy_ = np.meshgrid(np.linspace(1., 13., n), np.linspace(0., 4, n))
        xx_ = np.concatenate((xx_.flatten()[:, np.newaxis], yy_.flatten()[:, np.newaxis]), axis = 1)
        # Sample the 2D-Ga-MM Density Function
        W_, ll, L_ = __2D_GaMM(ga_2d_theta_, xx_)
        ww_ = np.sum(W_, axis = 1)

        # Evaluate Gamma Clustering Probabilities
        color_ = [['red', 'blue'][i] for i in __2D_GaMM(ga_2d_theta_, X_)[2]]
        # Display 2D GaMM Density Function
        plt.subplot(132)
        plt.title('Velocity Magnitude and Cloud Height 2D-Gamma Clustering', fontsize = 10)
        plt.scatter(X_[:, 0], X_[:, 1], c = color_, s = 5, label = 'Pixel')
        #plt.colorbar()
        plt.contour(xx_[:, 0].reshape(n, n), xx_[:, 1].reshape(n, n), ww_.reshape(n, n), 100, alpha = .25, label = 'PDF')
        plt.xlabel(r'Height [km]', fontsize = 15)
        plt.ylabel(r'$mag( \vec v )$ [pixels/frame]', fontsize = 15)
        plt.legend(loc = 'upper right', prop = {'size': 15})
        plt.grid()

        # Generate Grid of Data to sample the Density function of the 2D Gamma Mixture Model
        xx_, yy_ = np.meshgrid(np.linspace(1., 13., n), np.linspace(0., 13., n))
        xxx_ = np.concatenate((xx_.flatten()[:, np.newaxis], yy_.flatten()[:, np.newaxis]), axis = 1)
        # Sample the 2D-Ga-MM Density Function
        W_, ll, L_ = __2D_GaMM(ga_2dt_theta_, xxx_)
        www_ = np.sum(W_, axis = 1)

        color_ = [['red', 'blue'][i] for i in __2D_GaMM(ga_2dt_theta_, X_p_)[2]]
        plt.subplot(133)
        plt.title('Velocity Magnitude and Cloud Height 2DT-Gamma Clustering', fontsize = 10)
        plt.scatter(X_p_[:, 0], X_p_[:, 1], c = color_, s = 5, label = 'Pixel')
        #plt.colorbar()
        plt.contour(xxx_[:, 0].reshape(n, n), xxx_[:, 1].reshape(n, n), www_.reshape(n, n), 100, alpha = .25, label = 'PDF')
        plt.xlabel(r'Height [km]', fontsize = 15)
        plt.ylabel(r'$mag( \vec v )$ [pixels/frame]', fontsize = 15)
        plt.legend(loc = 'upper right', prop = {'size': 15})
        plt.grid()
        plt.show()
        return x_, w_, xx_, ww_, n

    # Plot Mixture Models Density Function
    def _plot_density_v2(X_, Y_, vm_theta_, ga_0_theta_, ga_1_theta_, n = 100, display = False):
        # Generate Grid of Data to sample the Density function of the Von Mises Mixture Model
        x_ = np.linspace(0., 2.*np.pi, n)
        # Sample the UVm-MM Density Function
        w_ = np.sum(__NVmMM(vm_theta_, x_)[0], axis = 1)

        # Display UVm-MM Density Function
        plt.figure(figsize = (20, 2.5))
        plt.subplot(131)
        plt.title('Velocity Angle 1D-Von Mises Clustering with Noise', fontsize = 10)
        plt.hist(Y_, 25, range = (0., 2*np.pi), color = 'blue', density = True, alpha = 0.5, label = 'Hist.')
        plt.ylim(0, 2.)
        plt.plot(x_, w_, '--', color = 'red', label = 'PDF')
        plt.ylabel(r'$Prob.$', fontsize = 15)
        plt.xlabel(r'$ang( \vec v )$ [rad.]', fontsize = 15)
        plt.legend(loc = 'upper right', prop = {'size': 15})
        plt.grid()

        # Generate Grid of Data to sample the Density function of the Von Mises Mixture Model
        xx_ = np.linspace(0., 13., n)
        # Sample the UVm-MM Density Function
        ww_ = np.sum(__GaMM(ga_0_theta_, xx_)[0], axis = 1)

        plt.subplot(132)
        plt.title('Height 1D-Gamma Clustering', fontsize = 10)
        plt.hist(X_[..., 0], 25, range = (1., 13.), color = 'blue', density = True, alpha = 0.5, label = 'Hist.')
        plt.ylim(0, 2.)
        plt.plot(xx_, ww_, '--', color = 'red', label = 'PDF')
        plt.ylabel(r'$Prob.$', fontsize = 15)
        plt.xlabel(r'$Hegiht$ [km]', fontsize = 15)
        plt.legend(loc = 'upper right', prop = {'size': 15})
        plt.grid()

        xxx_ = np.linspace(0., 10., n)
        # Sample the UVm-MM Density Function
        www_ = np.sum(__GaMM(ga_1_theta_, xxx_)[0], axis = 1)

        plt.subplot(133)
        plt.title('Velocity Magnitude 1D-Gamma Clustering', fontsize = 10)
        plt.hist(X_[..., 1], 25, range = (0., 10.), color = 'blue', density = True, alpha = 0.5, label = 'Hist.')
        plt.ylim(0, 2.5)
        plt.plot(xxx_, www_, '--', color = 'red', label = 'PDF')
        plt.ylabel(r'$Prob.$', fontsize = 15)
        plt.xlabel(r'$mag( \vec v )$ [pixels/frame]', fontsize = 15)
        plt.legend(loc = 'upper right', prop = {'size': 15})
        plt.grid()
        plt.show()
        return x_, w_, xx_, ww_, xxx_, www_

    # Plot Mixture Models Density Function
    def _plot_density_v3(X_, Y_, Z_, vm_theta_, be_theta_, ga_theta_, n = 100, display = False):
        # Generate Grid of Data to sample the Density function of the Von Mises Mixture Model
        x_ = np.linspace(0., 2.*np.pi, n)
        # Sample the UVm-MM Density Function
        w_ = np.sum(__NVmMM(vm_theta_, x_)[0], axis = 1)

        # Display UVm-MM Density Function
        plt.figure(figsize = (20, 2.5))
        plt.subplot(131)
        #plt.title('Velocity Angle 1D-Von Mises Clustering with Noise', fontsize = 10)
        plt.hist(Y_, 25, range = (0., 2*np.pi), color = 'blue', density = True, alpha = 0.5, label = 'Hist.')
        plt.ylim(0, 2.)
        plt.plot(x_, w_, '--', color = 'red', label = 'PDF')
        plt.ylabel(r'$Prob.$', fontsize = 15)
        plt.xlabel(r'$ang( \vec v )$ [rad.]', fontsize = 15)
        plt.legend(loc = 'upper right', prop = {'size': 15})
        plt.grid()

        # Generate Grid of Data to sample the Density function of the Von Mises Mixture Model
        xx_ = np.linspace(0., 1., n)
        # Sample the UVm-MM Density Function
        ww_ = np.sum(__BeMM(be_theta_, xx_)[0], axis = 1)

        # Display UVm-MM Density Function
        plt.subplot(132)
        plt.title('Height 1D-Beta Clustering', fontsize = 10)
        plt.hist(Z_, 25, range = (0., 1.), color = 'blue', density = True, alpha = 0.5, label = 'Hist.')
        plt.ylim(0, 7.5)
        plt.plot(xx_, ww_, '--', color = 'red', label = 'PDF')
        plt.ylabel(r'$Prob.$', fontsize = 15)
        plt.xlabel(r'Height [rate]', fontsize = 15)
        plt.legend(loc = 'upper right', prop = {'size': 15})
        plt.grid()

        xxx_ = np.linspace(0., 100., n)
        # Sample the UVm-MM Density Function
        www_ = np.sum(__GaMM(ga_theta_, xxx_)[0], axis = 1)

        plt.subplot(133)
        plt.title('Velocity Magnitude 1D-Gamma Clustering', fontsize = 10)
        plt.hist(X_, 25, range = (0., 100.), color = 'blue', density = True, alpha = 0.5, label = 'Hist.')
        #plt.ylim(0, 2.5)
        plt.plot(xxx_, www_, '--', color = 'red', label = 'PDF')
        plt.ylabel(r'$Prob.$', fontsize = 15)
        plt.xlabel(r'$mag( \vec v )$ [m/s]', fontsize = 15)
        plt.legend(loc = 'upper right', prop = {'size': 15})
        plt.grid()
        plt.show()

        return x_, w_, xx_, ww_, xxx_, www_

    # Get Datasets for the models
    def __get_datasets(H_, F_, Z_, I_segm_):
        I_bin_ = I_segm_.astype(bool)
        # Remove velocity Vectors where there are no clouds
        x_ = H_[I_bin_]
        z_ = Z_[I_bin_]
        # Velocity Vectors from Matrix to Vector
        v_ = np.concatenate((F_[I_bin_, 0].flatten()[:, np.newaxis], F_[I_bin_, 1].flatten()[:, np.newaxis]), axis = 1)
        # Trainsform Velocity Vectors to Polar Coordiantes
        m_, a_ = _cart_to_polar(v_[:, 0], v_[:, 1])
        # Get non zero common elements in each vector
        idx_x_ = x_ <= 0.
        idx_m_ = m_ == 0.
        # Get all non zero that are at least in one vector
        idx_zero_ = np.invert(idx_x_ | idx_m_)
        # Dataset For Height and Vectors Magnitude Clustering
        X_ = np.concatenate((x_[:, np.newaxis]/1000., m_[:, np.newaxis]), axis = 1)
        # Dataset For Vectors Angles Clustering
        Y_ = 2*np.pi - a_[:, np.newaxis]
        # Dataset for Markov Random Field Multi-Object Segmentation
        Z_ = z_[:, np.newaxis]
        #Z_ = np.concatenate((H_[..., np.newaxis]/1000., M_[..., np.newaxis]), axis = 2)
        return X_, Y_, Z_, idx_zero_

    # Get Datasets for the models
    def __get_trans_datasets(H_, F_, Z_, I_segm_, k_, idx_):
        I_bin_ = I_segm_.astype(bool)
        # Remove velocity Vectors where there are no clouds
        dx_ = Z_[I_bin_, 0][idx_]
        dy_ = Z_[I_bin_, 1][idx_]
        dz_ = H_[I_bin_][idx_]
        # Velocity Vectors from Matrix to Vector
        v_ = np.concatenate((F_[I_bin_, 0][idx_][:, np.newaxis], F_[I_bin_, 1][idx_][:, np.newaxis]), axis = 1)
        for k in np.unique(k_):
            index_ = k_ == k
            height = np.median(dz_[index_])/1000.
            v_[index_, 0] = v_[index_, 0] * dx_[index_] * height / 15.
            v_[index_, 1] = v_[index_, 1] * dy_[index_] * height / 15.
        # Trainsform Velocity Vectors to Polar Coordiantes
        m_, a_ = _cart_to_polar(v_[:, 0], v_[:, 1])
        # Dataset For Height and Vectors Magnitude Clustering
        X_ = m_[:, np.newaxis]
        # Dataset For Vectors Angles Clustering
        Y_ = 2*np.pi - a_[:, np.newaxis]
        Z_ = dz_[:, np.newaxis]/dz_.max()
        return X_, Y_, Z_

    # Assign Labels by 2D Mixture distribution
    def __assign_labels(labels_, I_segm_):
         # Variable Initialization
        Z_ = np.zeros((I_segm_.shape))
        # Assign labels to segmented index
        Z_[I_segm_] = labels_ + 1
        return Z_

    # Von Mises Mixture Model with noise
    def __NVmMM(theta_, X_, n_dist_theta = 3):
        n_clusters = int(theta_.shape[0]/n_dist_theta)
        # Sample Probabilities Initialization
        Z_ = np.zeros((X_.shape[0], n_clusters))
        # (1) Von Mises Distribution Probabilities
        for k in range(n_clusters):
            # Unpuck Cluster Parameters
            mu     = theta_[k:k + 1]
            kappa  = theta_[k + 1*n_clusters:k + 1*n_clusters + 1]
            weight = theta_[k + 2*n_clusters:k + 2*n_clusters + 1]
            # Evaluate Cluster Probabilities
            Z_[:, k] = weight * _VM(X_, mu, kappa)
        # Cluster label
        k_ = np.argmax(Z_, axis = 1)
        z_ = np.sum(Z_, axis = 0)
        return Z_, z_, k_

    # Gamma Mixture Model
    def __2D_GaMM(theta_, X_, n_dist_theta = 4):
        n_clusters = int(theta_.shape[0]/n_dist_theta)
        # Sample Probabilities Initialization
        Z_ = np.zeros((X_.shape[0], n_clusters))
        # (1) Gamma Distribution Probabilities
        for k in range(n_clusters):
            # Unpuck Cluster Parameters
            alpha  = theta_[k:k + 1]
            beta   = theta_[k + 1*n_clusters:k + 1*n_clusters + 1]
            kappa  = theta_[k + 2*n_clusters:k + 2*n_clusters + 1]
            weight = theta_[k + 3*n_clusters:k + 3*n_clusters + 1]
            # Evaluate Cluster Probabilities
            Z_[:, k] = weight * _2D_G(X_, alpha, beta, kappa)
        # Cluster label
        k_ = np.argmax(Z_, axis = 1)
        z_ = np.sum(Z_, axis = 0)
        return Z_, z_, k_

    # Gamma Mixture Model
    def __GaMM(theta_, X_, n_dist_theta = 3):
        n_clusters = int(theta_.shape[0]/n_dist_theta)
        # Sample Probabilities Initialization
        Z_ = np.zeros((X_.shape[0], n_clusters))
        # (1) Gamma Distribution Probabilities
        for k in range(n_clusters):
            # Unpuck Cluster Parameters
            alpha  = theta_[k:k + 1]
            beta   = theta_[k + n_clusters:k + n_clusters + 1]
            weight = theta_[k + 2*n_clusters:k + 2*n_clusters + 1]
            # Evaluate Cluster Probabilities
            Z_[:, k] = weight * _1D_G(X_, alpha, beta)
        # Cluster label
        k_ = np.argmax(Z_, axis = 1)
        z_ = np.sum(Z_, axis = 0)
        return Z_, z_, k_

    # Beta Mixture Model Distribution
    def __BeMM(theta_, X_, n_dist_theta = 3):
        n_clusters = int(theta_.shape[0]/n_dist_theta)
        # Sample Probabilities Initialization
        Z_ = np.zeros((X_.shape[0], n_clusters))
        # (1) Beta Distribution Probabilities
        for k in range(n_clusters):
            # Unpuck Cluster Parameters
            alpha  = theta_[k:k + 1]
            beta   = theta_[k + n_clusters:k + n_clusters + 1]
            weight = theta_[k + 2*n_clusters:k + 2*n_clusters + 1]
            # Evaluate Cluster Probabilities
            Z_[:, k] = weight * _B(X_, alpha, beta)
        # Cluster label
        k_ = np.argmax(Z_, axis = 1)
        z_ = np.sum(Z_, axis = 0)
        return Z_, z_, k_

    # Multi-Class Markov Random Field Model with Mixture-Model Likelihood function
    def __MRF(Z_, W_init_, beta, idx_clique = 1, n_eval = 10, verbose = False):
        # Cliques Order
        cliques_0_ = [[0, 0]]
        cliques_1_ = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        cliques_2_ = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
        cliques_   = [cliques_0_, cliques_1_, cliques_2_]
        # Evaluate Prior Distribution
        def ___eval_prior_proba(W_, x_, y_, cliques_, M, N, beta = 0, gamma = 0, alpha = 0):
            # Prior based on neigborhood class
            def ____neigborhood(w, W_, i, j, cliques_, beta):
                M, N = W_.shape
                prior = 0
                # Loop over neigbors
                for clique_ in cliques_:
                    k = i + clique_[0]
                    m = j + clique_[1]
                    if k < 0 or m < 0 or k >= M or m >= N:
                        pass
                    else:
                        if 0 == W_[k, m]:
                            pass
                        else:
                            if w == W_[k, m]:
                                prior += beta
                            else:
                                prior -= beta
                return prior
            # Current Evaluation Weights Initialization
            prior_ = np.zeros((M, N, 2))
            # Loop over Pixels in an Image
            for i, j in zip(x_, y_):
                for k in range(2):
                    # Prior Probability
                    prior_[i, j, k] = ____neigborhood(k + 1, W_, i, j, cliques_, beta)
            return prior_[x_, y_, :]
        # Energy Potential Function
        def __eval_energy(lik_, prior_):
            return lik_ + prior_
        # Classification and Energy of the system
        def __pixel_classification(U_):
            return np.argmax(U_, axis = 1)
        # Compute the Total Energy of current Weights
        def __total_energy(U_, W_):
            return U_[W_ == 0, 0].sum() + U_[W_ == 1, 1].sum()

        # Model Parameters Definitio
        cliques_ = cliques_[idx_clique]
        # Constants Initialization
        M, N, N_layers = Z_.shape
        # Select only Pixels not labeled as background
        x_, y_ = np.where(W_init_ != 0)
        lik_ = Z_[x_, y_, :]
        # Stopping criteria Initialization
        u_k  = - np.inf
        # Initialization of output variables
        U_hat_ = np.zeros((M, N, 2))
        W_hat_ = np.zeros((M, N))
        # loop over evaluations
        for k in range(n_eval):
            # Evaluate Neiborhood Cliques
            prior_ = ___eval_prior_proba(W_init_, x_, y_, cliques_, M, N, beta)
            #print(prior_.shape)
            # Current Evaluation Total Energy
            U_ = __eval_energy(lik_, prior_)
            # Update Pixels Labels by maximum energy
            W_ = __pixel_classification(U_)
            # Calculate Total Energy of the pixels
            u_k_1 = __total_energy(U_, W_)
            # Stop if it is a minima
            if (u_k >= u_k_1) or np.isnan(u_k_1):
                break
            if verbose:
                print('>>> No iter.: {} Energy: {}'.format(k, u_k_1))
            # If not keep optimizing
            u_k = u_k_1.copy()
            W_hat_[x_, y_] = W_ + 1
            U_hat_[x_, y_] = U_
            W_init_ = W_hat_.copy()
        # Return Pixels labels and Energy
        return W_hat_, U_hat_
    # Compute softmax values for each sets of scores in x
    def __softmax(x_):
        z_ = np.exp(x_)
        return z_ / np.tile(np.sum(z_, axis = 1)[:, np.newaxis], (1, 2))
    # Compute the sum of the values to add to 1
    def __hardmax(x_):
        x_ = x_ - x_.min()
        return np.nan_to_num(x_ / np.tile(np.sum(x_, axis = 1)[:, np.newaxis], (1, 2)))
    # Get Velocity Vector probabilites per clsuter
    def __get_proba(X_, labels_, I_segm_, idx_, beta, verbose):
        I_bin_ = I_segm_.astype(bool)
        # Variables Initialization
        M, N = I_segm_.shape
        Z_ = np.zeros((M, N, 2))
        # Computer Log-Probabilities
        n, N_layers = X_.shape
        # Case that 1 wind flow layer is detected
        if N_layers == 1:
            Z_[I_bin_, idx_[0]] = 1.
            # Labels by Mixture distribution
            I_labels_ = __assign_labels(labels_, I_bin_)
            I_probs_  = Z_.copy()
        # Case that 2 wind flow layers are detected
        if N_layers == 2:
            for i, j in zip(idx_, [0, 1]):
                Z_[I_bin_, i] = np.nan_to_num(np.log(X_[..., j]))
            # MRF get probabilities by uniformization of th neiborhood
            I_labels_, I_probs_ = __MRF(Z_, I_segm_, beta, idx_clique = 1, n_eval = 25, verbose = True)
            # Probabilities Normalization
            #I_probs_[I_bin_, :] = __softmax(I_probs_[I_bin_, :])
            I_probs_[I_bin_, :] = __hardmax(I_probs_[I_bin_, :])
        return I_labels_, I_probs_

    def _add_scores(scores_, Scores_, idx):
        for i in range(len((scores_))):
            Scores_[idx, :, i] = scores_[i]
        return Scores_

    # Variables Initialization
    M, N = I_segm_.shape
    I_labels_ = np.zeros((M, N))
    I_probs_ = np.zeros((M, N, 2))
    Scores_  = np.zeros((9, 6, 2))

    # Compute how many cloud pixels are in the image
    segmented_pixels = I_segm_.sum() > tau

    # If there are enough cloudy pixels on an image
    if segmented_pixels:

        # Get Datasets for the models
        X_, Y_, Q_, idx_zero_ = __get_datasets(H_, F_, Z_[..., 2], I_segm_)

        # Inference of a Von Misen Mixture Model with Uniform Noisey
        vm_theta_, vm_scores_, vm_n_clusters = __von_mises_mixture_model(_tools, Y_[idx_zero_, :], tau, verbose = verbose)
        Scores_ = _add_scores(vm_scores_, Scores_, idx = 0)
        n_clusters = vm_n_clusters
        # Inference of a Gamma Mixture Model
        ga_0_theta_, ga_0_scores_, ga_0_n_clusters = __gamma_mixture_model(_tools, X_[idx_zero_, 0], n_clusters, verbose)
        Scores_ = _add_scores(ga_0_scores_, Scores_, idx = 1)
        # Inference of a Gamma Mixture Model
        ga_1_theta_, ga_1_scores_, ga_1_n_clusters = __gamma_mixture_model(_tools, X_[idx_zero_, 1], n_clusters, verbose)
        Scores_ = _add_scores(ga_1_scores_, Scores_, idx = 2)

        # Evaluate Uniform Von-Mises Clustering Probabilities
        #W_, w_, vm_labels_ = __NVmMM(vm_theta_, np.squeeze(Y_))

        # Get Probabilities by Mixture Model or MRF
        #I_labels_, I_probs_ = __get_proba(W_, vm_labels_, I_segm_, idx_ = _tools.Ga_[-1], beta = beta, verbose = verbose)

        #display_ = _plot_density_v2(X_, Y_, vm_theta_, ga_0_theta_, ga_1_theta_, n = 100, display = False)

        # Inference of a Gamma Mixture Model
        be_theta_, be_scores_, be_n_clusters = __beta_mixture_model(_tools, X_[idx_zero_, 0]/14., n_clusters, verbose)
        Scores_ = _add_scores(be_scores_, Scores_, idx = 3)
        # Inference of a 2D Gamma Mixture Model
        ga_2d_theta_, ga_2d_scores_, ga_2d_n_clusters = __2D_gamma_mixture_model(_tools, X_[idx_zero_, :], n_clusters, verbose)
        Scores_ = _add_scores(ga_2d_scores_, Scores_, idx = 4)
        # Inference of a 2D transformation Gamma Mixture Model
        X_p_, ga_2dt_theta_, ga_2dt_scores_ga_, ga_2dt_n_clusters = __2DT_gamma_mixture_model(_tools, X_[idx_zero_, :], Q_[idx_zero_, :],
                                                                                              n_clusters, verbose)
        Scores_ = _add_scores(ga_2dt_scores_ga_, Scores_, idx = 5)


        #display_ = _plot_density_v1(X_, X_p_, be_theta_, ga_2d_theta_, ga_2dt_theta_, n = 100, display = False)

        # Evaluate Gamma Clustering Probabilities
        Q_, q_, ga_labels_ = __GaMM(ga_0_theta_, X_[idx_zero_, 0])

        X_, Y_, Z_ = __get_trans_datasets(H_, F_, Z_, I_segm_, ga_labels_, idx_zero_)

        # Inference of a Von Misen Mixture Model with Uniform Noisey
        vm_theta_, vm_scores_, vm_n_clusters = __von_mises_mixture_model(_tools, Y_, tau, verbose)
        Scores_ = _add_scores(vm_scores_, Scores_, idx = 6)
        n_clusters = vm_n_clusters
        # Inference of a Gamma Mixture Model
        be_theta_, be_scores_, be_n_clusters = __beta_mixture_model(_tools, Z_[:, 0], n_clusters, verbose)
        Scores_ = _add_scores(be_scores_, Scores_, idx = 7)
        # Inference of a Gamma Mixture Model
        ga_theta_, ga_scores_, ga_n_clusters = __gamma_mixture_model(_tools, X_[:, 0], n_clusters, verbose)
        Scores_ = _add_scores(ga_scores_, Scores_, idx = 8)

        # Display the inferred Density Models
        #display_ = _plot_density_v3(X_, Y_, Z_, vm_theta_, be_theta_, ga_theta_, n = 100, display = False)

        # # Display Classification by unsupervise clustering
        # I_vm_labels_ = np.zeros(I_segm_.shape)
        # I_ga_labels_ = np.zeros(I_segm_.shape)
        # I_vm_labels_[I_segm_.astype(bool)] = vm_labels_ + 1
        # I_ga_labels_[I_segm_.astype(bool)] = ga_labels_ + 1

        # Evaluate Gamma Clustering Probabilities
#         Q_, q_, ga_2d_labels_ = __2D_GaMM(ga_2d_theta_, X_)

#         # Evaluate Gamma Clustering Probabilities
#         Q_1_, q_1_, be_labels_ = __BeMM(be_theta_, X_[..., 0])


        # Evaluate Gamma Clustering Probabilities
        #W_, w_, vm_labels_ = __NVmMM(vm_theta_, Y_[idx_zero_, :])

    # * if not segmented pixels remove clusters
    else:
        vm_bic_  = [0., 0.]
        ga_bic_  = [0., 0.]
        display_ = [None, None, None, None, None]
    return _tools, I_labels_, I_probs_, Scores_

# Transform velocity vectors velocities according to the the average hight of a layer label
def _transform_velocity_vectors(I_norm_1_, I_norm_2_, I_labels_, Q_, dXYZ_, h_, opt_,
                                n_layers, fps = 1./15.):
    # Variables Initialization
    F_ = np.zeros((I_norm_2_.shape[0], I_norm_2_.shape[1], 2))
    F_prime_ = np.zeros((I_norm_2_.shape[0], I_norm_2_.shape[1], 2))
    # Case of No Clouds
    if n_layers == 0:
        return F_, F_prime_
    # Case of Clouds in one layer
    if n_layers == 1:
        F_ = _lucas_kanade(I_norm_1_, I_norm_2_, window_size = opt_[0], tau = opt_[1], sigma = opt_[3])
        # Force to 0 velocity vectors where cloud pixels where not segmented
        F_[I_labels_ == 0, :] = 0
        # Prespective Transformation Weighted Sum
        F_prime_[..., 0] = F_[..., 0] * dXYZ_[..., 0] * h_[0][0] * fps
        F_prime_[..., 1] = F_[..., 1] * dXYZ_[..., 1] * h_[0][0] * fps
        return F_, F_prime_
    # Case of Clouds in two layers
    if n_layers == 2:
        # Compute Velocity vectors Transformation Layer 2
        F_0_ = _weighted_lucas_kanade(I_norm_1_, I_norm_2_, Q_[..., 0], window_size = opt_[0], tau = opt_[1], sigma = opt_[3])
        F_1_ = _weighted_lucas_kanade(I_norm_1_, I_norm_2_, Q_[..., 1], window_size = opt_[0], tau = opt_[1], sigma = opt_[3])
        # Force to 0 velocity vectors where cloud pixels where not segmented
        F_0_[I_labels_ == 0, :] = 0
        F_1_[I_labels_ == 0, :] = 0
        # Weight Velocity Vectors
        F_0_[..., 0] = F_0_[..., 0] * Q_[..., 0]
        F_0_[..., 1] = F_0_[..., 1] * Q_[..., 0]
        F_1_[..., 0] = F_1_[..., 0] * Q_[..., 1]
        F_1_[..., 1] = F_1_[..., 1] * Q_[..., 1]
        # Prespective Transformation Weighted Sum
        F_prime_[..., 0] = (F_0_[..., 0] * dXYZ_[..., 0] * h_[0][0] * fps) + (F_1_[..., 0] * dXYZ_[..., 0] * h_[0][1] * fps)
        F_prime_[..., 1] = (F_0_[..., 1] * dXYZ_[..., 1] * h_[0][0] * fps) + (F_1_[..., 1] * dXYZ_[..., 1] * h_[0][1] * fps)
        return F_0_ + F_1_, F_prime_

# # Compute velocity vectors velocities according to the the average hight of a layer label
# def _velocity_vectors(I_norm_1_, I_norm_2_, I_labels_, Q_, opt_, n_layers):
#     # Variables Initialization
#     F_ = np.zeros((I_norm_2_.shape[0], I_norm_2_.shape[1], 2))
#     # Case of No Clouds
#     if n_layers == 0:
#         return F_
#     # Case of Clouds in one layer
#     if n_layers == 1:
#         F_ = _pyramidal_weighted_lucas_kanade(I_norm_1_, I_norm_2_, I_labels_ > 0. + 0., window_size = opt_[0], tau = opt_[1],
#                                               n_pyramid = opt_[2], sigma = opt_[3])
#         # Force to 0 velocity vectors where cloud pixels where not segmented
#         F_[I_labels_ == 0, :] = 0
#         return F_
#     # Case of Clouds in two layers
#     if n_layers == 2:
#         # Compute Velocity vectors Transformation Layer 2
#         F_0_, F_1_ = _pyramidal_posterior_weighted_lucas_kanade(I_norm_1_, I_norm_2_, Q_, window_size = opt_[0], tau = opt_[1],
#                                                                 n_pyramid = opt_[2], sigma = opt_[3])
#         # Force to 0 velocity vectors where cloud pixels where not segmented
#         F_0_[I_labels_ == 0, :] = 0
#         F_1_[I_labels_ == 0, :] = 0
#         # Weight Velocity Vectors
#         F_0_[..., 0] = F_0_[..., 0] * Q_[..., 0]
#         F_0_[..., 1] = F_0_[..., 1] * Q_[..., 0]
#         F_1_[..., 0] = F_1_[..., 0] * Q_[..., 1]
#         F_1_[..., 1] = F_1_[..., 1] * Q_[..., 1]
#         return F_0_ + F_1_

# Transform velocity vectors velocities according to the the average hight of a layer label
def _velocity_vectors(I_norm_1_, I_norm_2_, I_labels_, Q_, dXYZ_, h_, opt_, n_layers, fps = 1./15.):
    # Variables Initialization
    F_ = np.zeros((I_norm_2_.shape[0], I_norm_2_.shape[1], 2))
    F_prime_ = np.zeros((I_norm_2_.shape[0], I_norm_2_.shape[1], 2))
    # Case of No Clouds
    if n_layers == 0:
        return F_
    # Case of Clouds in one layer
    if n_layers == 1:
        F_ = _pyramidal_weighted_lucas_kanade(I_norm_1_, I_norm_2_, I_labels_ > 0. + 0.,
                                              window_size = opt_[0], tau = opt_[1],
                                              n_pyramid = opt_[2], sigma = opt_[3])
        # Force to 0 velocity vectors where cloud pixels where not segmented
        F_[I_labels_ == 0, :] = 0
        # Prespective Transformation Weighted Sum
        F_prime_[..., 0] = F_[..., 0] * dXYZ_[..., 0] * h_[0][0] * fps
        F_prime_[..., 1] = F_[..., 1] * dXYZ_[..., 1] * h_[0][0] * fps
        return F_
    # Case of Clouds in two layers
    if n_layers == 2:
        # Compute Velocity vectors Transformation Layer 2
        F_0_ = _pyramidal_weighted_lucas_kanade(I_norm_1_, I_norm_2_, Q_[..., 0],
                                                window_size = opt_[0], tau = opt_[1],
                                                n_pyramid = opt_[2], sigma = opt_[3])
        F_1_ = _pyramidal_weighted_lucas_kanade(I_norm_1_, I_norm_2_, Q_[..., 1],
                                                window_size = opt_[0], tau = opt_[1],
                                                n_pyramid = opt_[2], sigma = opt_[3])
        # Force to 0 velocity vectors where cloud pixels where not segmented
        F_0_[I_labels_ == 0, :] = 0
        F_1_[I_labels_ == 0, :] = 0
        # Weight Velocity Vectors
        F_0_[..., 0] = F_0_[..., 0] * Q_[..., 0]
        F_0_[..., 1] = F_0_[..., 1] * Q_[..., 0]
        F_1_[..., 0] = F_1_[..., 0] * Q_[..., 1]
        F_1_[..., 1] = F_1_[..., 1] * Q_[..., 1]
        # Prespective Transformation Weighted Sum
        F_prime_[..., 0] = (F_0_[..., 0] * dXYZ_[..., 0] * h_[0][0] * fps) + (F_1_[..., 0] * dXYZ_[..., 0] * h_[0][1] * fps)
        F_prime_[..., 1] = (F_0_[..., 1] * dXYZ_[..., 1] * h_[0][0] * fps) + (F_1_[..., 1] * dXYZ_[..., 1] * h_[0][1] * fps)
        return F_0_ + F_1_


__all__ = ['_cloud_velocity_vector_fb', '_cloud_velocity_vector_lk', '_cloud_velocity_vector_comparison', '_velocity_vectors',
           '_cloud_velocity_vector_average', '_cloud_velocity_vector_selection_v1', '_cloud_velocity_vector_selection_v2',
           '_cloud_velocity_vector_selection_v3', '_cloud_label_height', '_cloud_velocity_vector_wlk',
           '_cloud_velocity_vector_clustering', '_cloud_velocity_vector_hs', '_cloud_velocity_vector_clustering_v2',
           '_cloud_velocity_vector_average_v2', '_autoregressive_dataset', '_LK_cloud_velocity_vector', '_transform_velocity_vectors']
