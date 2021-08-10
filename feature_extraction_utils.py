import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.interpolate import interp1d
from scipy.stats import norm, multivariate_normal
from scipy.interpolate import griddata
from scipy.ndimage.filters import median_filter, gaussian_filter
from scipy.ndimage.morphology import binary_fill_holes

#from skimage.measure import label
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing

from datetime import datetime

from detection_clustering_utils import *

from scipy.stats import norm as _N
from scipy.stats import gamma as _Ga
from scipy.stats import vonmises as _Vm
from scipy.stats import beta as _Be
from scipy.stats import multivariate_normal

# Caculate potential lines function
def _potential_lines(u, v): return .5*( np.cumsum(u, axis = 1) + np.cumsum(v, axis = 0) )

# Caculate stramelines function
def _streamlines(u, v): return .5*( np.cumsum(u, axis = 0) - np.cumsum(v, axis = 1) )

# Calculate vorticity approximating hte veloctiy gradient by numerical diffenciation
def _vorticity(u, v): return np.gradient(u)[1] - np.gradient(v)[0]

# Calculate divergence approximating hte veloctiy gradient by numerical diffenciation
def _divergence(u, v): return np.gradient(u)[1] + np.gradient(v)[0]

# Calculate the magnitude of a vector
def _magnitude(u, v): return np.sqrt(u**2 + v**2)

# Clouds velocity field features extraction, and remove pixels with no clouds
def _cloud_features(F_, M_, I_segm_):
    # Remove Selected Pixels
    F_[~I_segm_.astype(bool), :] = 0.
    # Calculate Divergence and Vorticity
    M_ = _magnitude( F_[..., 0], F_[..., 1])
    D_ = _divergence(F_[..., 0], F_[..., 1])
    V_ = _vorticity( F_[..., 0], F_[..., 1])
    # Return features
    return F_, M_, D_, V_

def _hardmax(x_):
    x_ = x_ - x_.min()
    q_ = np.sum(x_, axis = 2)[..., np.newaxis]
    return np.nan_to_num(x_ / np.concatenate((q_, q_), axis = 2))

# Unform Image by Field Implementation
def _multiclass_ising_model(W_, cliques, beta, n_max_iter = 10):
    # Cliques
    cliques_1_ = [[ 0,  1], [ 0, -1], [1,  0], [-1, 0]]
    cliques_2_ = [[-1, -1], [-1,  1], [1, -1], [ 1, 1]]
    cliques_3_ = [[ 0,  2], [ 0, -2], [2,  0], [-2, 0]]
    cliques_4_ = [[-2, -2], [-2,  2], [2, -2], [ 2, 2]]
    cliques_   = [cliques_1_, cliques_1_ + cliques_2_,
                  cliques_1_ + cliques_2_ + cliques_3_,
                  cliques_1_ + cliques_2_ + cliques_3_ + cliques_4_][cliques]
    # Prior based on neigborhood class
    def __neigborhood(W_, i, j, labels_, cliques_, beta):
        M, N = W_.shape
        class_0_ = 0
        class_1_ = 0
        class_2_ = 0
        class_ = [class_0_, class_1_, class_2_]
        # Loop over neigbors
        for clique_ in cliques_:
            k = i + clique_[0]
            m = j + clique_[1]
            if k < 0 or m < 0 or k >= M or m >= N:
                continue
            else:
                for l in labels_:
                    if W_[k, m] == l:
                        class_[l] += 1
        i_class_ = class_ == np.max(class_)
        if i_class_.sum() < 2:
            return labels_[i_class_]
        else:
            if (labels_[i_class_] == W_[i, j]).any():
                return W_[i, j]
            if np.random.rand() <= 0.5:
                return labels_[i_class_][0]
            else:
                return labels_[i_class_][1]
    # Constants Init.
    D, N = W_.shape
    Y_hat_ = - np.ones(W_.shape)
    # Variables Init.
    W_init_ = W_.copy()
    W_hat_  = np.zeros(W_.shape)
    e_prev  = np.inf
    labels_ = np.unique(W_)
    # Loop until converge
    while True:
        k = 0
        # Loop Over Pixels in an image
        for i in range(D):
            for j in range(N):
                W_hat_[i, j] = __neigborhood(W_init_, i, j, labels_, cliques_, beta)
        # Compute Error
        e_now = np.sum(np.sqrt((W_init_ - W_hat_)**2))
        # Stop if Convergence or reach max. interations
        if e_now >= e_prev or k == n_max_iter:
            break
        else:
            # Update for the next interaation
            W_init_ = W_hat_.copy()
            e_prev  = e_now.copy()
            k += 0
        return W_init_

# Unform Image by Field Implementation
def _ising_model(W_, cliques, beta, n_max_iter = 10):
    # Cliques
    cliques_1_ = [[ 0,  1], [ 0, -1], [1,  0], [-1, 0]]
    cliques_2_ = [[-1, -1], [-1,  1], [1, -1], [ 1, 1]]
    cliques_3_ = [[ 0,  2], [ 0, -2], [2,  0], [-2, 0]]
    cliques_4_ = [[-2, -2], [-2,  2], [2, -2], [ 2, 2]]
    cliques_   = [cliques_1_, cliques_1_ + cliques_2_,
                  cliques_1_ + cliques_2_ + cliques_3_,
                  cliques_1_ + cliques_2_ + cliques_3_ + cliques_4_][cliques]
    # Prior based on neigborhood class
    def __neigborhood(W_, i, j, cliques_, beta):
        M, N = W_.shape
        total = 0
        # Loop over neigbors
        for clique_ in cliques_:
            k = i + clique_[0]
            m = j + clique_[1]
            if k < 0 or m < 0 or k >= M or m >= N:
                continue
            else:
                if W_[k, m] == W_[i, j]:
                    total += 1
                else:
                    total -= 1
        if total >= 0.:
            return W_[i, j]
        else:
            return 1 - W_[i, j]
    # Constants Init.
    D, N = W_.shape
    Y_hat_ = - np.ones(W_.shape)
    # Variables Init.
    W_init_ = W_.copy()
    W_hat_  = np.zeros(W_.shape)
    e_prev  = np.inf
    # Loop until converge
    while True:
        k = 0
        # Loop Over Pixels in an image
        for i in range(D):
            for j in range(N):
                W_hat_[i, j] = __neigborhood(W_init_, i, j, cliques_, beta)
        # Compute Error
        e_now = np.sum(np.sqrt((W_init_ - W_hat_)**2))
        # Stop if Convergence or reach max. interations
        if e_now >= e_prev or k == n_max_iter:
            break
        else:
            # Update for the next interaation
            W_init_ = W_hat_.copy()
            e_prev  = e_now.copy()
            k += 0
        return W_init_

# Calculate the average, meadian, and standard deviation of the height of a labelled object
def _cloud_label_height(H_, I_labels_, m_to_km = 1000., verbose = False):
    # Find Labels
    labels_ = np.unique(I_labels_)
    index_  = np.nonzero(labels_)[0]
    labels_ = labels_[index_]
    # Variables Initialization
    h_mean_ = np.zeros(labels_.shape[0])
    h_medi_ = np.zeros(labels_.shape[0])
    h_std_  = np.zeros(labels_.shape[0])
    # Loop over labels
    for i, j in zip(labels_, range(labels_.shape[0])):
        # Calculate Statistics
        h_mean_[j] = np.mean(H_[I_labels_ == i])/m_to_km
        h_medi_[j] = np.median(H_[I_labels_ == i])/m_to_km
        h_std_[j]  = np.std(H_[I_labels_ == i])/m_to_km
        if verbose:
            print('>> Label: {} Avg. H.: {} km Median H.: {} km Std. H.: {} km'.format(i,
                                                        h_mean_[j], h_medi_[j], h_std_[j]))
    return h_mean_, h_medi_, h_std_

# Labels ordered from highest to lowers
def _sort_cloud_height(I_labels_, H_, W_):
    # Calculate Average Height of the clouds in each Pixel
    h_ = _cloud_label_height(H_, I_labels_, verbose = False)
    # Keep Lowest Cloud Layer in index 0
    if len(h_[0]) > 1:
        if h_[0][1] > h_[0][0]:
            # Invert Probabilities
            Q_ = W_.copy()
            W_[..., 0] = Q_[..., 1]
            W_[..., 1] = Q_[..., 0]
            # Invert Labels
            Z_ = I_labels_.copy()
            I_labels_[Z_ == 1] = 2
            I_labels_[Z_ == 2] = 1
            # Get ordered heights
            h_ = _cloud_label_height(H_, I_labels_, verbose = False)
        W_[I_labels_ == 0, 0] = 0
        W_[I_labels_ == 0, 1] = 0
    else:
        W_[I_labels_ == 1, 0] = 1

    return W_, I_labels_, h_

# Apply Ising model to fine tune segmentation
def _label_clouds(I_labels_, I_segm_):
    # Set I_segm_2_ at the top if were detected
    idx_ = I_segm_ == 0
    if idx_.sum() > 0:
        I_labels_[idx_] = 0
    # How many labels are in the image?
    labels_ = np.unique(I_labels_)
    N_labels = labels_.shape[0]
    # Apply issing model if there is more than one layer
    if N_labels == 1:
        return I_labels_
    # Ising Model
    elif N_labels == 2:
        # Is it backgroun in the image?
        if labels_.sum() == 3:
            return _ising_model(I_labels_ - 1, cliques = 2, beta = 1, n_max_iter = 10) + 1
        else:
            return _ising_model(I_labels_, cliques = 2, beta = 1, n_max_iter = 10)
    # Multiclass Ising Model
    elif N_labels == 3:
        return _multiclass_ising_model(I_labels_, cliques = 2, beta = 1, n_max_iter = 10)

# Evaluate Detection Metric
def _predict_cloud_layer(_tools, temp_scores_1_, temp_scores_2_, ang_scores_1_, ang_scores_2_, mag_scores_1_, mag_scores_2_,
                         metric, method, epsilon_0, epsilon_1, beta):
    # Get Bayesian Metrics
    def __get_scores(temp_scores_1_, temp_scores_2_, ang_scores_1_, ang_scores_2_, mag_scores_1_, mag_scores_2_):
        # Total for each No. of clusters
        if method == 0:
            scores_1_ = np.asarray(temp_scores_1_)
            scores_2_ = np.asarray(temp_scores_2_)
        if method == 1:
            scores_1_ = np.asarray(ang_scores_1_)
            scores_2_ = np.asarray(ang_scores_2_)
        if method == 2:
            scores_1_ = np.asarray(mag_scores_1_)
            scores_2_ = np.asarray(mag_scores_2_)
        if method == 3:
            scores_1_ = np.asarray(temp_scores_1_) + np.asarray(ang_scores_1_)
            scores_2_ = np.asarray(temp_scores_2_) + np.asarray(ang_scores_2_)
        if method == 4:
            scores_1_ = np.asarray(temp_scores_1_) + np.asarray(mag_scores_1_)
            scores_2_ = np.asarray(temp_scores_2_) + np.asarray(mag_scores_2_)
        if method == 5:
            scores_1_ = np.asarray(mag_scores_1_) + np.asarray(ang_scores_1_)
            scores_2_ = np.asarray(mag_scores_2_) + np.asarray(ang_scores_2_)
        if method == 6:
            scores_1_ = np.asarray(temp_scores_1_) + np.asarray(ang_scores_1_) + np.asarray(mag_scores_1_)
            scores_2_ = np.asarray(temp_scores_2_) + np.asarray(ang_scores_2_) + np.asarray(mag_scores_2_)
        return scores_1_, scores_2_
    # Implementation of the Markov Prior
    def __Markov_Process(_tools, score_1, score_2):
        y_prev = _tools.y_prev_[-1]
        # Markov Prior
        if y_prev == 1:
            score_1 += beta
            score_2 -= beta
        if y_prev == 2:
            score_1 -= beta
            score_2 += beta
        return score_1, score_2
    # Implementation of Persistent Classificaiton
    def __Persitente(_tools, lag = 3):
        s_ = _tools.y_prev_[-lag:]
        if len(s_) < lag:
            _tools.y_hat_[_tools.j] = _tools.y_prev_[-1]
        else:
            y_hat, count = mode(s_, axis = 0)
            _tools.y_hat_[_tools.j] = y_hat[0]
        return _tools
    # Maximum likelihood dectection criteria
    def __MAP(_tools, score_1, score_2, metric):
        # Maximum Likelihood Criteria
        if metric == 1:
            if score_1 > score_2:
                _tools.y_hat_[_tools.j] = 1
            else:
                _tools.y_hat_[_tools.j] = 2
        # Minimum Likelihood Criteria
        else:
            if score_1 < score_2:
                _tools.y_hat_[_tools.j] = 1
            else:
                _tools.y_hat_[_tools.j] = 2
        # Save Class according to Criteria
        _tools.y_prev_.append(_tools.y_hat_[_tools.j])
        return _tools

    print('Sample: {}'.format(_tools.j))
    # Total for each No. of clusters
    scores_1_, scores_2_ = __get_scores(temp_scores_1_, temp_scores_2_, ang_scores_1_, ang_scores_2_, mag_scores_1_, mag_scores_2_)
    # H, LL, BIC, AIC, CLC, ICL
    print('>>> Bayesian Metrics: ')
    # print('log p(T|w, z = 1) = {}'.format(temp_scores_1_[metric]))
    # print('log p(T|w, z = 2) = {}'.format(temp_scores_2_[metric]))
    # print('log p(A|b, z = 1) = {}'.format(ang_scores_1_[metric]))
    # print('log p(A|b, z = 2) = {}'.format(ang_scores_2_[metric]))
    score_1 = scores_1_[metric]
    score_2 = scores_2_[metric]
    print('log p(T, A, M|w, b, z = 1) = {}'.format(score_1))
    print('log p(T, A, M|w, z = 2) = {} '.format(score_2))
    # Markov Process Posterior
    score_1, score_2 = __Markov_Process(_tools, score_1, score_2)
    print('log p(T, A, M|w, b, z = 1) + log p (z = 1) = {}'.format(score_1))
    print('log p(T, A, M|w, z = 2) + log p (z = 2) = {} '.format(score_2))
    # Compute MAP for each metric
    _tools = __MAP(_tools, score_1, score_2, metric)
    # Persistent Predicted Class
    #_tools = __Persitente(_tools, lag = 3)
    return _tools

__all__ = ['_potential_lines', '_streamlines', '_vorticity', '_divergence', '_magnitude',
           '_label_clouds', '_sort_cloud_height', '_ising_model', '_predict_cloud_layer',
           '_multiclass_ising_model', '_hardmax', '_cloud_label_height']
