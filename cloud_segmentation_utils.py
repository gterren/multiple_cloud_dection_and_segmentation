import numpy as np
from utils import *
from scipy.stats import gamma, mode, multivariate_normal
from scipy.ndimage.filters import median_filter
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt


def _cloud_segmentation(label, I_norm_, I_scatter_, K_, H_, M_,
                        m_ = None, idx_var_ = None, idx_shape_ = None):
    # Segmentation iff clouds on the image
    if label == 0 or label == 1:
        # Parameters - Temperature, Scatter Radiation, Normalized Pixels Intensity,
        # Hegiht, Velocity Magnitude
        X_ = np.concatenate((K_[..., np.newaxis], I_scatter_[..., np.newaxis],
                             I_norm_[..., np.newaxis], H_[..., np.newaxis],
                             M_[..., np.newaxis]), axis = 2)
        return _cloud_ICM_segmentation(X_, m_, idx_var_, idx_shape_)
    if label == 2 or label == 3:
        return np.ones(I_norm_.shape)

# Threasholding Segmentation
def _cloud_THR_segmentation(I_norm_, tau = 12.5):
    # Segmentation according to each pixels amospheric distance
    def __threshold_segmentation(I_norm_, tau):
        return I_norm_ > tau
    return __threshold_segmentation(I_norm_, tau).astype(bool)

def _get_data(X_, _vars, _shape, degree = 0):
    # Image padding with madian filter
    def __median_padding(x_, M, N, D):
        I_ = np.zeros((M + 2, N + 2, D))
        # Zero-pad image
        I_[1:-1,1:-1, ...] = x_
        # loop over dimension
        for j in range(D):
            # Aply median filter
            I_prime_ = median_filter(I_[..., j], size = 7, mode = 'reflect')
            # Median fileter Papped Image
            I_[0, :, j]  = I_prime_[0, :]
            I_[-1, :, j] = I_prime_[-1, :]
            I_[:, 0, j]  = I_prime_[:, 0]
            I_[:, -1, j] = I_prime_[:, -1]
        return I_
    # Data Polynomial Expansion
    def __polynomial(X_, degree):
        return PolynomialFeatures(degree).fit_transform(X_)
    # Pixels Structures
    idx_0_ = np.matrix([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype = bool)
    idx_1_ = np.matrix([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype = bool)
    idx_2_ = np.matrix([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype = bool)
    str_idx_ = [idx_0_, idx_1_, idx_2_][_shape]
    # Variables Set
    var_idx_ = [[3], [0, 3], [3, 4], [3, 5], [0, 3, 4], [0, 3, 5], [0, 3, 4, 5]][_vars]
    # Select Variables
    x_ = X_[:, :, var_idx_].copy()
    # Variables Initialization
    M, N, D = x_.shape
    # Image padding with madian filter
    I_ = __median_padding(x_, M, N, D)
    # Variables Initialization
    xx_ = np.zeros((0, len(var_idx_)*np.sum(str_idx_)))
    # Loop over x-axis Pixels
    for i in range(1, M + 1):
        # Loop over y-axis Pixels
        for j in range(1, N + 1):
            W_  = I_[i-1:i+2, j-1:j+2, :]
            x_  = W_[str_idx_, :].flatten()[np.newaxis, ...]
            xx_ = np.concatenate((xx_, x_), axis = 0)
    # Perform Polynomial Expanssion iff need
    if degree > 0:
        return __polynomial(xx_, degree)
    else:
        return xx_

# Maximum Likelihood classification
def _maximum_likelihood(X_, model_):
    # Evaluate Likelyhood
    def __likelihood(X_, _N_0, _N_1):
        return _N_0.logpdf(X_), _N_1.logpdf(X_)
    # Classify by maximum likelihood
    def __classify(W_hat_, lik_):
        W_hat_[lik_[0] < lik_[1]] = 1
        return W_hat_

    cliques_, beta, _N_0, _N_1 = model_
    # Variable Initialization
    M, D   = X_.shape
    W_hat_ = np.zeros((M))
    return __classify(W_hat_, lik_ = __likelihood(X_, _N_0, _N_1))

# Maximum Likelihood Classification
def _classify(Z_, prob, invert_label = False):
    # Variables Initizalization
    labels_ = np.zeros((Z_.shape[0]))
    # Maximum Likelihood Classification
    idx_ = Z_[:, 0] * prob < Z_[:, 1] * (1. - prob)
    labels_[idx_] = 1.
    # Invert Labels iff need
    if invert_label:
        return 1. - labels_.reshape(60, 80)
    else:
        return labels_.reshape(60, 80)

def _cloud_NBC_segmentation(X_, m_, idx_var_, idx_shape_):
    def __predict__probrabilities(X_, m_):
        _NBC = m_
        return _NBC.predict_log_proba(X_)
    # Model and Classification threshold
    model_, prob = m_
    # Get Dataset Variables
    X_ = _get_data(X_, idx_var_, idx_shape_)
    # Predict Classification Probabilities
    Z_ = __predict__probrabilities(X_, model_)
    # Maximum Likelihood Classification
    return _classify(Z_, prob).astype(bool)

def _cloud_GC_segmentation(X_, m_, idx_var_, idx_shape_):
    def __predict_probrabilities(X_, m_):
        def ___predict_log_proba(X_, _N_0, _N_1, prior_0_, prior_1_):
            Z_hat_ = np.zeros((X_.shape[0], 2))
            Z_hat_[:, 0] = _N_0.logpdf(X_) #+ np.log(prior_0_)
            Z_hat_[:, 1] = _N_1.logpdf(X_) #+ np.log(prior_1_)
            return Z_hat_
        _N_0, _N_1, prior_0_, prior_1_ = m_
        return ___predict_log_proba(X_, _N_0, _N_1, prior_0_, prior_1_)
    # Model and Classification threshold
    model_, prob = m_
    # Get Dataset Variables
    X_ = _get_data(X_, idx_var_, idx_shape_)
    # Predict Classification Probabilities
    Z_ = __predict_probrabilities(X_, model_)
    # Maximum Likelihood Classification
    return _classify(Z_, prob).astype(bool)

def _cloud_GMM_segmentation(X_, m_, idx_var_, idx_shape_):
    def __predict_probrabilities(X_, m_):
        _GMM, invert_label = m_
        return _GMM.predict_proba(X_)
    # Model and Classification threshold
    model_, prob = m_
    # Get Dataset Variables
    X_ = _get_data(X_, idx_var_, idx_shape_)
    # Predict Classification Probabilities
    Z_ = __predict_probrabilities(X_, model_)
    # Maximum Likelihood Classification
    return _classify(Z_, prob, invert_label = model_[1]).astype(bool)

def _cloud_KMS_segmentation(X_, m_, idx_var_, idx_shape_):
    def __predict_probrabilities(X_, m_):
        _KMS, invert_label = m_
        Z_hat_ = _KMS.transform(X_)
        return Z_hat_ / np.sum(Z_hat_, axis = 1)[..., np.newaxis]
    # Model and Classification threshold
    model_, prob = m_
    # Get Dataset Variables
    X_ = _get_data(X_, idx_var_, idx_shape_)
    # Predict Classification Probabilities
    Z_ = __predict_probrabilities(X_, model_)
    # Maximum Likelihood Classification
    return _classify(Z_, prob, invert_label = np.invert(model_[1]) ).astype(bool)

def _cloud_RRC_segmentation(X_, m_, idx_var_, idx_shape_):
    def __predict_probrabilities(X_, w_):
        w_, degree = model_
        Z_hat_ = np.zeros((X_.shape[0], 2))
        z_ = X_ @ w_
        Z_hat_[:, 1] = 1./(1. + np.exp(- z_))
        Z_hat_[:, 0] = 1. - Z_hat_[:, 1]
        return Z_hat_
    # Model and Classification threshold
    model_, prob = m_
    # Get Dataset Variables
    X_ = _get_data(X_, idx_var_, idx_shape_, degree = model_[1])
    # Predict Classification Probabilities
    Z_ = __predict_probrabilities(X_, model_)
    # Maximum Likelihood Classification
    return _classify(Z_, prob).astype(bool)

def _cloud_SVC_segmentation(X_, m_, idx_var_, idx_shape_):
    def __predict_probrabilities(X_, m_):
        _SVC, degree = m_
        Z_hat_ = np.zeros((X_.shape[0], 2))
        z_ = _SVC.decision_function( X_ )
        Z_hat_[:, 1] = 1./(1. + np.exp(- z_))
        Z_hat_[:, 0] = 1. - Z_hat_[:, 1]
        return Z_hat_
    # Model and Classification threshold
    model_, prob = m_
    # Get Dataset Variables
    X_ = _get_data(X_, idx_var_, idx_shape_, degree = model_[1])
    # Predict Classification Probabilities
    Z_ = __predict_probrabilities(X_, model_)
    # Maximum Likelihood Classification
    return _classify(Z_, prob).astype(bool)

def _cloud_ICM_segmentation(X_, m_, idx_var_, idx_shape_):

    # Markov Random Field Model
    def __predict_proba(X_, W_init_, args_, n_eval = 10):
        # Evaluate Likelhood
        def __likelihood(X_, _N_0, _N_1, M, N, D):
            x_ = X_.reshape(M*N, D)
            return _N_0.logpdf(x_), _N_1.logpdf(x_)
        # Energy Potential Function
        def __prior(W_, cliques_, beta, M, N):
            # Prior based on neigborhood class
            def ___neigborhood(w, W_, i, j, cliques_, beta, M, N):
                prior = 0
                # Loop over neigbors
                for clique_ in cliques_:
                    k = i + clique_[0]
                    m = j + clique_[1]
                    if k < 0 or m < 0 or k >= M or m >= N:
                        pass
                    else:
                        if w == W_[k, m]:
                            prior += beta
                        else:
                            prior -= beta
                return prior
            # Variable Initialization
            prior_0_ = np.zeros((M, N))
            prior_1_ = np.zeros((M, N))
            for i in range(M):
                for j in range(N):
                    # Energy function Value and Prior Probability
                    prior_0_[i, j] = ___neigborhood(0, W_, i, j, cliques_, beta, M, N)
                    prior_1_[i, j] = ___neigborhood(1, W_, i, j, cliques_, beta, M, N)
            return prior_0_.flatten(), prior_1_.flatten()
        # Compute softmax values for each sets of scores in x
        def __softmax(x_):
            z_ = np.exp(x_)
            return z_ / np.tile(np.sum(z_, axis = 1)[:, np.newaxis], (1, 2))
        # Compute Probability
        def __energy(lik_, pri_, M, N):
            Z_ = np.zeros((M*N, 2))
            W_ = np.zeros((M*N))
            Z_[:, 0] = lik_[0] + pri_[0]
            Z_[:, 1] = lik_[1] + pri_[1]
            idx_ = Z_[:, 0] < Z_[:, 1]
            W_[idx_] = 1
            U_ = Z_[:, 0]
            U_[idx_] = Z_[idx_, 0]
            u = U_.sum()
            return __softmax(Z_).reshape(M, N, 2), W_.reshape(M, N), u

        # Unpack Argumenets
        cliques_, beta, _N_0, _N_1 = args_
        # Constants Initialization
        M, N, D, n = X_.shape
        Z_hat_ = np.zeros((M, N, 2, n))
        W_hat_ = W_init_.copy()
        # loop over samples
        for i in range(n):
            u_k = - np.inf
            # If Inference of the distribution is necessary
            lik_ = __likelihood(X_[..., i], _N_0, _N_1, M, N, D)
            for j in range(n_eval):
                # Current Evaluation Weights Initialization
                pri_ = __prior(W_hat_[..., i], cliques_, beta, M, N)
                # Compute Probability
                Z_hat_[..., i], W_hat_[..., i], u_k_1 = __energy(lik_, pri_, M, N)
                if u_k_1 > u_k:
                    u_k = u_k_1.copy()
                else:
                    break
        return Z_hat_

    # Model and Classification threshold
    model_, prob = m_
    # Get Dataset Variables
    X_ = _get_data(X_, idx_var_, idx_shape_)
    # Maximum likelihood Initialization
    W_ = _maximum_likelihood(X_, model_)
    # Dataset in Matrix from
    N, dim = X_.shape
    X_ = X_.reshape(60, 80, dim)[..., np.newaxis]
    W_ = W_.reshape(60, 80)[..., np.newaxis]
    # Predict Classification Probabilities
    Z_ = __predict_proba(X_, W_, model_, n_eval = 10)
    Z_ = np.squeeze(Z_).reshape(N, 2)
    # Maximum Likelihood Classification
    return _classify(Z_, prob).astype(bool)

def _cloud_SA_segmentation(X_, m_, idx_var_, idx_shape_):
    # Stochastic Optimization of the Markov Random Field Model
    def __predict_proba(X_, W_init_, args_, n_eval = 5, T_init = 4., T_min = 1e-4, epsilon = 0.75):
        # Cooling Function
        def __exp_cooling(T, epsilon):
            return T * epsilon
        # Perturbation Generate from Importance Sampling
        def __importance_perturbation(U_, M, N):
            E_ = U_.copy()
            E_ = np.absolute(E_ - np.max(E_))
            E_ = E_ / np.sum(E_)
            i_, j_ = np.where(E_ == np.random.choice(E_.flatten(), 1, p = E_.flatten()))
            return i_[0], j_[0]
        # Evaluate Likelhood
        def __likelihood(X_, _N_0, _N_1, M, N, D):
            x_ = X_.reshape(M*N, D)
            Z_ = np.zeros((M, N, 2))
            Z_[..., 0] = _N_0.logpdf(x_).reshape(M, N)
            Z_[..., 1] = _N_1.logpdf(x_).reshape(M, N)
            return Z_
        # Pixels' maximum energy state and Labeling
        def __eval_energy(Z_, M, N):
            W_ = np.zeros((M, N))
            E_ = Z_[..., 0].copy()
            idx_ = Z_[..., 0] < Z_[..., 1]
            E_[idx_] = Z_[idx_, 1]
            W_[idx_] = 1
            return E_, W_
        # Prior based on neigborhood class
        def __neigborhood(w, W_, i, j, cliques_, beta):
            M, N = W_.shape
            prior = 0
            # Loop over neigbors
            for clique_ in cliques_:
                k = i + clique_[0]
                m = j + clique_[1]
                if k < 0 or m < 0 or k >= M or m >= N:
                    pass
                else:
                    if w == W_[k, m]:
                        prior += beta
                    else:
                        prior -= beta
            return prior
        # Compute Posterior Probabilities
        def __eval_posterior(Z_, U_, W_, i, j, cliques_, beta):
            E_ = U_.copy()
            G_ = W_.copy()

            Z_[i, j, 0] = Z_[i, j, 0] + __neigborhood(0, W_, i, j, cliques_, beta)
            Z_[i, j, 1] = Z_[i, j, 1] + __neigborhood(1, W_, i, j, cliques_, beta)

            if G_[i, j] == 1:
                G_[i, j] = 0
                E_[i, j] = Z_[i, j, 0]
            else:
                G_[i, j] = 1
                E_[i, j] = Z_[i, j, 1]
            return E_, G_, Z_
        # Layer Energy to Probabilites
        def __softmax(x_):
            z_ = np.exp(x_)
            return z_ / np.tile( np.sum(z_, axis = 1)[:, np.newaxis], (1, 2))
        # Main Run Simulate Anniling Initialization
        def __run_optimization(X_, W_, M, N, D, args_):
            # Variables Unpacking
            cliques_, beta, _N_0, _N_1 = args_
            # Eval Likelihood
            Z_ = __likelihood(X_, _N_0, _N_1, M, N, D)
            # Initialization of Energy Function
            U_, W_ = __eval_energy(Z_, M, N)
            # Variables initialization
            n_accept = 0
            T = T_init
            # Run until Temeperature is too low
            while T > T_min:
                # Run for this temperature
                for _ in range(n_eval):
                    # Labels Perturbation
                    k, m = __importance_perturbation(U_, M, N)
                    # Eval Perturbation Energy Function
                    E_, G_, Z_ = __eval_posterior(Z_, U_, W_, k, m, cliques_, beta)
                    # Incremenet in the Energy Function
                    du = U_.sum() - E_.sum()
                    # If it is negative get it
                    if du <= 0:
                        # Accept Perturbation
                        n_accept += 1
                        W_ = G_.copy()
                        U_ = E_.copy()
                    else:
                        # Exponential Cooling of the Incrmenet
                        rho = np.exp( - du / T)
                        # Otherwise random acceptation
                        if np.random.uniform() < rho:
                            # Accept
                            n_accept += 1
                            W_ = G_.copy()
                            U_ = E_.copy()
                        else:
                            # Reject
                            pass
                # Update temperature for acceptation
                T = __exp_cooling(T, epsilon)
            return __softmax(Z_.reshape(M*N, 2))

        M, N, D, n = X_.shape
        Z_hat_ = np.zeros((M*N, 2, n))
        for i in range(n):
            Z_hat_[..., i] = __run_optimization(X_[..., i], W_init_[..., i], M, N, D, args_)
        return Z_hat_.reshape(M, N, 2, n)

    # Model and Classification threshold
    model_, prob = m_
    # Get Dataset Variables
    X_ = _get_data(X_, idx_var_, idx_shape_)
    # Maximum likelihood Initialization
    W_ = _maximum_likelihood(X_, model_)
    # Dataset in Matrix from
    N, dim = X_.shape
    X_ = X_.reshape(60, 80, dim)[..., np.newaxis]
    W_ = W_.reshape(60, 80)[..., np.newaxis]
    # Predict Classification Probabilities
    Z_ = __predict_proba(X_, W_, model_, n_eval = 5, T_init = 4., T_min = 1e-4, epsilon = 0.75)
    Z_ = np.squeeze(Z_).reshape(N, 2)
    # Maximum Likelihood Classification
    return _classify(Z_, prob).astype(bool)

def _cloud_MRF_segmentation(X_, m_, idx_var_, idx_shape_):

    # Markov Random Field Model
    def __predict_proba(X_, W_init_, args_ = None, n_eval = 10):
        # Evaluate Likelhood
        def __likelihood(X_, _N_0, _N_1, M, N, D):
            x_ = X_.reshape(M*N, D)
            return _N_0.logpdf(x_), _N_1.logpdf(x_)
        # Energy Potential Function
        def __prior(W_, cliques_, beta, M, N):
            # Prior based on neigborhood class
            def ___neigborhood(w, W_, i, j, cliques_, beta, M, N):
                prior = 0
                # Loop over neigbors
                for clique_ in cliques_:
                    k = i + clique_[0]
                    m = j + clique_[1]
                    if k < 0 or m < 0 or k >= M or m >= N:
                        pass
                    else:
                        if w == W_[k, m]:
                            prior += beta
                        else:
                            prior -= beta
                return prior
            # Variable Initialization
            prior_0_ = np.zeros((M, N))
            prior_1_ = np.zeros((M, N))
            # Loop over Pixels in an Image
            for i in range(M):
                for j in range(N):
                    # Energy function Value and Prior Probability
                    prior_0_[i, j] = ___neigborhood(0, W_, i, j, cliques_, beta, M, N)
                    prior_1_[i, j] = ___neigborhood(1, W_, i, j, cliques_, beta, M, N)
            return prior_0_.flatten(), prior_1_.flatten()
        # Compute softmax values for each sets of scores in x
        def __softmax(x_):
            z_ = np.exp(x_)
            return z_ / np.tile(np.sum(z_, axis = 1)[:, np.newaxis], (1, 2))
        # Compute Probability
        def __energy(lik_, pri_, M, N):
            Z_ = np.zeros((M*N, 2))
            W_ = np.zeros((M*N))
            Z_[:, 0] = lik_[0] + pri_[0]
            Z_[:, 1] = lik_[1] + pri_[1]
            idx_ = Z_[:, 0] < Z_[:, 1]
            W_[idx_] = 1
            U_ = Z_[:, 0]
            U_[idx_] = Z_[idx_, 0]
            u = U_.sum()
            return __softmax(Z_).reshape(M, N, 2), W_.reshape(M, N), u

        # Unpack Argumenets
        cliques_, beta, _N_0, _N_1 = args_

        # Constants Initialization
        M, N, D, n = X_.shape
        Z_hat_ = np.zeros((M, N, 2, n))
        W_hat_ = W_init_.copy()

        # loop over samples
        for i in range(n):

            u_k = - np.inf
            # If Inference of the distribution is necessary
            lik_ = __likelihood(X_[..., i], _N_0, _N_1, M, N, D)
            for j in range(n_eval):
                # Current Evaluation Weights Initialization
                pri_ = __prior(W_hat_[..., i], cliques_, beta, M, N)
                # Compute Probability
                Z_hat_[..., i], W_hat_[..., i], u_k_1 = __energy(lik_, pri_, M, N)
                if u_k_1 > u_k:
                    u_k = u_k_1.copy()
                else:
                    break
        return Z_hat_

    # Model and Classification threshold
    model_, prob = m_
    # Get Dataset Variables
    X_ = _get_data(X_, idx_var_, idx_shape_)
    # Maximum likelihood Initialization
    W_ = _maximum_likelihood(X_, model_)
    # Dataset in Matrix from
    N, dim = X_.shape
    X_ = X_.reshape(60, 80, dim)[..., np.newaxis]
    W_ = W_.reshape(60, 80)[..., np.newaxis]
    # Predict Classification Probabilities
    Z_ = __predict_proba(X_, W_, model_)
    Z_ = np.squeeze(Z_).reshape(N, 2)
    # Maximum Likelihood Classification
    return _classify(Z_, prob).astype(bool)

def _classify_v0(Z_hat_, prob, invert_label):
    # Variables Initizalization
    labels_ = np.zeros((Z_hat_.shape[0]))
    # Maximum Likelihood Classification
    idx_ = Z_hat_[:, 0] * prob < Z_hat_[:, 1] * (1. - prob)
    labels_[idx_] = 1.
    if invert_label:
        return 1. - labels_
    else:
        return labels_

# Test Results for Computing Time
def _rrc_predict(X_, model_):
    def __predict_proba(X_, w_, invert_label):
        Z_hat_ = np.zeros((X_.shape[0], 2))
        z_ = X_ @ w_
        Z_hat_[:, 1] = 1./(1. + np.exp(- z_))
        Z_hat_[:, 0] = 1. - Z_hat_[:, 1]
        return Z_hat_
    w_, invert_label = model_[0]
    theta_           = model_[1]
    # Do the segmentation
    return _classify_v0(Z_hat_ = __predict_proba(X_, w_, invert_label), prob = theta_[0],
                        invert_label = invert_label)

# Test Results for Computing Time
def _svc_predict(X_, model_):
    def __predict_proba(X_, _SVC, invert_label):
        Z_hat_ = np.zeros((X_.shape[0], 2))
        z_ = _SVC.decision_function(X_)
        Z_hat_[:, 1] = 1./(1. + np.exp(- z_))
        Z_hat_[:, 0] = 1. - Z_hat_[:, 1]
        return Z_hat_
    _SVC, invert_label = model_[0]
    theta_             = model_[1]
    # Do the segmentation
    return _classify_v0(Z_hat_ = __predict_proba(X_, _SVC, invert_label), prob = theta_[0],
                        invert_label = invert_label)

# Test Results for Computing Time
def _mrf_predict(X_, _cliques, model_):
    # Markov Random Field Model
    def __predict_proba(X_, _cliques, beta, _N_0, _N_1, n_eval):
        # Evaluate Likelhood
        def __likelihood(X_, _N_0, _N_1, M, N, D):
            x_ = X_.reshape(M*N, D)
            return _N_0.logpdf(x_), _N_1.logpdf(x_)
        # Energy Potential Function
        def __prior(W_, cliques_, beta, M, N):
            # Prior based on neigborhood class
            def ___neigborhood(w, W_, i, j, cliques_, beta, M, N):
                prior = 0
                # Loop over neigbors
                for clique_ in cliques_:
                    k = i + clique_[0]
                    m = j + clique_[1]
                    if k < 0 or m < 0 or k >= M or m >= N:
                        pass
                    else:
                        if w == W_[k, m]:
                            prior += beta
                        else:
                            prior -= beta
                return prior
            # Variable Initialization
            prior_0_ = np.zeros((M, N))
            prior_1_ = np.zeros((M, N))
            for i in range(M):
                for j in range(N):
                    # Energy function Value and Prior Probability
                    prior_0_[i, j] = ___neigborhood(0, W_, i, j, cliques_, beta, M, N)
                    prior_1_[i, j] = ___neigborhood(1, W_, i, j, cliques_, beta, M, N)
            return prior_0_.flatten(), prior_1_.flatten()
        # Compute softmax values for each sets of scores in x
        def __softmax(x_):
            z_ = np.exp(x_)
            return z_ / np.tile(np.sum(z_, axis = 1)[:, np.newaxis], (1, 2))
        # Compute the sum of the values to add to 1
        def __hardmax(x_):
            x_ = x_ - x_.min()
            return np.nan_to_num(x_ / np.tile(np.sum(x_, axis = 1)[:, np.newaxis], (1, 2)))
        # Compute Pixels' Energy
        def __energy(lik_, pri_, M, N):
            # Variables Initialization
            Z_ = np.zeros((M*N, 2))
            W_ = np.zeros((M*N))
            # Labels Energy per pixel
            Z_[..., 0] = lik_[0] + pri_[0]
            Z_[..., 1] = lik_[1] + pri_[1]
            # Maximum Energy Classification
            idx_ = Z_[..., 0] < Z_[..., 1]
            # Compute the total energy
            U_ = Z_[..., 0].copy()
            U_[idx_] = Z_[idx_, 1]
            W_[idx_] = 1
            Z_ =__hardmax(Z_)
            return Z_.reshape(M, N, 2), W_.reshape(M, N), U_.sum()
        # Maximum Likelihood classification
        def __ML(X_, _N_0, _N_1):
            # Evaluate Likelyhood
            def __likelihood(X_, _N_0, _N_1):
                M, N, D = X_.shape
                x_ = X_.reshape(M*N, D)
                return _N_0.logpdf(x_), _N_1.logpdf(x_)
            # Classify by maximum likelihood
            def __classify(W_hat_, lik_):
                M, N = W_hat_.shape
                index_ = (lik_[0] < lik_[1]).reshape(M, N)
                W_hat_[index_] = 1
                return W_hat_
            # Variable Initialization
            M, N, D, n = X_.shape
            W_hat_ = np.zeros((M, N, n))
            # loop over Images
            for i in range(n):
                W_hat_[..., i] = __classify(W_hat_[..., i], lik_ = __likelihood(X_[..., i],
                                                                                _N_0, _N_1))
            return W_hat_
        # Cliques
        C_0_ = [[0, 0]]
        C_1_ = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        C_2_ = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
        # Variables Initialization
        cliques_ = [C_0_, C_1_, C_1_ + C_2_][_cliques]
        W_init_  = __ML(X_, _N_0, _N_1)
        # Constants Initialization
        M, N, D, n = X_.shape
        Z_hat_ = np.zeros((M, N, 2, n))
        W_hat_ = W_init_.copy()
        # loop over samples
        for i in range(n):
            u_k = - np.inf
            # If Inference of the distribution is necessary
            lik_ = __likelihood(X_[..., i], _N_0, _N_1, M, N, D)
            for j in range(n_eval):
                # Current Evaluation Weights Initialization
                pri_ = __prior(W_hat_[..., i], cliques_, beta, M, N)
                # Compute Probability
                Z_hat_[..., i], W_hat_[..., i], u_k_1 = __energy(lik_, pri_, M, N)
                if u_k_1 <= u_k:
                    break
                else:
                    u_k = u_k_1.copy()
        return Z_hat_
    MRF_, invert_label = model_[0]
    theta_             = model_[1]
    # Do the segmentation
    Z_hat_ = __predict_proba(X_, _cliques, beta = theta_[2], _N_0 = MRF_[0], _N_1 = MRF_[1],
                             n_eval = 100)
    Z_hat_ = np.swapaxes(Z_hat_, 2, 3)
    Z_hat_ = Z_hat_.reshape(Z_hat_.shape[0]*Z_hat_.shape[1]*Z_hat_.shape[2], Z_hat_.shape[3])
    return _classify_v0(Z_hat_, prob = theta_[0], invert_label = invert_label)

# Apply Scaling
def _scaling(X_, _scaler):
    return _scaler.transform(X_)

def _image_segmentation_v0(models_, names_, I_norm_2_, M_lk_, I_scatter_, I_diffuse_,
                           K_0_, H_0_, K_1_, H_1_, K_2_, H_2_):
    def __get_dataset(X_, var_idx_, str_idx_):
        def ___median_padding(x_, M, N, D, n):
            I_ = np.zeros((M + 2, N + 2, D, n))
            # Zero-pad image
            I_[1:-1,1:-1, ...] = x_
            # loop over images
            for i in range(n):
                # loop over dimension
                for j in range(D):
                    # Aply median filter
                    I_prime_ = median_filter(I_[..., j, i], size = 7, mode = 'reflect')
                    # Median fileter Papped Image
                    I_[0, :, j, i]  = I_prime_[0, :]
                    I_[-1, :, j, i] = I_prime_[-1, :]
                    I_[:, 0, j, i]  = I_prime_[:, 0]
                    I_[:, -1, j, i] = I_prime_[:, -1]
            return I_
        # Grab Data only desired Features
        x_ = X_[..., var_idx_, :].copy()
        # Variables Initialization
        M, N, D, n = x_.shape
        X_ = np.zeros((0, len(var_idx_)*np.sum(str_idx_), n))
        # Image padding with madian filter
        I_ = ___median_padding(x_, M, N, D, n)
        # Loop over Pixels
        for i in range(1, M + 1):
            for j in range(1, N + 1):
                # Grab the entired 2nd order neiborhood
                w_ = I_[i - 1:i + 2, j - 1:j + 2, ...].copy()
                # Grab desired Neiborhood
                x_ = w_[str_idx_, ...]
                # Concatenate Pixel Features Vector
                X_ = np.concatenate((X_, x_.reshape(x_.shape[0]*x_.shape[1],
                                                    x_.shape[2])[np.newaxis, ...]), axis = 0)
        return X_[..., 0]

    # Pixels Structures
    idx_0_ = np.matrix([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype = bool)
    idx_1_ = np.matrix([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype = bool)
    idx_2_ = np.matrix([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype = bool)
    var_   = [[4, 5], [6, 7], [3, 9], [0, 1, 3]]
    str_   = [idx_0_, idx_1_, idx_2_]

    X_ = np.concatenate((I_norm_2_[..., np.newaxis], M_lk_[..., np.newaxis],
                         I_scatter_[..., np.newaxis], I_diffuse_[..., np.newaxis],
                         K_0_[..., np.newaxis] - 273.15, H_0_[..., np.newaxis]/1000.,
                         K_1_[..., np.newaxis] - 273.15, H_1_[..., np.newaxis]/1000.,
                         K_2_[..., np.newaxis] - 273.15, H_2_[..., np.newaxis]/1000.),
                         axis = 2)[..., np.newaxis]

    # RRC model dataset - rrc_131
    idx_mod  = 4
    degree   = 1
    idx_var  = 3
    idx_str  = 1
    var_idx_ = var_[idx_var]
    str_idx_ = str_[idx_str]
    model_   = models_[idx_mod][0]
    name_    = names_[idx_mod]
    x_         = __get_dataset(X_, var_idx_, str_idx_)
    x_         = PolynomialFeatures(degree).fit_transform(x_)
    W_hat_rrc_ = _rrc_predict(x_, model_).reshape(60, 80)

    # SVC model dataset - svc_130
    idx_mod  = 5
    degree   = 1
    idx_var  = 3
    idx_str  = 0
    var_idx_ = var_[idx_var]
    str_idx_ = str_[idx_str]
    model_   = models_[idx_mod][0]
    name_    = names_[idx_mod]
    x_         = __get_dataset(X_, var_idx_, str_idx_)
    x_         = PolynomialFeatures(degree).fit_transform(x_)
    W_hat_svc_ = _svc_predict(x_, model_).reshape(60, 80)

    # ICM-MRF model dataset - icm-mrf_121
    idx_mod  = 8
    cliques  = 1
    idx_var  = 2
    idx_str  = 1
    var_idx_ = var_[idx_var]
    str_idx_ = str_[idx_str]
    model_   = models_[idx_mod][0]
    name_    = names_[idx_mod]
    x_             = __get_dataset(X_, var_idx_, str_idx_)
    x_             = x_.reshape(60, 80, x_.shape[-1])[..., np.newaxis]
    W_hat_icm_mrf_ = _mrf_predict(x_, cliques, model_).reshape(60, 80)

    plt.figure(figsize = (7.5, 5))
    plt.title(r'$p ( y_{i,j} = 1| \Theta, x_{i,j})$', fontsize = 25)
    plt.imshow((W_hat_rrc_ + W_hat_svc_ + W_hat_icm_mrf_)/3., cmap = 'jet')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar().ax.tick_params(labelsize = 20)
    plt.show()

    W_hat_mean_ = (W_hat_rrc_ + W_hat_svc_ + W_hat_icm_mrf_)/3. >= 0.5

    return W_hat_mean_

def _image_segmentation_v1(models_, names_, I_diffuse_, H_2_):
    def __get_dataset(X_, var_idx_, str_idx_):
        def ___median_padding(x_, M, N, D, n):
            I_ = np.zeros((M + 2, N + 2, D, n))
            # Zero-pad image
            I_[1:-1,1:-1, ...] = x_
            # loop over images
            for i in range(n):
                # loop over dimension
                for j in range(D):
                    # Aply median filter
                    I_prime_ = median_filter(I_[..., j, i], size = 7, mode = 'reflect')
                    # Median fileter Papped Image
                    I_[0, :, j, i]  = I_prime_[0, :]
                    I_[-1, :, j, i] = I_prime_[-1, :]
                    I_[:, 0, j, i]  = I_prime_[:, 0]
                    I_[:, -1, j, i] = I_prime_[:, -1]
            return I_
        # Grab Data only desired Features
        x_ = X_.copy()
        # Variables Initialization
        M, N, D, n = x_.shape
        X_ = np.zeros((0, len(var_idx_)*np.sum(str_idx_), n))
        # Image padding with madian filter
        I_ = ___median_padding(x_, M, N, D, n)
        # Loop over Pixels
        for i in range(1, M + 1):
            for j in range(1, N + 1):
                # Grab the entired 2nd order neiborhood
                w_ = I_[i - 1:i + 2, j - 1:j + 2, ...].copy()
                # Grab desired Neiborhood
                x_ = w_[str_idx_, ...]
                # Concatenate Pixel Features Vector
                X_ = np.concatenate((X_, x_.reshape(x_.shape[0]*x_.shape[1],
                                                    x_.shape[2])[np.newaxis, ...]), axis = 0)
        return X_[..., 0]

    # Pixels Structures
    idx_0_ = np.matrix([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype = bool)
    idx_1_ = np.matrix([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype = bool)
    idx_2_ = np.matrix([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype = bool)
    var_   = [[4, 5], [6, 7], [3, 9], [0, 1, 3]]
    str_   = [idx_0_, idx_1_, idx_2_]

    X_ = np.concatenate((I_diffuse_[..., np.newaxis],
                         H_2_[..., np.newaxis]/1000.), axis = 2)[..., np.newaxis]

    # ICM-MRF model dataset - icm-mrf_121
    idx_mod  = 8
    cliques  = 1
    idx_var  = 2
    idx_str  = 1
    var_idx_ = var_[idx_var]
    str_idx_ = str_[idx_str]
    model_   = models_[idx_mod][0]
    name_    = names_[idx_mod]
    x_             = __get_dataset(X_, var_idx_, str_idx_)
    x_             = x_.reshape(60, 80, x_.shape[-1])[..., np.newaxis]
    W_hat_icm_mrf_ = _mrf_predict(x_, cliques, model_).reshape(60, 80)
    return W_hat_icm_mrf_ > 0.

def _image_segmentation_v2(models_, names_, I_norm_2_, M_lk_, I_scatter_, I_diffuse_,
                           K_0_, H_0_, K_1_, H_1_, K_2_, H_2_):
    def __get_dataset(X_, var_idx_, str_idx_):
        def ___median_padding(x_, M, N, D, n):
            I_ = np.zeros((M + 2, N + 2, D, n))
            # Zero-pad image
            I_[1:-1,1:-1, ...] = x_
            # loop over images
            for i in range(n):
                # loop over dimension
                for j in range(D):
                    # Aply median filter
                    I_prime_ = median_filter(I_[..., j, i], size = 7, mode = 'reflect')
                    # Median fileter Papped Image
                    I_[0, :, j, i]  = I_prime_[0, :]
                    I_[-1, :, j, i] = I_prime_[-1, :]
                    I_[:, 0, j, i]  = I_prime_[:, 0]
                    I_[:, -1, j, i] = I_prime_[:, -1]
            return I_
        # Grab Data only desired Features
        x_ = X_[..., var_idx_, :].copy()
        # Variables Initialization
        M, N, D, n = x_.shape
        X_ = np.zeros((0, len(var_idx_)*np.sum(str_idx_), n))
        # Image padding with madian filter
        I_ = ___median_padding(x_, M, N, D, n)
        # Loop over Pixels
        for i in range(1, M + 1):
            for j in range(1, N + 1):
                # Grab the entired 2nd order neiborhood
                w_ = I_[i - 1:i + 2, j - 1:j + 2, ...].copy()
                # Grab desired Neiborhood
                x_ = w_[str_idx_, ...]
                # Concatenate Pixel Features Vector
                X_ = np.concatenate((X_, x_.reshape(x_.shape[0]*x_.shape[1],
                                                    x_.shape[2])[np.newaxis, ...]), axis = 0)
        return X_[..., 0]

    # Pixels Structures
    idx_0_ = np.matrix([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype = bool)
    idx_1_ = np.matrix([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype = bool)
    idx_2_ = np.matrix([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype = bool)
    var_   = [[4, 5], [6, 7], [3, 9], [0, 1, 3]]
    str_   = [idx_0_, idx_1_, idx_2_]

    X_ = np.concatenate((I_norm_2_[..., np.newaxis], M_lk_[..., np.newaxis],
                         I_scatter_[..., np.newaxis], I_diffuse_[..., np.newaxis],
                         K_0_[..., np.newaxis] - 273.15, H_0_[..., np.newaxis]/1000.,
                         K_1_[..., np.newaxis] - 273.15, H_1_[..., np.newaxis]/1000.,
                         K_2_[..., np.newaxis] - 273.15, H_2_[..., np.newaxis]/1000.),
                         axis = 2)[..., np.newaxis]

    # RRC model dataset - rrc_131
    idx_mod  = 4
    degree   = 1
    idx_var  = 3
    idx_str  = 1
    var_idx_ = var_[idx_var]
    str_idx_ = str_[idx_str]
    model_   = models_[idx_mod][0]
    name_    = names_[idx_mod]
    x_         = __get_dataset(X_, var_idx_, str_idx_)
    x_         = PolynomialFeatures(degree).fit_transform(x_)
    W_hat_rrc_ = _rrc_predict(x_, model_).reshape(60, 80)

    return W_hat_rrc_ > 0.

__all__ = ['_cloud_THR_segmentation', '_cloud_NBC_segmentation', '_cloud_GC_segmentation',
           '_cloud_GMM_segmentation', '_cloud_KMS_segmentation', '_cloud_segmentation',
           '_cloud_RRC_segmentation', '_cloud_SVC_segmentation', '_cloud_SA_segmentation',
           '_cloud_ICM_segmentation', '_cloud_MRF_segmentation', '_image_segmentation_v0',
           '_image_segmentation_v1', '_image_segmentation_v2']
