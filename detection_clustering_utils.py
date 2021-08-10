import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from utils import *

from scipy import optimize
from scipy.ndimage import median_filter
from scipy.stats import vonmises, norm, beta, uniform, multivariate_normal, dirichlet
from scipy.special import logsumexp, iv, beta, gamma, digamma
from time import time

# Compute BIC, AIC, CLC and ICL Bayesian Metrics
def _error_metrics(W_, ll_, z_, theta_):
    n, k = W_.shape
    # Count the number of model parameters
    K = 0
    for theta in theta_: K += np.size(theta)
    # Compute Indicator log-likelihood
    log_lik_ = np.zeros(n)
    for i in range(n):
        for j in range(k):
            if z_[i] == j: log_lik_[i] = ll_[i, j]
    log_lik_ = np.nan_to_num(log_lik_)
    # Compute Log-likelihood
    LL = log_lik_.sum()
    # Posterior Probabilities
    W_ = np.nan_to_num(W_)
    # Compute Entropy
    H = - (W_ * np.nan_to_num(np.log(W_))).sum()
    # Compute Error Metrics
    BIC = -2.*LL + K*np.log(n)
    AIC = -2.*LL + 2*K
    CLC = -2.*LL + 2*H
    ICL = BIC + 2*H
    return [H, LL, BIC, AIC, CLC, ICL]

# # Compute BIC, AIC, CLC and ICL Bayesian Metrics with indicator function in the posterior
# def _error_metrics(W_, ll_, z_, theta_):
#     n, k = W_.shape
#     # Count the number of model parameters
#     K = 0
#     for theta in theta_: K += np.size(theta)
#     # COmpute Log-likelihood
#     LL = logsumexp(ll_, axis = 1).sum()
#     # Posterior Probabilities
#     W_ = np.nan_to_num(W_)
#     # Compute Entropy
#     H = - (W_ * np.nan_to_num(np.log(W_)) ).sum()
#     # Compute Error Metrics
#     BIC = -2.*LL + K*np.log(n)
#     AIC = -2.*LL + 2*K
#     CLC = -2.*LL + 2*H
#     ICL = BIC + 2*H
#     return [H, LL, BIC, AIC, CLC, ICL]

# Log-Gaussian Distribution
def _log_N(X_, m, s):
    return norm(m, np.sqrt(s)).logpdf(X_)#[:, np.newaxis]

# Calulate mixture Model Log-likelihood
def _gaussian_mixture_model_log_likelihood(m_, s_, X_, w_, n_clusters):
    log_P_ = np.empty((X_.shape[0], 0))
    log_w_ = np.log(w_)
    # weighted log-likelihood for each distribution
    for k in range(n_clusters):
        log_p_ = log_w_[k] + _log_N(X_, m_[k], s_[k])
        log_P_ = np.concatenate((log_P_, log_p_), axis = 1)
    # Log-sum-exp of all distributions log-likelihoods
    log_lik_ = logsumexp(log_P_, axis = 1)[:, np.newaxis]
    return np.nan_to_num(log_P_), np.nan_to_num(log_lik_)

# Multivariate Gaussian Mixture Model
def _EM_GMM(X_, n_clusters = 2, n_init = 3, tol = 0.1, metric = 1,
            reg_term = 1e-5, n_iter = 1000, verbose = False, alpha_0 = 0.):
    init_lim_low = X_.min()
    init_lim_up  = X_.max()
    # Check for EM algoritm convergenceG
    def __convergence(ll_k_1_, ll_k, tol):
        if abs(ll_k - ll_k_1_) < tol: return True
        else: return False
    # Maximization Step
    def __maximization(X_, m_, s_, W_, n_clusters):
        # Analitical Solution of the mean
        def ___mean(X_, W_):
            return W_.T @ X_ / W_.sum()
        # Analitical Solution of the variance
        def ___convariance(X_, W_, m):
            return ( (X_.T @ (np.eye(W_.shape[0]) * W_) @ X_) / W_.sum()) - m**2
        # Calculate the Responsabilies
        def ___resposabilities(W_):
            return W_.sum(axis = 0)[:, np.newaxis] + np.finfo(W_.dtype).eps
        # Regularization via Variational Inverence
        def ___variational_prior(w_):
            if alpha_0 != 0.:
                alpha_ = alpha_0 + w_
                w_ = []
                for alpha in alpha_:
                    w_.append(np.exp(digamma(alpha) - digamma(alpha_.sum())))
                return np.stack(w_)
            else:
                return w_ / W_.shape[0]
        # Regularization via MAP
        def ___MAP_prior(w_, N, K):
            alpha_ = np.ones(w_.shape) * alpha_0
            return (w_ + alpha_ - 1.)/(N - K + alpha_.sum())
        # Probabilities and mixture model weights
        w_ = ___resposabilities(W_)
        # Force weightes to be within the limit
        #w_ = ___variational_prior(w_)
        w_ = ___MAP_prior(w_, N = W_.shape[0], K = W_.shape[1])[:, 0]
        for k in range(n_clusters):
            # Analytical Solution of the parameters
            m_[k] = ___mean(X_, W_[..., k])
            s_[k] = ___convariance(X_, W_[..., k], m_[k])
        return m_, s_, w_
    # Random Initialization of VM mixture model Paramters
    def __rand_init(X_):
        # Constants Definition
        N, dim = X_.shape
        ODLL_k = - np.inf
        CDLL_k = - np.inf
        # Variable Initialization
        m_init_ = np.empty((n_clusters))
        s_init_ = np.empty((n_clusters))
        # Randon Initialization VM Distributions weights
        w_init_ = np.random.uniform(0, 1, n_clusters)#[:, np.newaxis]
        w_init_/= w_init_.sum()
        # Von Mises Distribution Parameters Initialization
        for k in range(n_clusters):
            # Random Initialization of the VM Parameters
            m_init_[k] = np.random.uniform(init_lim_low, init_lim_up, 1)[0]
            s_init_[k] = np.random.uniform(0, init_lim_up, 1)[0]
        return m_init_.copy(), s_init_.copy(), w_init_.copy(), ODLL_k, CDLL_k, dim
    # Expectation Step
    def __expectation(X_, m_, s_, w_, n_clusters):
        log_P_, log_lik_ = _gaussian_mixture_model_log_likelihood(m_, s_, X_, w_, n_clusters)
        log_w = dirichlet(np.ones(w_.shape[0]) * alpha_0).logpdf(w_)
        return np.exp(log_P_ - log_lik_), np.sum(log_lik_) + log_w, log_P_
    # Run Initialization of Expectation-Maximization algorithm
    def __EM(X_):
        m_, s_, w_, ODLL_k, ll_k, dim  = __rand_init(X_)
        # Stat iterative Expectation-Maximization Algorithm
        for i in range(n_iter):
            XX_ = X_.copy()
            # Expectation Step
            W_, ll_k_1, ll_k_1_ = __expectation(XX_, m_, s_, w_, n_clusters)
            if verbose:
                print('>>> No iter.: {} log-likelihood: {}'.format(i, ll_k_1))
            # Check for EM convergence
            if __convergence(ll_k_1, ll_k, tol):
                break
            else:
                # Maximization Step
                m_, s_, w_ = __maximization(XX_, m_, s_, W_, n_clusters)
                ll_k  = ll_k_1
                ll_k_ = ll_k_1_
        # Sample Cluster Label
        z_ = np.argmax(W_, axis = 1)
        # Get Scores
        scores_ = _error_metrics(W_, ll_k_, z_, theta_ = [m_, s_, w_])
        return [m_, s_, w_, z_, ll_k, scores_]
    # Storage Variable Initialization
    x_, f_ = [], []
    # Loop over Initialization
    for i in range(n_init):
        t1 = time()
        # Run Expectation-Maximization
        res_ = __EM(X_)
        if verbose:
            print('>>> GMM: No. Init.: {} Time: {} s, CDLL: {}'.format(i, time() - t1, res_[-2]))
        x_.append(res_)
        #f_.append(res_[-2])
        f_.append(res_[-1][metric])
    # Return the highest log-likelihood results
    if metric == 1:
        x_ = x_[np.argmax(f_)]
        print('>>> GMM: CDLL: {}'.format(x_[-1][metric]))
    else:
        x_ = x_[np.argmin(f_)]
    return np.concatenate((x_[0], x_[1], x_[2]), axis = 0), x_[4], x_[5]

# Log-Multivariate Normal Distribution
def _log_MN(X_, m_, C_, reg_term):
    return multivariate_normal(m_, C_  + np.eye(X_.shape[1])*reg_term).logpdf(X_)[:, np.newaxis]

# Calulate mixture Model Log-likelihood
def _multivariate_gaussian_mixture_model_log_likelihood(m_, C_, X_, w_, reg_term, n_clusters):
    log_P_ = np.empty((X_.shape[0], 0))
    log_w_ = np.log(w_)
    # weighted log-likelihood for each distribution
    for k in range(n_clusters):
        log_p_ = log_w_[k, ...] + _log_MN(X_, m_[k, ...], C_[k, ...], reg_term)
        log_P_ = np.concatenate((log_P_, log_p_), axis = 1)
    # Log-sum-exp of all distributions log-likelihoods
    log_lik_ = logsumexp(log_P_, axis = 1)[:, np.newaxis]
    return np.nan_to_num(log_P_), np.nan_to_num(log_lik_)

# Multivariate Gaussian Mixture Model
def _EM_MGMM(X_, n_clusters = 2, n_init = 3, tol = 0.1, metric = 1,
             reg_term = 1e-5, n_iter = 1000, verbose = False, alpha_0 = 0.):
    init_lim_low =  X_.min()
    init_lim_up  =  X_.max()
    # Check for EM algoritm convergenceG
    def __convergence(ll_k_1_, ll_k, tol):
        if abs(ll_k - ll_k_1_) < tol: return True
        else: return False
    # Maximization Step
    def __maximization(X_, m_, C_, W_, n_clusters):
        # Analitical soluction of the mean
        def ___mean(X_, W_):
            return W_.T @ X_ / W_.sum()
        # Analitical soluction of the Covariance
        def ___convariance(X_, W_, m_):
            return ( (X_.T @ (np.eye(W_.shape[0]) * W_) @ X_) / W_.sum()) - (m_[:, np.newaxis] @ m_[:, np.newaxis].T)
        # Calculate the Responsabilies
        def ___resposabilities(W_):
            return W_.sum(axis = 0)[:, np.newaxis] + np.finfo(W_.dtype).eps
        # Regularization via Variational Inverence
        def ___variational_prior(w_):
            if alpha_0 != 0.:
                alpha_ = alpha_0 + w_
                w_ = []
                for alpha in alpha_:
                    w_.append(np.exp(digamma(alpha) - digamma(alpha_.sum())))
                return np.stack(w_)
            else:
                return w_ / W_.shape[0]
        # Regularization via MAP
        def ___MAP_prior(w_, N, K):
            alpha_ = np.ones(w_.shape) * alpha_0
            return (w_ + alpha_ - 1.)/(N - K + alpha_.sum())
        # Probabilities and mixture model weights
        w_ = ___resposabilities(W_)
        # Force weightes to be within the limit
        #w_ = ___variational_prior(w_)
        w_ = ___MAP_prior(w_, N = W_.shape[0], K = W_.shape[1])
        for k in range(n_clusters):
            # Analytical Solution of the parameters
            m_[k, ...] = ___mean(X_, W_[..., k])
            C_[k, ...] = ___convariance(X_, W_[..., k], m_[k, ...])
        return m_, C_, w_
    # Random Initialization of VM mixture model Paramters
    def __rand_init(X_):
        # Constants Definition
        N, dim = X_.shape
        ODLL_k = - np.inf
        CDLL_k = - np.inf
        # Variable Initialization
        m_init_ = np.empty((n_clusters, dim))
        C_init_ = np.empty((n_clusters, dim, dim))
        # Randon Initialization VM Distributions weights
        w_init_ = np.random.uniform(0, 1, n_clusters)[:, np.newaxis]
        w_init_/= w_init_.sum()
        # Von Mises Distribution Parameters Initialization
        for k in range(n_clusters):
            # Random Initialization of the VM Parameters
            m_init_[k, ...] = np.random.uniform(0, init_lim_up, 1)[0]
            C_init_[k, ...] = np.eye(dim) * np.random.uniform(0, init_lim_up, 1)[0]
        return m_init_.copy(), C_init_.copy(), w_init_.copy(), ODLL_k, CDLL_k, dim
    # Expectation Step
    def __expectation(X_, m_, C_, w_, reg_term, n_clusters):
        log_P_, log_lik_ = _multivariate_gaussian_mixture_model_log_likelihood(m_, C_, X_, w_, reg_term, n_clusters)
        log_w = dirichlet(np.ones(w_.shape[0]) * alpha_0).logpdf(w_)
        return np.exp(log_P_ - log_lik_), np.sum(log_lik_) + log_w, log_P_
    # Complete Data log-likelihood
    def __complete_data_log_likelihood(X_, m_, C_, W_, w_, n_clusters):
        log_Z_ = np.zeros((X_.shape[0], n_clusters))
        log_w_ = np.log(w_)
        # weighted log-likelihood for each distribution
        for k in range(n_clusters):
            log_Z_[:, k] = W_[:, k] * log_w_[k, ...] + W_[:, k] * np.squeeze(_log_MN(X_, m_[k, ...], C_[k, ...]))
        return log_Z_.sum()
    def _rise_flag(m_, C_, w_, ll_, ll):
        if np.isnan(m_).any() or np.isnan(C_).any() or np.isnan(w_).any() or np.isnan(ll_).any() or np.isnan(ll):
            return False
        if np.isinf(m_).any() or np.isinf(C_).any() or np.isinf(w_).any() or np.isinf(ll_).any() or np.isinf(ll):
            return False
        return True
    # Run Initialization of Expectation-Maximization algorithm
    def __EM(X_):
        m_, C_, w_, ODLL_k, ll_k, dim  = __rand_init(X_)
        # Stat iterative Expectation-Maximization Algorithm
        for i in range(n_iter):
            XX_ = X_.copy()
            # Expectation Step
            W_, ll_k_1, ll_k_1_ = __expectation(XX_, m_, C_, w_, reg_term, n_clusters)
            if verbose:
                print('>>> No iter.: {} Data log-likelihood: {}'.format(i, ll_k_1))
            # Check for EM convergence
            if __convergence(ll_k_1, ll_k, tol):
                break
            else:
                # Maximization Step
                m_, C_, w_ = __maximization(XX_, m_, C_, W_, n_clusters)
                flag = _rise_flag(m_, C_, w_, ll_k_1_, ll_k_1)
                ll_k_ = ll_k_1_
                ll_k  = ll_k_1
            # Re-start if gradient optimization did not converge
            if not flag:
                m_, C_, w_, ODLL_k, ll_k, dim  = __rand_init(X_)
        # Sample Cluster Label
        z_ = np.argmax(W_, axis = 1)
        # Get Scores
        scores_ = _error_metrics(W_, ll_k_, z_, theta_ = [m_, C_, w_])
        return [m_, C_, w_, z_, ll_k, scores_]
    # Storage Variable Initialization
    x_, f_ = [], []
    # Loop over Initialization
    for i in range(n_init):
        t1 = time()
        # Run Expectation-Maximization
        res_ = __EM(X_)
        if verbose:
            print('>>> MGMM: No. Init.: {} Time: {} s, CDLL: {}'.format(i, time() - t1, res_[-2]))
        x_.append(res_)
        f_.append(res_[-1][metric])
    # Return the highest log-likelihood results
    if metric == 1:
        x_ = x_[np.argmax(f_)]
        print('>>> MGMM: CDLL: {}'.format(x_[-1][metric]))
    else:
        x_ = x_[np.argmin(f_)]
    #return np.concatenate((x_[0], x_[1], np.squeeze(x_[2], axis = 1)), axis = 0), x_[4], x_[5]
    return [x_[0], x_[1], x_[2]], x_[4], x_[5]

# Log-Von Mises Distribution
def _log_VM(X_, mu, kappa):
    return kappa*np.cos(X_ - mu) - np.log(2*np.pi) - np.log(iv(0, kappa))

# Calulate mixture Model Log-likelihood
def _von_mises_mixture_model_log_likelihood(m_, k_, X_, w_, n_clusters):
    log_P_ = np.empty((X_.shape[0], 0))
    log_w_ = np.log(w_)
    # weighted log-likelihood for each distribution
    for k in range(n_clusters):
        log_p_ = log_w_[k] + _log_VM(X_, m_[k], k_[k])
        log_P_ = np.concatenate((log_P_, log_p_), axis = 1)
    # Log-sum-exp of all distributions log-likelihoods
    log_lik_ = logsumexp(log_P_, axis = 1)[:, np.newaxis]
    #return log_P_, log_lik_
    return np.nan_to_num(log_P_), np.nan_to_num(log_lik_)

# Von Meses Mixture Model
def _EM_VmMM(X_, n_clusters = 2, n_init = 3, tol = 0.1, metric = 1, n_iter = 1000,
             reg_term = 1e-5, w_init_ = None, verbose = False, alpha_0 = 0.):
    init_lim_low = 1e-5
    init_lim_up  = 100.
    # Check for EM algoritm convergenceG
    def __convergence(ll_k_1_, ll_k, tol):
        if abs(ll_k - ll_k_1_) < tol: return True
        else: return False
    # Log-Von Mises Distribution partial derivation
    def __grad_log_VM(X_, mu, kappa):
        g_mu_    = kappa * np.sin(X_ - mu)
        g_kappa_ = np.cos(X_ - mu) - ( iv(1, kappa)/iv(0, kappa) )
        return np.concatenate((g_mu_, g_kappa_), axis = 1)
    # Gradient Based marginal likelihood optimization
    def __optimization(X_, m_, k_, W_, bounds_):
        # Log-Likelihood Evaluation
        def ___f(x_, X_, W_):
            # Unpack Variables
            m_ = x_[:n_clusters]
            k_ = x_[n_clusters:]
            # Variable Initialization
            log_l_ = 0
            # Loop over clusters Parameters
            for k in range(n_clusters):
                # Calculate Mixture Model Marginal Log-likelihood
                log_f_ = _log_VM(X_, m_[k], k_[k])
                log_f_ = W_[:, k] * log_f_[:, 0]
                log_l_+= log_f_.sum()
            return - log_l_
        # Log-Likelihood Gradient Evaluation
        def ___g(x_, X_, W_):
            # Unpack Variables
            m_ = x_[:n_clusters]
            k_ = x_[n_clusters:]
            # Variable Initialization
            log_l_ = np.zeros((n_clusters*2, 1))
            # Loop over clusters Parameters
            for k in range(n_clusters):
                # Calculate gradient of Mixture Model Marginal Log-likelihood
                log_g_ = __grad_log_VM(X_, m_[k], k_[k])
                log_g_ = W_[:, k][:, np.newaxis] * log_g_
                log_g_ = np.sum(log_g_, axis = 0)
                log_l_[k] = log_g_[0]
                log_l_[k + n_clusters] = log_g_[1]
            return - log_l_
        # Gradient Based Optimization
        opt_ = optimize.minimize(fun = ___f, jac = ___g, x0 = np.concatenate((m_, k_)),
                                 bounds = bounds_, args = (X_, W_), method = 'SLSQP')
        return opt_['x'][:n_clusters], opt_['x'][n_clusters:], opt_['success']
    # Maximization Step
    def __maximization(X_, m_, k_, W_, bounds_, n_clusters):
        # Calculate the Responsabilies
        def ___resposabilities(W_):
            return W_.sum(axis = 0)[:, np.newaxis] + np.finfo(W_.dtype).eps
        # Regularization via Variational Inverence
        def ___variational_prior(w_):
            if alpha_0 != 0.:
                alpha_ = alpha_0 + w_
                w_ = []
                for alpha in alpha_:
                    w_.append(np.exp(digamma(alpha) - digamma(alpha_.sum())))
                return np.stack(w_)
            else:
                return w_ / W_.shape[0]
        # Regularization via MAP
        def ___MAP_prior(w_, N, K):
            alpha_ = np.ones(w_.shape) * alpha_0
            return (w_ + alpha_ - 1.)/(N - K + alpha_.sum())
        # Probabilities and mixture model weights
        w_ = ___resposabilities(W_)
        # Force weightes to be within the limit
        #w_ = ___variational_prior(w_)
        w_ = ___MAP_prior(w_, N = W_.shape[0], K = W_.shape[1])
        # Optimize Distribution Parameters by Gradient
        m_, k_, flag = __optimization(X_, m_, k_, W_, bounds_)
        return m_, k_, w_, flag
    # Random Initialization of VM mixture model Paramters
    def __rand_init(X_):
        # Constants Definition
        N, dim = X_.shape
        ll_k   = - np.inf
        # Variable Initialization
        bounds_ = []
        m_init_ = np.empty((n_clusters, dim))
        k_init_ = np.empty((n_clusters, dim))
        # Randon Initialization VM Distributions weights
        w_init_ = np.random.uniform(0, 1, n_clusters)[:, np.newaxis]
        w_init_/= w_init_.sum()
        # Von Mises Distribution Parameters Initialization
        for k in range(n_clusters):
            # Define Variable Boundaries for Optimization
            bounds_.insert(0, (init_lim_low, 2*np.pi))
            bounds_.append((init_lim_low, init_lim_up))
            # Random Initialization of the VM Parameters
            m_init_[k] = np.random.uniform(0, 2*np.pi, 1)[0]
            k_init_[k] = np.random.uniform(init_lim_low, init_lim_up, 1)[0]
        return m_init_.copy(), k_init_.copy(), w_init_.copy(), bounds_, ll_k, dim

    def __init(X_, w_init_):
        # Constants Definition
        N, dim = X_.shape
        ll_k   = - np.inf
        # Variable Initialization
        bounds_ = []
        # Randon Initialization VM Distributions weights
        m_init_ = np.array(w_init_[:n_clusters])[:, np.newaxis]
        k_init_ = np.array(w_init_[n_clusters:2*n_clusters])[:, np.newaxis]
        w_init_ = np.array(w_init_[2*n_clusters:])[:, np.newaxis]
        w_init_ = w_init_ + np.random.uniform(0, 1)
        w_init_/= w_init_.sum()
        # Von Mises Distribution Parameters Initialization
        for k in range(n_clusters):
            # Define Variable Boundaries for Optimization
            bounds_.insert(0, (init_lim_low, 2*np.pi))
            bounds_.append((init_lim_low, init_lim_up))
            # Random Initialization of the VM Parameters
            m_init_[k] += np.random.uniform(0, 2*np.pi, 1)[0]/10.
            k_init_[k] += np.random.uniform(0, 20., 1)[0]/10.
            if m_init_[k] < 0.: m_init_[k] = 0
            if m_init_[k] > 2*np.pi: m_init_[k] = 2*np.pi
        return m_init_.copy(), k_init_.copy(), w_init_.copy(), bounds_, ll_k, dim
    # Expectation Step
    def __expectation(X_, m_, k_, w_, n_clusters):
        log_P_, log_lik_ = _von_mises_mixture_model_log_likelihood(m_, k_, X_, w_, n_clusters)
        log_w = dirichlet(np.ones(w_.shape[0]) * alpha_0).logpdf(w_)
        return np.exp(log_P_ - log_lik_), np.sum(log_lik_) + log_w, log_P_
    # Run Initialization of Expectation-Maximization algorithm
    def __EM(X_, w_init_):
        if w_init_ is None:
            m_, k_, w_, bounds_, ll_k, dim  = __rand_init(X_)
        else:
            m_, k_, w_, bounds_, ll_k, dim  = __init(X_, w_init_)
        # Stat iterative Expectation-Maximization Algorithm
        for i in range(n_iter):
            XX_ = X_.copy()
            # Expectation Step
            W_, ll_k_1, ll_k_1_ = __expectation(XX_, m_, k_, w_, n_clusters)
            if verbose:
                print('>>> No iter.: {} log-lik.: {}'.format(i, ll_k_1))
            # Check for EM convergence
            if __convergence(ll_k_1, ll_k, tol):
                break
            else:
                # Maximization Step
                m_, k_, w_, flag = __maximization(XX_, m_, k_, W_, bounds_, n_clusters)
                ll_k  = ll_k_1
                ll_k_ = ll_k_1_
            # Re-start if gradient optimization did not converge
            if not flag:
                m_, k_, w_, bounds_, ll_k, dim  = __rand_init(X_)
        # Sample Cluster Label
        z_ = np.argmax(W_, axis = 1)
        # Get Scores
        scores_ = _error_metrics(W_, ll_k_, z_, theta_ = [m_, k_, w_])
        return [m_, k_, w_, z_, ll_k, scores_]
    # Storage Variable Initialization
    x_, f_ = [], []
    # Loop over Initialization
    for i in range(n_init):
        t1 = time()
        # Run Expectation-Maximization
        res_ = __EM(X_, w_init_)
        if verbose:
            print('>>> VmMM: No. Init.: {} Time: {} s CDLL: {}'.format(i, time() - t1, res_[-2]))
        x_.append(res_)
        f_.append(res_[-1][metric])
    # Return the highest log-likelihood results
    if metric == 1:
        x_ = x_[np.argmax(f_)]
        print('>>> VmMM: CDLL: {}'.format(x_[-1][metric]))
    else:
        x_ = x_[np.argmin(f_)]
    return np.concatenate((x_[0], x_[1], np.squeeze(x_[2], axis = 1)), axis = 0), x_[4], x_[5]

# Von Meses Mixture Model with noise
def _EM_NVmMM(X_, n_clusters = 2, n_init = 3, tol = 0.1, metric = 1, n_iter = 1000,
              reg_term = 1e-5, w_init_ = None, verbose = True):
    # Check for EM algoritm convergence
    def __convergence(ll_k_1_, ll_k, tol):
        if abs(ll_k - ll_k_1_) < tol: return True
        else: return False
    # Log-Von Mises Distribution
    def __log_VM(X_, mu, kappa):
        return kappa*np.cos(X_ - mu) - np.log(2*np.pi) - np.log(iv(0, kappa))
    # Log-Uniform Distribution
    def __log_U(X_, a = 0., b = 2*np.pi):
        return uniform(a, b).logpdf(X_)
    # Log-Von Mises Distribution partial derivation
    def __grad_log_VM(X_, mu, kappa):
        g_mu_    = kappa * np.sin(X_ - mu)
        g_kappa_ = np.cos(X_ - mu) - ( iv(1, kappa)/iv(0, kappa) )
        return np.concatenate((g_mu_, g_kappa_), axis = 1)
    # Gradient Based marginal likelihood optimization
    def __optimization(X_, m_, k_, W_, bounds_):
        # Log-Likelihood Evaluation
        def ___f(x_, X_, W_):
            # Unpack Variables
            m_ = x_[:n_clusters]
            k_ = x_[n_clusters:]
            # Variable Initialization
            log_l_ = 0
            # Loop over clusters Parameters
            for k in range(n_clusters):
                # Calculate Mixture Model Marginal Log-likelihood
                log_f_ = __log_VM(X_, m_[k], k_[k])
                log_f_ = W_[:, k] * log_f_[:, 0]
                log_l_+= log_f_.sum()
            # Calculate Uniform Noise Distribution Marginal Log-likelihood
            log_f_ = __log_U(X_)
            log_f_ = W_[:, -1] * log_f_[:, 0]
            log_l_+= log_f_.sum()
            return - log_l_
        # Log-Likelihood Gradient Evaluation
        def ___g(x_, X_, W_):
            # Unpack Variables
            m_ = x_[:n_clusters]
            k_ = x_[n_clusters:]
            # Variable Initialization
            log_l_ = np.zeros((n_clusters*2, 1))
            # Loop over clusters Parameters
            for k in range(n_clusters):
                # Calculate gradient of Mixture Model Marginal Log-likelihood
                log_g_ = __grad_log_VM(X_, m_[k], k_[k])
                log_g_ = W_[:, k][:, np.newaxis] * log_g_
                log_g_ = np.sum(log_g_, axis = 0)
                log_l_[k] = log_g_[0]
                log_l_[k + n_clusters] = log_g_[1]
            return - log_l_
        # Gradient Based Optimization
        opt_ = optimize.minimize(___f, np.concatenate((m_, k_)), jac = ___g,
                                 bounds = bounds_, args = (X_, W_), method = 'SLSQP')
        return opt_['x'][:n_clusters], opt_['x'][n_clusters:]
    # Maximization Step
    def __maximization(X_, m_, k_, W_, bounds_, n_clusters):
        # Calculate the Responsabilies
        def ___resposabilities(W_):
            w_ = W_.sum(axis = 0)[:, np.newaxis] + np.finfo(W_.dtype).eps
            return w_ / W_.shape[0]
        # Probabilities and mixture model weights
        w_ = ___resposabilities(W_)
        # Optimize Distribution Parameters by Gradient
        m_, k_ = __optimization(X_, m_, k_, W_, bounds_)
        return m_, k_, w_
    # Random Initialization of VM mixture model Paramters
    def __init(X_):
        # Constants Definition
        N, dim  = X_.shape
        ll_k    = - np.inf
        # Variable Initialization
        bounds_ = []
        m_init_ = np.empty((n_clusters, dim))
        k_init_ = np.empty((n_clusters, dim))
        # Randon Initialization VM Distributions weights and Noise Distribution
        w_init_ = np.random.uniform(0, 1, n_clusters + 1)[:, np.newaxis]
        w_init_/= w_init_.sum()
        # Von Mises Distribution Parameters Initialization
        #mu = np.pi/2. + np.random.uniform(-.5, .5, 1)[0]
        for k in range(n_clusters):
            # Define Variable Boundaries for Optimization
            bounds_.insert(0, (0., 2*np.pi))
            bounds_.append((0., 150.))
            # Random Initialization of the VM Parameters
            m_init_[k] = np.random.uniform(0., 2*np.pi, 1)[0]
            k_init_[k] = np.random.uniform(0, 25., 1)[0]
            #mu = np.pi + np.random.uniform(-1., 1., 1)[0]
        return m_init_.copy(), k_init_.copy(), w_init_.copy(), bounds_, ll_k, dim
    # Calulate mixture Model Log-likelihood
    def __mixture_model_log_likelihood(m_, k_, X_, w_):
        log_P_ = np.empty((X_.shape[0], 0))
        log_w_ = np.log(w_)
        # weighted log-likelihood for each distribution
        for k in range(n_clusters):
            log_p_ = log_w_[k] + __log_VM(X_, m_[k], k_[k])
            log_P_ = np.concatenate((log_P_, log_p_), axis = 1)
        # Uniform Noise Distribution
        log_p_ = log_w_[-1] + __log_U(X_)
        log_P_ = np.concatenate((log_P_, log_p_), axis = 1)
        # Log-sum-exp of all distributions log-likelihoods
        log_lik_ = logsumexp(log_P_, axis = 1)[:, np.newaxis]
        return log_P_, log_lik_
    # Expectation Step
    def __expectation(X_, m_, k_, w_, n_clusters):
        log_P_, log_lik_ = __mixture_model_log_likelihood(m_, k_, X_, w_)
        return np.exp(log_P_ - log_lik_), np.sum(log_lik_), log_P_
    # Run Initialization of Expectation-Maximization algorithm
    def __EM(X_, w_init_):
        if w_init_ is None:
            m_, k_, w_, bounds_, ll_k, dim  = __init(X_)
        else:
            m_, k_, w_, bounds_, ll_k, dim  = w_init_
        # Stat iterative Expectation-Maximization Algorithm
        for i in range(n_iter):
            # Expectation Step
            W_, ll_k_1_, l_ = __expectation(X_, m_, k_, w_, n_clusters)
            # Maximization Step
            m_, k_, w_ = __maximization(X_, m_, k_, W_, bounds_, n_clusters)
            if verbose:
                print('>>> No iter.: {} log-lik.: {}'.format(i, ll_k_1_))
            # Check for EM convergence
            if __convergence(ll_k_1_, ll_k, tol):
                break
            else:
                ll_k = ll_k_1_
        # Sample Cluster Label
        z_ = np.argmax(W_, axis = 1)
        # Get Scores
        scores_ = _error_metrics(W_, l_, z_, theta_ = [m_, k_, w_])
        return [m_, k_, w_, z_, ll_k_1_, scores_]
    # Storage Variable Initialization
    x_, f_ = [], []
    # Loop over Initialization
    for i in range(n_init):
        if verbose: print('>>> No. Init.: {}'.format(i))
        # Run Expectation-Maximization
        res_ = __EM(X_, w_init_)
        x_.append(res_)
        f_.append(res_[-2])
    # Return the highest log-likelihood results
    x_ = x_[np.argmax(f_)]
    return np.concatenate((x_[0], x_[1], np.squeeze(x_[2], axis = 1)), axis = 0), x_[4], x_[5]


# Log-Gamma Distribution
def _log_G(X_, alpha, theta):
    log_f_ = (alpha - 1)*np.log(X_) - (X_/theta) - alpha*np.log(theta) - np.log(gamma(alpha))
    return log_f_

# Calulate mixture Model Log-likelihood
def _1Dgamma_mixture_model_log_likelihood(m_, k_, X_, w_, n_clusters):
    log_P_ = np.empty((X_.shape[0], 0))
    log_w_ = np.log(w_)
    # weighted log-likelihood for each distribution
    for k in range(n_clusters):
        log_p_ = log_w_[k] + _log_G(X_, m_[k], k_[k])
        log_P_ = np.concatenate((log_P_, log_p_), axis = 1)
    # Log-sum-exp of all distributions log-likelihoods
    log_lik_ = logsumexp(log_P_, axis = 1)[:, np.newaxis]
    return np.nan_to_num(log_P_), np.nan_to_num(log_lik_)

# 1D-Gamma Mixture Model
def _EM_GaMM(X_, n_clusters = 2, n_init = 3, tol = 0.1, n_iter = 1000, metric = 1, w_init_ = None, verbose = False, alpha_0 = 0):
    #init_lim_low = 0.01
    #init_lim_up  = 200.
    # Check for EM algoritm convergence
    def __convergence(ll_k_1_, ll_k, tol):
        if abs(ll_k - ll_k_1_) < tol: return True
        else: return False
    # Gradient of log-gamma Distribution
    def __grad_log_G(X_, alpha, theta):
        g_alpha_ = np.log(X_) - np.log(theta) - digamma(alpha)
        g_theta_ = (1./theta) * ( (X_/theta) - alpha)
        return np.concatenate((g_alpha_, g_theta_), axis = 1)
    # Gradient Based marginal likelihood optimization
    def __optimization(X_, m_, k_, W_, bounds_):
        # Log-Likelihood Evaluation
        def ___f(x_, X_, W_):
            # Unpack Variables
            m_ = x_[:n_clusters]
            k_ = x_[n_clusters:]
            # Variable Initialization
            log_l_ = 0
            # Loop over clusters Parameters
            for k in range(n_clusters):
                # Calculate Mixture Model Marginal Log-likelihood
                log_f_ = _log_G(X_, m_[k], k_[k])
                log_f_ = W_[:, k] * log_f_[:, 0]
                log_l_+= log_f_.sum()
            return - log_l_
        # Log-Likelihood Gradient Evaluation
        def ___g(x_, X_, W_):
            # Unpack Variables
            m_ = x_[:n_clusters]
            k_ = x_[n_clusters:]
            # Variable Initialization
            log_l_ = np.zeros((n_clusters*2, 1))
            # Loop over clusters Parameters
            for k in range(n_clusters):
                # Calculate gradient of Mixture Model Marginal Log-likelihood
                log_g_ = __grad_log_G(X_, m_[k], k_[k])
                log_g_ = W_[:, k][:, np.newaxis] * log_g_
                log_g_ = np.sum(log_g_, axis = 0)
                log_l_[k] = log_g_[0]
                log_l_[k + n_clusters] = log_g_[1]
            return - log_l_
        # Gradient Based Optimization
        opt_ = optimize.minimize(fun = ___f, jac = ___g, x0 = np.concatenate((m_, k_)),
                                 bounds = bounds_, args = (X_, W_), method = 'SLSQP')
        return opt_['x'][:n_clusters], opt_['x'][n_clusters:], opt_['success']
    # Maximization Step
    def __maximization(X_, m_, k_, W_, bounds_, n_clusters):
        # Calculate the Responsabilies
        def ___resposabilities(W_):
            return W_.sum(axis = 0)[:, np.newaxis] + np.finfo(W_.dtype).eps
        # Regularization via Variational Inverence
        def ___variational_prior(w_):
            if alpha_0 != 0.:
                alpha_ = alpha_0 + w_
                w_ = []
                for alpha in alpha_:
                    w_.append(np.exp(digamma(alpha) - digamma(alpha_.sum())))
                return np.stack(w_)
            else:
                return w_ / W_.shape[0]
        # Regularization via MAP
        def ___MAP_prior(w_, N, K):
            alpha_ = np.ones(w_.shape) * alpha_0
            return (w_ + alpha_ - 1.)/(N - K + alpha_.sum())
        # Probabilities and mixture model weights
        w_ = ___resposabilities(W_)
        # Force weightes to be within the limit
        #w_ = ___variational_prior(w_)
        w_ = ___MAP_prior(w_, N = W_.shape[0], K = W_.shape[1])
        # Optimize Distribution Parameters by Gradient
        m_, k_, flag = __optimization(X_, m_, k_, W_, bounds_)
        return m_, k_, w_, flag
    # Random Initialization of VM mixture model Paramters
    def __rand_init(X_):
        init_lim_low = X_.min()
        init_lim_up  = X_.max()
        # Constants Definition
        N, dim  = X_.shape
        ll_k    = - np.inf
        # Variable Initialization
        bounds_ = []
        m_init_ = np.empty((n_clusters, dim))
        k_init_ = np.empty((n_clusters, dim))
        # Randon Initialization VM Distributions weights and Noise Distribution
        w_init_ = np.random.uniform(0, 1, n_clusters)[:, np.newaxis]
        w_init_/= w_init_.sum()
        for k in range(n_clusters):
            # Define Variable Boundaries for Optimization
            bounds_.insert(0, (1e-5, 500.))
            bounds_.append((1e-5, 500.))
            # Random Initialization of the Gamma Parameters
            m_init_[k] = np.random.uniform(init_lim_low, init_lim_up, 1)[0]
            k_init_[k] = np.random.uniform(init_lim_low, init_lim_up, 1)[0]
        return m_init_.copy(), k_init_.copy(), w_init_.copy(), bounds_, ll_k, dim

    def __init(X_, w_init_):
        init_lim_low = X_.min()
        init_lim_up  = X_.max()
        # Constants Definition
        N, dim = X_.shape
        ll_k   = - np.inf
        # Variable Initialization
        bounds_ = []
        # Randon Initialization VM Distributions weights
        m_init_ = np.array(w_init_[:n_clusters])[:, np.newaxis]
        k_init_ = np.array(w_init_[n_clusters:2*n_clusters])[:, np.newaxis]
        w_init_ = np.array(w_init_[2*n_clusters:])[:, np.newaxis]
        w_init_ = w_init_ + np.random.uniform(0, 1)
        w_init_/= w_init_.sum()
        # Von Mises Distribution Parameters Initialization
        for k in range(n_clusters):
            # Define Variable Boundaries for Optimization
            bounds_.insert(0, (1e-5, 500.))
            bounds_.append((1e-5, 500.))
            # Random Initialization of the Gamma Parameters
            m_init_[k] += np.random.uniform(init_lim_low, init_lim_up, 1)[0]/10
            k_init_[k] += np.random.uniform(init_lim_low, init_lim_up, 1)[0]/10
            if m_init_[k] < 0.: m_init_[k] = 0.
            if k_init_[k] < 0.: k_init_[k] = 0.
        return m_init_.copy(), k_init_.copy(), w_init_.copy(), bounds_, ll_k, dim
    # Expectation Step
    def __expectation(X_, m_, k_, w_, n_clusters):
        log_P_, log_lik_ = _1Dgamma_mixture_model_log_likelihood(m_, k_, X_, w_, n_clusters)
        log_w = dirichlet(np.ones(w_.shape[0]) * alpha_0).logpdf(w_)
        return np.exp(log_P_ - log_lik_), np.sum(log_lik_) + log_w, log_P_
    def _rise_flag(m_, k_, w_, ll_, ll):
        if np.isnan(m_).any() or np.isnan(k_).any() or np.isnan(w_).any() or np.isnan(ll_).any() or np.isnan(ll):
            return False
        if np.isinf(m_).any() or np.isinf(k_).any() or np.isinf(w_).any() or np.isinf(ll_).any() or np.isinf(ll):
            return False
        if ll == 0.0:
            return False
        return True
    # Run Initialization of Expectation-Maximization algorithm
    def __EM(X_, w_init_):
        if w_init_ is None:
            m_, k_, w_, bounds_, ll_k, dim  = __rand_init(X_)
        else:
            m_, k_, w_, bounds_, ll_k, dim  = __init(X_, w_init_)
        # Stat iterative Expectation-Maximization Algorithm
        for i in range(n_iter):
            XX_ = X_.copy()
            # Expectation Step
            W_, ll_k_1, ll_k_1_ = __expectation(XX_, m_, k_, w_, n_clusters)
            if verbose:
                print('>>> No iter.: {} Data log-likelihood: {}'.format(i, ll_k_1))
            # Check for EM convergence
            if __convergence(ll_k_1, ll_k, tol):
                break
            else:
                # Maximization Step
                m_, k_, w_, flag1 = __maximization(XX_, m_, k_, W_, bounds_, n_clusters)
                flag2 = _rise_flag(m_, k_, w_, ll_k_1_, ll_k_1)
                ll_k  = ll_k_1
                ll_k_ = ll_k_1_
            # Optimization Error re-start iteration
            if not flag1 or not flag2:
                X_ += 1e-5
                m_, k_, w_, bounds_, ll_k, dim  = __rand_init(X_)
        # Sample Cluster Label
        z_ = np.argmax(W_, axis = 1)
        # Get Scores
        scores_ = _error_metrics(W_, ll_k_, z_, theta_ = [m_, k_, w_])
        return [m_, k_, w_, z_, ll_k, scores_]
    # Storage Variable Initialization
    x_, f_ = [], []
    # Loop over Initialization
    for i in range(n_init):
        t1 = time()
        # Run Expectation-Maximization
        res_ = __EM(X_, w_init_)
        if verbose:
            print('>>> GaMM: No. Init.: {} Time: {} s CDLL: {}'.format(i, time() - t1, res_[-2]))
        x_.append(res_)
        f_.append(res_[-1][metric])
    # Return the highest log-likelihood results
    if metric == 1:
        x_ = x_[np.argmax(f_)]
        print('>>> GaMM: CDLL: {}'.format(x_[-1][metric]))
    else:
        x_ = x_[np.argmin(f_)]
    return np.concatenate((x_[0], x_[1], np.squeeze(x_[2], axis = 1)), axis = 0), x_[4], x_[5]

# Log-Von Mises Distribution
def _log_VM(X_, mu, kappa):
    return kappa*np.cos(X_ - mu) - np.log(2*np.pi) - np.log(iv(0, kappa))

# Von Mises Distribution
def _VM(theta_, mu, kappa):
    z_1 = np.exp(kappa*np.cos(theta_ - mu))
    z_2 = 2*np.pi * iv(0, kappa)
    return np.squeeze(z_1 / z_2)

# Draw a Sample from a Von Mises Distribution
def _draw_sample_VM(n_samples = 1, mu = -1., kappa = 7., a = 0., b = 2*np.pi):
    x_ = uniform(a, 2*b).rvs(100*n_samples)
    z_ = _VM(x_, mu = -1., kappa = 7.)
    return np.random.choice(x_, n_samples, p = z_/z_.sum())

# Von Mises Mixture Model
def _VMM(x_, X_):
    n_clusters = int(x_.shape[0]/3)
    # Sample Probabilities Initialization
    Z_ = np.zeros((X_.shape[0], n_clusters))
    # (1) Von Mises Distribution Probabilities
    for k in range(n_clusters):
        # Unpuck Cluster Parameters
        mu    = x_[k:k + 1]
        kappa = x_[k + n_clusters:k + n_clusters + 1]
        w     = x_[k + 2*n_clusters:k + 2*n_clusters + 1]
        # Evaluate Cluster Probabilities
        Z_[:, k] = w * _VM(X_, mu, kappa)
    # Sample Probabilities
    #z_ = np.sum(Z_, axis = 1)
    # Cluster label
    k_ = np.argmax(Z_, axis = 1)
    return Z_, k_

# Uniform Distribution
def _U(X_, a = 0., b = 2*np.pi):
    return uniform(a, b).pdf(X_)

# Von Mises Mixture Model with noise
def _NVmMM(x_, X_):
    n_clusters = int(x_.shape[0]/3)
    # Sample Probabilities Initialization
    Z_ = np.zeros((X_.shape[0], n_clusters + 1))
    # (1) Von Mises Distribution Probabilities
    for k in range(n_clusters):
        # Unpuck Cluster Parameters
        mu    = x_[k:k + 1]
        kappa = x_[k + n_clusters:k + n_clusters + 1]
        w     = x_[k + 2*n_clusters:k + 2*n_clusters + 1]
        # Evaluate Cluster Probabilities
        Z_[:, k] = w * _VM(X_, mu, kappa)
    # Evaluate Uniform Probabilities
    Z_[:, -1] = x_[-1:] * _U(X_)
    # Cluster label
    k_ = np.argmax(Z_, axis = 1)
    return Z_, k_

# Gamma Distribution
def _1D_G(X_, alpha, theta):
    f_ = ( (X_**(alpha - 1)) * (np.exp(- (X_/theta))) ) / ( gamma(alpha)*theta**alpha )
    return f_

# Gamma Mixture Model
def _GaMM(x_, X_):
    n_clusters = int(x_.shape[0]/3)
    # Sample Probabilities Initialization
    Z_ = np.zeros((X_.shape[0], n_clusters))
    # (1) Gamma Distribution Probabilities
    for k in range(n_clusters):
        # Unpuck Cluster Parameters
        alpha = x_[k:k + 1]
        theta = x_[k + n_clusters:k + n_clusters + 1]
        w     = x_[k + 2*n_clusters:k + 2*n_clusters + 1]
        # Evaluate Cluster Probabilities
        Z_[:, k] = w * _1D_G(X_, alpha, theta)
    # Cluster label
    k_ = np.argmax(Z_, axis = 1)
    return Z_, k_

def _B(X_, a, b):
    return ( (X_**(a - 1.))*((1. - X_)**(b - 1.)) ) / beta(a, b)

# Beta Mixture Model Distribution
def _BeMM(x_, X_):
    n_clusters = int(x_.shape[0]/3)
    # Sample Probabilities Initialization
    Z_ = np.zeros((X_.shape[0], n_clusters))
    # (1) Beta Distribution Probabilities
    for k in range(n_clusters):
        # Unpuck Cluster Parameters
        a = x_[k:k + 1]
        b = x_[k + n_clusters:k + n_clusters + 1]
        w = x_[k + 2*n_clusters:k + 2*n_clusters + 1]
        # Evaluate Cluster Probabilities
        Z_[:, k] = w * _B(X_, a, b)
    # Cluster label
    k_ = np.argmax(Z_, axis = 1)
    return Z_, k_

# Log-beta Distribution
def _log_beta(X_, a, b):
    log_f_ = - np.log(beta(a, b)) - np.log(X_) + a*np.log(X_) - np.log(1. - X_) + b*np.log(1. - X_)
    return log_f_

# Calulate mixture Model Log-likelihood
def _beta_mixture_model_log_likelihood(a_, b_, X_, w_, n_clusters):
    log_P_ = np.empty((X_.shape[0], 0))
    log_w_ = np.log(w_)
    #X_ -= 1e-10
    # weighted log-likelihood for each distribution
    for k in range(n_clusters):
        log_p_ = log_w_[k] + _log_beta(X_, a_[k], b_[k])
        log_P_ = np.concatenate((log_P_, log_p_), axis = 1)
    # Log-sum-exp of all distributions log-likelihoods
    log_lik_ = logsumexp(log_P_, axis = 1)[:, np.newaxis]
    return np.nan_to_num(log_P_), np.nan_to_num(log_lik_)

# Beta Mixture Model Inferece
def _EM_BeMM(X_, n_clusters = 2, n_init = 3, tol = 0.1, metric = 1, n_iter = 1000,
             verbose = False, alpha_0 = 0.):
    init_lim_low = .01
    init_lim_up  = 500.
    # Check for EM algoritm convergence
    def __convergence(ll_k_1_, ll_k, tol):
        if abs(ll_k - ll_k_1_) < tol: return True
        else: return False
    # Gradient of log-beta Distribution
    def __grad_log_beta(X_, a, b):
        g_a_ = np.log(X_) - ( beta(a, b) * (digamma(a) - digamma( a + b )) )/beta(a, b)
        g_b_ = np.log(1. - X_) - ( beta(a, b) * (digamma(b) - digamma( b + a )) )/beta(a, b)
        return np.concatenate((g_a_, g_b_), axis = 1)
    # Gradient Based marginal likelihood optimization
    def __optimization(X_, a_, b_, W_, bounds_):
        # Log-Likelihood Evaluation
        def ___f(x_, X_, W_):
            # Unpack Variables
            a_ = x_[:n_clusters]
            b_ = x_[n_clusters:]
            # Variable Initialization
            log_l_ = 0
            # Loop over clusters Parameters
            for k in range(n_clusters):
                # Calculate Mixture Model Marginal Log-likelihood
                log_f_ = _log_beta(X_, a_[k], b_[k])
                log_f_ = W_[:, k] * log_f_[:, 0]
                log_l_+= log_f_.sum()
            return - log_l_
        # Log-Likelihood Gradient Evaluation
        def ___g(x_, X_, W_):
            # Unpack Variables
            a_ = x_[:n_clusters]
            b_ = x_[n_clusters:]
            # Variable Initialization
            log_l_ = np.zeros((n_clusters*2, 1))
            # Loop over clusters Parameters
            for k in range(n_clusters):
                # Calculate gradient of Mixture Model Marginal Log-likelihood
                log_g_ = __grad_log_beta(X_, a_[k], b_[k])
                log_g_ = W_[:, k][:, np.newaxis] * log_g_
                log_g_ = np.sum(log_g_, axis = 0)
                log_l_[k] = log_g_[0]
                log_l_[k + n_clusters] = log_g_[1]
            return - log_l_
        x0 = np.concatenate((a_, b_))
        # Gradient Based Optimization
        opt_ = optimize.minimize(fun = ___f, x0 = np.concatenate((a_, b_)),
                                 bounds = bounds_, args = (X_, W_), method = 'SLSQP')
        return opt_['x'][:n_clusters], opt_['x'][n_clusters:], opt_['success']
    # Maximization Step
    def __maximization(X_, a_, b_, W_, bounds_, n_clusters):
        # Calculate the Responsabilies
        def ___resposabilities(W_):
            return W_.sum(axis = 0)[:, np.newaxis] + np.finfo(W_.dtype).eps
        # Regularization via Variational Inverence
        def ___variational_prior(w_):
            if alpha_0 != 0.:
                alpha_ = alpha_0 + w_
                w_ = []
                for alpha in alpha_:
                    w_.append(np.exp(digamma(alpha) - digamma(alpha_.sum())))
                return np.stack(w_)
            else:
                return w_ / W_.shape[0]
        # Regularization via MAP
        def ___MAP_prior(w_, N, K):
            alpha_ = np.ones(w_.shape) * alpha_0
            return (w_ + alpha_ - 1.)/(N - K + alpha_.sum())
        # Probabilities and mixture model weights
        w_ = ___resposabilities(W_)
        # Force weightes to be within the limit
        #w_ = ___variational_prior(w_)
        w_ = ___MAP_prior(w_, N = W_.shape[0], K = W_.shape[1])
        # Optimize Distribution Parameters by Gradient
        a_, b_, flag = __optimization(X_, a_, b_, W_, bounds_)
        return a_, b_, w_, flag
    # Random Initialization of VM mixture model Paramters
    def __init(X_):
        # Constants Definition
        N, dim  = X_.shape
        ll_k    = - np.inf
        # Variable Initialization
        bounds_ = []
        a_init_ = np.empty((n_clusters, dim))
        b_init_ = np.empty((n_clusters, dim))
        # Randon Initialization VM Distributions weights and Noise Distribution
        w_init_ = np.random.uniform(0, 1, n_clusters)[:, np.newaxis]
        w_init_/= w_init_.sum()
        # Beta Distribution Parameters Initialization
        for k in range(n_clusters):
            # Define Variable Boundaries for Optimization
            bounds_.insert(0, (init_lim_low, init_lim_up))
            bounds_.append((init_lim_low, init_lim_up))
            # Random Initialization of the Beta Parameters
            a_init_[k] = np.random.uniform(init_lim_low, 100, 1)[0]
            b_init_[k] = np.random.uniform(init_lim_low, 100, 1)[0]
        return a_init_.copy(), b_init_.copy(), w_init_.copy(), bounds_, a_init_, b_init_, w_init_, ll_k, dim
    # Expectation Step
    def __expectation(X_, a_, b_, w_, n_clusters):
        log_P_, log_lik_ = _beta_mixture_model_log_likelihood(a_, b_, X_, w_, n_clusters)
        log_w = dirichlet(np.ones(w_.shape[0]) * alpha_0).logpdf(w_)
        return np.exp(log_P_ - log_lik_), np.sum(log_lik_) + log_w, log_P_
    # Run Initialization of Expectation-Maximization algorithm
    def __EM(X_):
        a_, b_, w_, bounds_, a_init_, b_init_, w_init_, ll_k, dim = __init(X_)
        # Stat iterative Expectation-Maximization Algorithm
        for i in range(n_iter):
            XX_ = X_.copy()
            # Expectation Step
            W_, ll_k_1, ll_k_1_ = __expectation(XX_, a_, b_, w_, n_clusters)
            if verbose:
                print('>>> No iter.: {} log-lik.: {}'.format(i, ll_k_1))
            # Check for EM convergence
            if __convergence(ll_k_1, ll_k, tol):
                break
            else:
                # Maximization Step
                a_, b_, w_, flag = __maximization(XX_, a_, b_, W_, bounds_, n_clusters)
                ll_k  = ll_k_1
                ll_k_ = ll_k_1_
            # Re-start Optimization of there was any error
            if not flag:
                a_, b_, w_, bounds_, a_init_, b_init_, w_init_, ll_k, dim = __init(X_)
        # Sample Cluster Label
        z_ = np.argmax(W_, axis = 1)
        # Get Scores
        scores_ = _error_metrics(W_, ll_k_, z_, theta_ = [a_, b_, w_])
        return [a_, b_, w_, z_, ll_k, scores_]
    # Storage Variable Initialization
    x_, f_ = [], []
    # Loop over Initialization
    for i in range(n_init):
        t1 = time()
        # Run Expectation-Maximization
        res_ = __EM(X_)
        if verbose:
            print('>>> BeMM: No. Init.: {} Time: {} s CDLL: {}'.format(i, time() - t1, res_[-2]))
        x_.append(res_)
        f_.append(res_[-1][metric])
    # Return the highest log-likelihood results
    if metric == 1:
        x_ = x_[np.argmax(f_)]
        print('>>> BeMM: CDLL: {}'.format(x_[-1][metric]))
    else:
        x_ = x_[np.argmin(f_)]
    return np.concatenate((x_[0], x_[1], np.squeeze(x_[2], axis = 1)), axis = 0), x_[4], x_[5]

# Log-2D-Gamma Distribution
def _log_2D_G(X_, alpha, beta, kappa):
    log_f_ = alpha*np.log(beta) + (alpha + kappa - 1.)*np.log(X_[:, 0]) + (kappa - 1.)*np.log(X_[:, 1])
    log_f_ += - beta*X_[:, 0] - X_[:, 0]*X_[:, 1] - np.log(gamma(alpha)) - np.log(gamma(kappa))
    return log_f_[:, np.newaxis]

# Calulate mixture Model Log-likelihood
def _2Dgamma_mixture_model_log_likelihood(a_, b_, k_, X_, w_, n_clusters):
    log_P_ = np.empty((X_.shape[0], 0))
    log_w_ = np.log(w_)
    # weighted log-likelihood for each distribution
    for k in range(n_clusters):
        log_p_ = log_w_[k] + _log_2D_G(X_, a_[k], b_[k], k_[k])
        log_P_ = np.concatenate((log_P_, log_p_), axis = 1)
    # Log-sum-exp of all distributions log-likelihoods
    log_lik_ = logsumexp(log_P_, axis = 1)[:, np.newaxis]
    return log_P_, log_lik_

# 2D-Gamma Mixture Model
def _EM_2D_GaMM(X_, n_clusters = 2, n_init = 3, tol = 0.1, metric = 1, n_iter = 1000,
                reg_term = 1e-5, w_init_ = None, verbose = False, alpha_0 = 0.):
    init_lim_low = 1.
    init_lim_up  = 100.
    # Check for EM algoritm convergence
    def __convergence(ll_k_1_, ll_k, tol):
        if abs(ll_k - ll_k_1_) < tol: return True
        else: return False
    # Gradient of log-2D-Gamma Distribution
    def __grad_log_2D_G(X_, alpha, beta, kappa):
        N = X_.shape[0]
        g_alpha_ = np.log(beta) + np.log(X_[:, 0]) - digamma(alpha)
        g_beta_  = alpha/beta - X_[:, 0]
        g_kappa_ = np.log(X_[:, 0]) + np.log(X_[:, 1]) - digamma(kappa)
        return np.concatenate((g_alpha_[:, np.newaxis], g_beta_[:, np.newaxis], g_kappa_[:, np.newaxis]), axis = 1)
    # Gradient Based marginal likelihood optimization
    def __optimization(X_, a_, b_, k_, W_, bounds_):
        # Log-Likelihood Evaluation
        def ___f(x_, X_, W_):
            n_dist_theta = 3
            # Unpack Variables
            a_ = x_[:n_clusters]
            b_ = x_[n_clusters:2*n_clusters]
            k_ = x_[2*n_clusters:]
            # Variable Initialization
            log_l_ = 0
            # Loop over clusters Parameters
            for k in range(n_clusters):
                # Calculate Mixture Model Marginal Log-likelihood
                log_f_ = _log_2D_G(X_, a_[k], b_[k], k_[k])
                log_f_ = W_[:, k] * log_f_[:, 0]
                log_l_+= log_f_.sum()
            return - log_l_
        # Log-Likelihood Gradient Evaluation
        def ___g(x_, X_, W_):
            n_dist_theta = 3
            # Unpack Variables
            a_ = x_[:n_clusters]
            b_ = x_[n_clusters:2*n_clusters]
            k_ = x_[2*n_clusters:]
            # Variable Initialization
            log_l_ = np.zeros((n_clusters*n_dist_theta, 1))
            # Loop over clusters Parameters
            for k in range(n_clusters):
                # Calculate gradient of Mixture Model Marginal Log-likelihood
                log_g_ = __grad_log_2D_G(X_, a_[k], b_[k], k_[k])
                log_g_ = W_[:, k][:, np.newaxis] * log_g_
                log_g_ = np.mean(log_g_, axis = 0)
                log_l_[k] = log_g_[0]
                log_l_[k + n_clusters] = log_g_[1]
                log_l_[k + 2*n_clusters] = log_g_[2]
            return - log_l_
        x_ = np.concatenate((a_, b_, k_))
        # Gradient Based Optimization
        opt_ = optimize.minimize(___f, x_, jac = ___g, bounds = bounds_, args = (X_, W_), method = 'SLSQP')
        return opt_['x'][:n_clusters], opt_['x'][n_clusters:2*n_clusters], opt_['x'][2*n_clusters:3*n_clusters:]
    # Maximization Step
    def __maximization(X_, a_, b_, k_, W_, bounds_, n_clusters):
        # Calculate the Responsabilies
        def ___resposabilities(W_):
            return W_.sum(axis = 0)[:, np.newaxis] + np.finfo(W_.dtype).eps
        # Regularization via Variational Inverence
        def ___variational_prior(w_):
            if alpha_0 != 0.:
                alpha_ = alpha_0 + w_
                w_ = []
                for alpha in alpha_:
                    w_.append(np.exp(digamma(alpha) - digamma(alpha_.sum())))
                return np.stack(w_)
            else:
                return w_ / W_.shape[0]
        # Regularization via MAP
        def ___MAP_prior(w_, N, K):
            alpha_ = np.ones(w_.shape) * alpha_0
            return (w_ + alpha_ - 1.)/(N - K + alpha_.sum())
        # Probabilities and mixture model weights
        w_ = ___resposabilities(W_)
        # Force weightes to be within the limit
        #w_ = ___variational_prior(w_)
        w_ = ___MAP_prior(w_, N = W_.shape[0], K = W_.shape[1])
        # Optimize Distribution Parameters by Gradient
        a_, b_, k_ = __optimization(X_, a_, b_, k_, W_, bounds_)
        return a_, b_, k_, w_
    # Random Initialization of VM mixture model Paramters
    def __init(X_):
        # Constants Definition
        N, dim  = X_.shape
        ll_k    = - np.inf
        # Variable Initialization
        bounds_ = []
        a_init_ = np.empty((n_clusters))
        b_init_ = np.empty((n_clusters))
        k_init_ = np.empty((n_clusters))
        # Randon Initialization VM Distributions weights and Noise Distribution
        w_init_  = np.random.uniform(0, 1, n_clusters)[:, np.newaxis]
        w_init_ /= w_init_.sum()
        # Gamma Distribution Parameters Initialization
        for k in range(n_clusters):
            # Define Variable Boundaries for Optimization
            bounds_.insert(0, (.1, init_lim_up))
            bounds_.append((.1, init_lim_up))
            bounds_.append((.1, init_lim_up))
            # Random Initialization of the Gamma Parameters
            a_init_[k] = np.random.uniform(init_lim_low, init_lim_up, 1)[0]
            b_init_[k] = np.random.uniform(init_lim_low, init_lim_up, 1)[0]
            k_init_[k] = np.random.uniform(init_lim_low, init_lim_up, 1)[0]
        return a_init_.copy(), b_init_.copy(), k_init_.copy(), w_init_.copy(), bounds_, ll_k, dim
    # Expectation Step
    def __expectation(X_, a_, b_, k_, w_, n_clusters):
        log_P_, log_lik_ = _2Dgamma_mixture_model_log_likelihood(a_, b_, k_, X_, w_, n_clusters)
        log_w = dirichlet(np.ones(w_.shape[0]) * alpha_0).logpdf(w_)
        return np.exp(log_P_ - log_lik_), np.sum(log_lik_) + log_w, log_P_
    # Run Initialization of Expectation-Maximization algorithm
    def __EM(X_, w_init_):
        if w_init_ is None:
            a_, b_, k_, w_, bounds_, ll_k, dim = __init(X_)
        else:
            a_, b_, k_, w_, bounds_, ll_k, dim = w_init_
        # Stat iterative Expectation-Maximization Algorithm
        for i in range(n_iter):
            # Expectation Step
            W_, ll_k_1, ll_k_1_ = __expectation(X_, a_, b_, k_, w_, n_clusters)

            if verbose:
                print('>>> No iter.: {} log-lik.: {}'.format(i, ll_k_1))
            # Check for EM convergence
            if __convergence(ll_k_1, ll_k, tol):
                break
            else:
                # Maximization Step
                a_, b_, k_, w_ = __maximization(X_, a_, b_, k_, W_, bounds_, n_clusters)
                ll_k  = ll_k_1
                ll_k_ = ll_k_1_
        # Sample Cluster Label
        z_ = np.argmax(W_, axis = 1)
        # Get Scores
        scores_ = _error_metrics(W_, ll_k_, z_, theta_ = [a_, b_, k_, w_])
        return [a_, b_, k_, w_, z_, ll_k_1, scores_]
    # Storage Variable Initialization
    x_, f_ = [], []
    # Loop over Initialization
    for i in range(n_init):
        t1 = time()
        # Run Expectation-Maximization
        res_ = __EM(X_, w_init_)
        if verbose:
            print('>>> BGaMM: No. Init.: {} Time: {}'.format(i, time() - t1))
        x_.append(res_)
        f_.append(res_[-1][metric])
    # Return the highest log-likelihood results
    if metric == 1:
        x_ = x_[np.argmax(f_)]
        print('>>> BGaMM: CDLL: {}'.format(x_[-1][metric]))
    else:
        x_ = x_[np.argmin(f_)]
    # Return the highest log-likelihood results
    #x_ = x_[np.argmax(f_)]
    return np.concatenate((x_[0], x_[1], x_[2], np.squeeze(x_[3], axis = 1)), axis = 0), x_[5], x_[6]


# 2D-Gamma with geometric transformation Mixture Model
def _EM_2DT_GaMM(X_, Z_, n_clusters = 2, n_init = 3, tol = 0.1, n_iter = 1000,
                 reg_term = 1e-5, w_init_ = None, verbose = False):
    # Check for EM algoritm convergence
    def __convergence(ll_k_1_, ll_k, tol):
        if abs(ll_k - ll_k_1_) < tol: return True
        else: return False
    # Log-2D-Gamma Distribution
    def __log_2D_G(X_, alpha, beta, kappa):
        log_f_ = alpha*np.log(beta) + (alpha + kappa - 1.)*np.log(X_[:, 0]) + (kappa - 1.)*np.log(X_[:, 1])
        log_f_ += - beta*X_[:, 0] - X_[:, 0]*X_[:, 1] - np.log(gamma(alpha)) - np.log(gamma(kappa))
        return log_f_[:, np.newaxis]
    # Gradient of log-2D-Gamma Distribution
    def __grad_log_2D_G(X_, alpha, beta, kappa):
        N = X_.shape[0]
        g_alpha_ = np.log(beta) + np.log(X_[:, 0]) - digamma(alpha)
        g_beta_  = alpha/beta - X_[:, 0]
        g_kappa_ = np.log(X_[:, 0]) + np.log(X_[:, 1]) - digamma(kappa)
        return np.concatenate((g_alpha_[:, np.newaxis], g_beta_[:, np.newaxis], g_kappa_[:, np.newaxis]), axis = 1)
    # Gradient Based marginal likelihood optimization
    def __optimization(X_, a_, b_, k_, W_, bounds_):
        # Log-Likelihood Evaluation
        def ___f(x_, X_, W_):
            n_dist_theta = 3
            # Unpack Variables
            a_ = x_[:n_clusters]
            b_ = x_[n_clusters:2*n_clusters]
            k_ = x_[2*n_clusters:]
            # Variable Initialization
            log_l_ = 0
            # Loop over clusters Parameters
            for k in range(n_clusters):
                # Calculate Mixture Model Marginal Log-likelihood
                log_f_ = s_2D_G(X_, a_[k], b_[k], k_[k])
                log_f_ = W_[:, k] * log_f_[:, 0]
                log_l_+= log_f_.sum()
            return - log_l_
        # Log-Likelihood Gradient Evaluation
        def ___g(x_, X_, W_):
            n_dist_theta = 3
            # Unpack Variables
            a_ = x_[:n_clusters]
            b_ = x_[n_clusters:2*n_clusters]
            k_ = x_[2*n_clusters:]
            # Variable Initialization
            log_l_ = np.zeros((n_clusters*n_dist_theta, 1))
            # Loop over clusters Parameters
            for k in range(n_clusters):
                # Calculate gradient of Mixture Model Marginal Log-likelihood
                log_g_ = __grad_log_2D_G(X_, a_[k], b_[k], k_[k])
                log_g_ = W_[:, k][:, np.newaxis] * log_g_
                log_g_ = np.mean(log_g_, axis = 0)
                log_l_[k] = log_g_[0]
                log_l_[k + n_clusters] = log_g_[1]
                log_l_[k + 2*n_clusters] = log_g_[2]
            return - log_l_
        x_ = np.concatenate((a_, b_, k_))
        # Gradient Based Optimization
        opt_ = optimize.minimize(___f, x_, jac = ___g, bounds = bounds_, args = (X_, W_), method = 'SLSQP')
        #print(opt_['x'], opt_['x'].shape, n_clusters, 2*n_clusters, 3*n_clusters)
        return opt_['x'][:n_clusters], opt_['x'][n_clusters:2*n_clusters], opt_['x'][2*n_clusters:3*n_clusters:]
    # Maximization Step
    def __maximization(X_, a_, b_, k_, W_, bounds_, n_clusters):
        # Calculate the Responsabilies
        def ___resposabilities(W_):
            w_ = W_.sum(axis = 0)[:, np.newaxis] + np.finfo(W_.dtype).eps
            return w_ / W_.shape[0]
        # Probabilities and mixture model weights
        w_ = ___resposabilities(W_)
        # Optimize Distribution Parameters by Gradient
        a_, b_, k_ = __optimization(X_, a_, b_, k_, W_, bounds_)
        return a_, b_, k_, w_
    # Random Initialization of VM mixture model Paramters
    def __init(X_):
        # Constants Definition
        N, dim  = X_.shape
        ll_k    = - np.inf
        # Variable Initialization
        bounds_ = []
        a_init_ = np.empty((n_clusters))
        b_init_ = np.empty((n_clusters))
        k_init_ = np.empty((n_clusters))
        # Randon Initialization VM Distributions weights and Noise Distribution
        w_init_  = np.random.uniform(0, 1, n_clusters)[:, np.newaxis]
        w_init_ /= w_init_.sum()
        # Gamma Distribution Parameters Initialization
        for k in range(n_clusters):
            # Define Variable Boundaries for Optimization
            bounds_.insert(0, (.1, 500.))
            bounds_.append((.1, 500.))
            bounds_.append((.1, 500.))
            # Random Initialization of the Gamma Parameters
            a_init_[k] = np.random.uniform(1., 5., 1)[0]
            b_init_[k] = np.random.uniform(1., 5., 1)[0]
            k_init_[k] = np.random.uniform(1., 5., 1)[0]
        return a_init_.copy(), b_init_.copy(), k_init_.copy(), w_init_.copy(), bounds_, ll_k, dim
    # Calulate mixture Model Log-likelihood
    def __mixture_model_log_likelihood(a_, b_, k_, X_, w_):
        log_P_ = np.empty((X_.shape[0], 0))
        log_w_ = np.log(w_)
        # weighted log-likelihood for each distribution
        for k in range(n_clusters):
            log_p_ = log_w_[k] + __log_2D_G(X_, a_[k], b_[k], k_[k])
            log_P_ = np.concatenate((log_P_, log_p_), axis = 1)
        # Log-sum-exp of all distributions log-likelihoods
        log_lik_ = logsumexp(log_P_, axis = 1)[:, np.newaxis]
        return log_P_, log_lik_
    # Expectation Step
    def __expectation(X_, a_, b_, k_, w_, n_clusters):
        log_P_, log_lik_ = __mixture_model_log_likelihood(a_, b_, k_, X_, w_)
        return np.exp(log_P_ - log_lik_), np.sum(log_lik_), log_P_

    def __geo_trans(X_, Z_, W_, n_clusters):
        X_p_ = X_.copy()
        z_ = np.argmax(W_, axis = 1)
        for i in range(n_clusters):
            idx_ = z_ == i
            X_p_[idx_, 1] = np.median(X_[idx_, 0]) * Z_[idx_, 0] * X_p_[idx_, 1] / 1000.
        return X_p_

    # Run Initialization of Expectation-Maximization algorithm
    def __EM(X_, Z_, w_init_):
        if w_init_ is None:
            alpha_, beta_, kappa_, weight_, bounds_, ll_k, dim = __init(X_)
        else:
            alpha_, beta_, kappa_, w_, bounds_, ll_k, dim = w_init_

        X_p_ = X_.copy()
        X_p_[:, 1] = np.median(X_[:, 0]) * Z_[:, 0] * X_p_[:, 1] / 1000.
        # Stat iterative Expectation-Maximization Algorithm
        for i in range(n_iter):
            # Expectation Step
            W_, ll_k_1, l_ = __expectation(X_p_, alpha_, beta_, kappa_, weight_, n_clusters)
            # Maximization Step
            alpha_, beta_, kappa_, weight_ = __maximization(X_p_, alpha_, beta_, kappa_, W_, bounds_, n_clusters)
            if verbose:
                print('>>> No iter.: {} log-lik.: {}'.format(i, ll_k_1))
            # Check for EM convergence
            if __convergence(ll_k_1, ll_k, tol): break
            else:
                ll_k = ll_k_1
                X_p_ = __geo_trans(X_, Z_, W_, n_clusters)

        # Sample Cluster Label
        z_ = np.argmax(W_, axis = 1)
        # Get Scores
        scores_ = _error_metrics(W_, l_, z_, theta_ = [alpha_, beta_, kappa_, weight_])
        return [alpha_, beta_, kappa_, weight_, z_, X_p_, ll_k_1, scores_]
    # Storage Variable Initialization
    x_, f_ = [], []
    # Loop over Initialization
    for i in range(n_init):
        if verbose: print('>>> No. Init.: {}'.format(i))
        # Run Expectation-Maximization
        res_ = __EM(X_, Z_, w_init_)
        x_.append(res_)
        f_.append(res_[-2])
    # Return the highest log-likelihood results
    x_ = x_[np.argmax(f_)]
    return np.concatenate((x_[0], x_[1], x_[2], np.squeeze(x_[3], axis = 1)), axis = 0), x_[5], x_[6], x_[7]

# Gamma Distribution
def _2D_G(X_, alpha, beta, kappa):
    f_ = ( (beta**alpha) * (X_[:, 0]**(alpha + kappa - 1.)) * (X_[:, 1]**(kappa - 1.))) / ( gamma(alpha) * gamma(kappa) )
    f_ *= np.exp( - beta * X_[:, 0] ) * np.exp( - X_[:, 0] * X_[:, 1] )
    return f_

# Gamma Mixture Model
def _2D_GaMM(theta_, X_, n_dist_theta = 4):
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
    return Z_, k_

# 1D-Gamma Mixture Model
def _EM_GaMM_TC(X_, n_clusters = 2, n_init = 3, tol = 0.1, n_iter = 1000,
                reg_term = 1e-5, w_init_ = None, verbose = True):
    init_lim_low = 1.
    init_lim_up  = 10.
    # Check for EM algoritm convergence
    def __convergence(ll_k_1_, ll_k, tol):
        if abs(ll_k - ll_k_1_) < tol: return True
        else: return False
    # Log-Gamma Distribution
    def __log_G(X_, alpha, theta):
        log_f_ = (alpha - 1)*np.log(X_) - (X_/theta) - alpha*np.log(theta) - np.log(gamma(alpha))
        return log_f_
    # Gradient of log-gamma Distribution
    def __grad_log_G(X_, alpha, theta):
        g_alpha_ = np.log(X_) - np.log(theta) - digamma(alpha)
        g_theta_ = (1./theta) * ( (X_/theta) - alpha)
        return np.concatenate((g_alpha_, g_theta_), axis = 1)
    # Gradient Based marginal likelihood optimization
    def __optimization(X_, m_, k_, W_, bounds_):
        # Log-Likelihood Evaluation
        def ___f(x_, X_, W_):
            # Unpack Variables
            m_ = x_[:n_clusters]
            k_ = x_[n_clusters:]
            # Variable Initialization
            log_l_ = 0
            # Loop over clusters Parameters
            for k in range(n_clusters):
                # Calculate Mixture Model Marginal Log-likelihood
                log_f_ = __log_G(X_, m_[k], k_[k])
                log_f_ = W_[:, k] * log_f_[:, 0]
                log_l_+= log_f_.sum()
            return - log_l_
        # Log-Likelihood Gradient Evaluation
        def ___g(x_, X_, W_):
            # Unpack Variables
            m_ = x_[:n_clusters]
            k_ = x_[n_clusters:]
            # Variable Initialization
            log_l_ = np.zeros((n_clusters*2, 1))
            # Loop over clusters Parameters
            for k in range(n_clusters):
                # Calculate gradient of Mixture Model Marginal Log-likelihood
                log_g_ = __grad_log_G(X_, m_[k], k_[k])
                log_g_ = W_[:, k][:, np.newaxis] * log_g_
                log_g_ = np.sum(log_g_, axis = 0)
                log_l_[k] = log_g_[0]
                log_l_[k + n_clusters] = log_g_[1]
            return - log_l_
        def ___ineq_c(x_, dist = 10):
            # Unpack Variables
            k_ = x_[:n_clusters]
            theta_ = x_[n_clusters:]
            # Variable Initialization
            E_ = np.zeros((n_clusters, 1))
            # Loop over clusters Parameters
            for k in range(n_clusters):
                E_[k] = k_[k]*theta_[k]
            return np.absolute(E_[0] - E_[1]) - dist

        if n_clusters == 2:
            constr_ = {'type': 'ineq', 'fun': ___ineq_c}
            # Gradient Based Optimization
            opt_ = optimize.minimize(fun = ___f, x0 = np.concatenate((m_, k_)), jac = ___g,
                                     constraints = constr_, bounds = bounds_, args = (X_, W_), method = 'SLSQP')
        else:
           # Gradient Based Optimization
            opt_ = optimize.minimize(fun = ___f, x0 = np.concatenate((m_, k_)),
                                     bounds = bounds_, args = (X_, W_), method = 'SLSQP')
        return opt_['x'][:n_clusters], opt_['x'][n_clusters:]
    # Maximization Step
    def __maximization(X_, m_, k_, W_, bounds_, n_clusters):
        # Calculate the Responsabilies
        def ___resposabilities(W_):
            w_ = W_.sum(axis = 0)[:, np.newaxis] + np.finfo(W_.dtype).eps
            return w_ / W_.shape[0]
        # Probabilities and mixture model weights
        w_ = ___resposabilities(W_)
        # Optimize Distribution Parameters by Gradient
        m_, k_ = __optimization(X_, m_, k_, W_, bounds_)
        return m_, k_, w_
    # Random Initialization of VM mixture model Paramters
    def __rand_init(X_):
        # Constants Definition
        N, dim  = X_.shape
        ll_k    = - np.inf
        # Variable Initialization
        bounds_ = []
        m_init_ = np.empty((n_clusters, dim))
        k_init_ = np.empty((n_clusters, dim))
        # Randon Initialization VM Distributions weights and Noise Distribution
        w_init_ = np.random.uniform(0, 1, n_clusters + 1)[:, np.newaxis]
        w_init_/= w_init_.sum()
        for k in range(n_clusters):
            # Define Variable Boundaries for Optimization
            bounds_.insert(0, (.1, 500.))
            bounds_.append((.1, 500.))
            # Random Initialization of the Gamma Parameters
            m_init_[k] = np.random.uniform(init_lim_low, init_lim_up, 1)[0]
            k_init_[k] = np.random.uniform(init_lim_low, init_lim_up, 1)[0]
        return m_init_.copy(), k_init_.copy(), w_init_.copy(), bounds_, ll_k, dim

    def __init(X_, w_init_):
        # Constants Definition
        N, dim = X_.shape
        ll_k   = - np.inf
        # Variable Initialization
        bounds_ = []
        # Randon Initialization VM Distributions weights
        m_init_ = np.array(w_init_[:n_clusters])[:, np.newaxis]
        k_init_ = np.array(w_init_[n_clusters:2*n_clusters])[:, np.newaxis]
        w_init_ = np.array(w_init_[2*n_clusters:])[:, np.newaxis]
        w_init_ = w_init_ + np.random.uniform(0, 1)
        w_init_/= w_init_.sum()
        # Von Mises Distribution Parameters Initialization
        for k in range(n_clusters):
            # Define Variable Boundaries for Optimization
            bounds_.insert(0, (.1, 500.))
            bounds_.append((.1, 500.))
        return m_init_.copy(), k_init_.copy(), w_init_.copy(), bounds_, ll_k, dim
    # Calulate mixture Model Log-likelihood
    def __mixture_model_log_likelihood(m_, k_, X_, w_):
        log_P_ = np.empty((X_.shape[0], 0))
        log_w_ = np.log(w_)
        # weighted log-likelihood for each distribution
        for k in range(n_clusters):
            log_p_ = log_w_[k] + __log_G(X_, m_[k], k_[k])
            log_P_ = np.concatenate((log_P_, log_p_), axis = 1)
        # Log-sum-exp of all distributions log-likelihoods
        log_lik_ = logsumexp(log_P_, axis = 1)[:, np.newaxis]
        return log_P_, log_lik_
    # Expectation Step
    def __expectation(X_, m_, k_, w_, n_clusters):
        log_P_, log_lik_ = __mixture_model_log_likelihood(m_, k_, X_, w_)
        return np.exp(log_P_ - log_lik_), np.sum(log_lik_), log_P_
    # Run Initialization of Expectation-Maximization algorithm
    def __EM(X_, w_init_):
        if w_init_ is None:
            m_, k_, w_, bounds_, ll_k, dim  = __rand_init(X_)
        else:
            m_, k_, w_, bounds_, ll_k, dim  = __init(X_, w_init_)
        # Stat iterative Expectation-Maximization Algorithm
        for i in range(n_iter):
            # Expectation Step
            W_, ll_k_1, l_ = __expectation(X_, m_, k_, w_, n_clusters)
            if verbose:
                print('>>> No iter.: {} log-lik.: {}'.format(i, ll_k_1))
            # Check for EM convergence
            if __convergence(ll_k_1, ll_k, tol):
                break
            else:
                # Maximization Step
                m_, k_, w_ = __maximization(X_, m_, k_, W_, bounds_, n_clusters)
                ll_k = ll_k_1
        # Sample Cluster Label
        z_ = np.argmax(W_, axis = 1)
        # Get Scores
        scores_ = _error_metrics(W_, ll_k, theta_ = [m_, k_, w_])
        # if ll_k_1 == 0.:
        #     ll_k_1 == -np.inf
        return [m_, k_, w_, z_, ll_k, scores_]
    # Storage Variable Initialization
    x_, f_ = [], []
    # Loop over Initialization
    for i in range(n_init):
        if verbose: print('>>> No. Init.: {}'.format(i))
        # Run Expectation-Maximization
        res_ = __EM(X_, w_init_)
        x_.append(res_)
        f_.append(res_[-2])
    # Return the highest log-likelihood results
    x_ = x_[np.argmax(f_)]
    return np.concatenate((x_[0], x_[1], np.squeeze(x_[2], axis = 1)), axis = 0), x_[4], x_[5]

# Beta Mixture Model Inferece
def _EM_BeMM_TC(X_, dt = 0, t_max = 0, n_clusters = 2, n_init = 3, tol = 0.1,
                reg_term = 1e-5, n_iter = 1000, verbose = False):
    # Check for EM algoritm convergence
    def __convergence(ll_k_1_, ll_k, tol):
        if abs(ll_k - ll_k_1_) < tol: return True
        else: return False
    # Log-beta Distribution
    def __log_beta(X_, a, b):
        log_f_ = - np.log(beta(a, b)) - np.log(X_) + a*np.log(X_) - np.log(1. - X_) + b*np.log(1. - X_)
        return log_f_
    # Gradient of log-beta Distribution
    def __grad_log_beta(X_, a, b):
        g_a_ = np.log(X_) - ( beta(a, b) * (digamma(a) - digamma( a + b )) )/beta(a, b)
        g_b_ = np.log(1. - X_) - ( beta(a, b) * (digamma(b) - digamma( b + a )) )/beta(a, b)
        return np.concatenate((g_a_, g_b_), axis = 1)
    # Gradient Based marginal likelihood optimization
    def __optimization(X_, a_, b_, W_, bounds_):
        # Log-Likelihood Evaluation
        def ___f(x_, X_, W_):
            # Unpack Variables
            a_ = x_[:n_clusters]
            b_ = x_[n_clusters:]
            # Variable Initialization
            log_l_ = 0
            # Loop over clusters Parameters
            for k in range(n_clusters):
                # Calculate Mixture Model Marginal Log-likelihood
                log_f_ = __log_beta(X_, a_[k], b_[k])
                log_f_ = W_[:, k] * log_f_[:, 0]
                log_l_+= log_f_.sum()
            return - log_l_
        # Log-Likelihood Gradient Evaluation
        def ___g(x_, X_, W_):
            # Unpack Variables
            a_ = x_[:n_clusters]
            b_ = x_[n_clusters:]
            # Variable Initialization
            log_l_ = np.zeros((n_clusters*2, 1))
            # Loop over clusters Parameters
            for k in range(n_clusters):
                # Calculate gradient of Mixture Model Marginal Log-likelihood
                log_g_ = __grad_log_beta(X_, a_[k], b_[k])
                log_g_ = W_[:, k][:, np.newaxis] * log_g_
                log_g_ = np.sum(log_g_, axis = 0)
                log_l_[k] = log_g_[0]
                log_l_[k + n_clusters] = log_g_[1]
            return - log_l_
        # Temperature Constain
        def ___ineq_c(x_):
            # Unpack Variables
            a_ = x_[:n_clusters]
            b_ = x_[n_clusters:]
            # Variable Initialization
            E_ = np.zeros((n_clusters, 1))
            # Loop over clusters Parameters
            for k in range(n_clusters):
                E_[k] = t_max*a_[k]/(a_[k] + a_[k])
            return np.absolute(E_[0] - E_[1]) - dt
        # Use contrain only when there are 2 clusters
        if n_clusters == 2:
            # Constrain optimization
            constr_ = {'type': 'ineq', 'fun': ___ineq_c}
            opt_ = optimize.minimize(fun = ___f, jac = ___g, x0 = np.concatenate((a_, b_)),
                                     constraints = constr_,bounds = bounds_, args = (X_, W_), method = 'SLSQP')
        else:
            # Gradient Based Optimization
            opt_ = optimize.minimize(fun = ___f, jac = ___g, x0 = np.concatenate((a_, b_)),
                                     bounds = bounds_, args = (X_, W_), method = 'SLSQP')
        return opt_['x'][:n_clusters], opt_['x'][n_clusters:]
    # Maximization Step
    def __maximization(X_, a_, b_, W_, bounds_, n_clusters):
        # Calculate the Responsabilies
        def ___resposabilities(W_):
            w_ = W_.sum(axis = 0)[:, np.newaxis] + np.finfo(W_.dtype).eps
            return w_ / W_.shape[0]
        # Probabilities and mixture model weights
        w_ = ___resposabilities(W_)
        # Optimize Distribution Parameters by Gradient
        a_, b_ = __optimization(X_, a_, b_, W_, bounds_)
        return a_, b_, w_
    # Random Initialization of VM mixture model Paramters
    def __init(X_):
        # Constants Definition
        N, dim  = X_.shape
        ll_k    = - np.inf
        # Variable Initialization
        bounds_ = []
        a_init_ = np.empty((n_clusters, dim))
        b_init_ = np.empty((n_clusters, dim))
        # Randon Initialization VM Distributions weights and Noise Distribution
        w_init_ = np.random.uniform(0, 1, n_clusters)[:, np.newaxis]
        w_init_/= w_init_.sum()
        # Beta Distribution Parameters Initialization
        for k in range(n_clusters):
            # Define Variable Boundaries for Optimization
            bounds_.insert(0, (.1, 50.))
            bounds_.append((.1, 50.))
            # Random Initialization of the Beta Parameters
            a_init_[k] = np.random.uniform(.1, .9, 1)[0]
            b_init_[k] = np.random.uniform(.1, .9, 1)[0]
        return a_init_.copy(), b_init_.copy(), w_init_.copy(), bounds_, a_init_, b_init_, w_init_, ll_k, dim
    # Calulate mixture Model Log-likelihood
    def __mixture_model_log_likelihood(a_, b_, X_, w_):
        log_P_ = np.empty((X_.shape[0], 0))
        log_w_ = np.log(w_)
        X_ -= 1e-10
        # weighted log-likelihood for each distribution
        for k in range(n_clusters):
            log_p_ = log_w_[k] + __log_beta(X_, a_[k], b_[k])
            log_P_ = np.concatenate((log_P_, log_p_), axis = 1)
        # Log-sum-exp of all distributions log-likelihoods
        log_lik_ = logsumexp(log_P_, axis = 1)[:, np.newaxis]
        return log_P_, log_lik_
    # Expectation Step
    def __expectation(X_, a_, b_, w_, n_clusters):
        log_P_, log_lik_ = __mixture_model_log_likelihood(a_, b_, X_, w_)
        return np.exp(log_P_ - log_lik_), np.sum(log_lik_), log_P_

    # Run Initialization of Expectation-Maximization algorithm
    def __EM(X_):
        a_, b_, w_, bounds_, a_init_, b_init_, w_init_, ll_k, dim = __init(X_)
        # Stat iterative Expectation-Maximization Algorithm
        for i in range(n_iter):
            # Expectation Step
            W_, ll_k_1_, l_ = __expectation(X_, a_, b_, w_, n_clusters)
            # Maximization Step
            a_, b_, w_ = __maximization(X_, a_, b_, W_, bounds_, n_clusters)
            if verbose: print('>>> No iter.: {} log-lik.: {}'.format(i, ll_k_1_))
            # Check for EM convergence
            if __convergence(ll_k_1_, ll_k, tol): break
            else: ll_k = ll_k_1_
        # Sample Cluster Label
        z_ = np.argmax(W_, axis = 1)
        # Get Scores
        scores_ = _error_metrics(W_, l_, z_, theta_ = [a_, b_, w_])
        #print(scores_)
        return [a_, b_, w_, z_, ll_k_1_, scores_]
    # Storage Variable Initialization
    x_, f_ = [], []
    X_[X_ == 1.] -= reg_term
    X_[X_ == 0.] += reg_term
    # Loop over Initialization
    for i in range(n_init):
        if verbose: print('>>> No. Init.: {}'.format(i))
        # Run Expectation-Maximization
        res_ = __EM(X_)
        x_.append(res_)
        f_.append(res_[-2])
    # Return the highest log-likelihood results
    x_ = x_[np.argmax(f_)]
    return np.concatenate((x_[0], x_[1], np.squeeze(x_[2], axis = 1)), axis = 0), x_[4], x_[5]

__all__ = ['_EM_GMM', '_EM_VmMM', '_VMM', '_VM','_EM_NVmMM', '_NVmMM', '_EM_MGMM',  '_EM_GaMM',
           '_GaMM', '_EM_2D_GaMM', '_2D_GaMM', '_2D_G', '_1D_G', '_B', '_BeMM', '_EM_BeMM',
           '_EM_2DT_GaMM', '_EM_GaMM_TC', '_EM_BeMM_TC', '_multivariate_gaussian_mixture_model_log_likelihood',
           '_gaussian_mixture_model_log_likelihood', '_beta_mixture_model_log_likelihood', '_von_mises_mixture_model_log_likelihood',
           '_1Dgamma_mixture_model_log_likelihood', '_2Dgamma_mixture_model_log_likelihood', '_error_metrics']
