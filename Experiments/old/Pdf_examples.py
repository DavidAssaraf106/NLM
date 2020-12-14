from Bayesian_pdf import log_normal_prior, get_log_prior, get_log_likelihood, log_logistic_likelihood
from scipy.stats import multivariate_normal
from NLM_Example import fit_MLE_1
import numpy as np


def normal_prior():
    function = get_log_prior('normal', {'mean': np.array([1, 2]), 'covariance_matrix': np.eye(2)}, 2)
    likelihood = (function(np.array([1, 2]).reshape(1, -1)))
    true_var = multivariate_normal(mean=[1, 2], cov=[[1, 0], [0, 1]])
    log_true_likelihood = np.log(true_var.pdf([1, 2]))
    return np.round(log_true_likelihood, 3) == np.round(likelihood, 3)

def logistic_likelihood():
    nlm, _, _, X, y = fit_MLE_1(2000, 2, exigence=False)
    function = get_log_likelihood('logistic', {}, nlm, X, y, nlm.D)
    log_likelihood = function(np.ones(nlm.params['H']))
    print('Final log_likelihood', log_likelihood)


if __name__ == '__main__':
    logistic_likelihood()
