#  Here, you will find all of the priors and likelihoods we have dealt with during the AM207 course, and more
#  The standard input format will be : (W, params, X, y) where we will impose some shape on X and y for likelihoods and
#  (W, params) for the priors
import autograd.numpy as np


"""
The structure we impose on the weights for the prior are shape (1, D)
"""


def log_normal_prior(params):  # checked
    try:
        mean = params['mean']
        cov = params['covariance_matrix']
    except KeyError:
        raise KeyError('Missing argument for the parameters of the prior distribution')
    assert np.array(mean).shape[0] == cov.shape[0], 'The shapes of the mean and cov of the prior are not right'

    def log_normal_prior(W):
        assert (W.shape[1] == np.array(mean).shape[0])
        D = W.shape[1]
        logprior = -0.5 * (np.log(np.linalg.det(cov)) + D * np.log(2 * np.pi))
        logprior += -0.5 * np.dot(np.dot(W-mean.reshape(1, -1), np.linalg.inv(cov)), (W-mean.reshape(1, -1)).T)
        return logprior

    return log_normal_prior


# adapt it to the softmax version for multi-class settings. Here, it is only the Logistic setting.

def log_logistic_likelihood(params, nlm, X, y, D):
    def sigmoid(z):
        return 1. / (1. + np.exp(-z))

    def log_logistic(W):
        mapped_X = nlm.forward(nlm.weights, X, partial=True)  # feature map of the inputs, dimension D
        dot_product = np.dot(W, mapped_X)
        theta = sigmoid(dot_product)
        theta = np.clip(theta, 1e-15, 1 - 1e-15)
        loglkhd = y * np.log(theta) + (1 - y) * np.log(1 - theta)
        return np.sum(loglkhd)

    return log_logistic


def get_log_prior(type, params, D):
    mapping = {'normal': log_normal_prior}
    return mapping[type](params)


def get_log_likelihood(type, params, nlm, X, y, D):
    mapping = {'logistic': log_logistic_likelihood}
    return mapping[type](params, nlm, X, y, D)




