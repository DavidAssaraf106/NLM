<<<<<<< HEAD
﻿import autograd.numpy as npfrom sklearn.linear_model import LogisticRegressionfrom autograd import graddef hmc(log_prior, log_likelihood, num_samples, step_size, L, init, burn, thin):    """    :param log_prior: The log of the prior on our parameters    :param log_likelihood: The log of the likelihood of our data under our model    :param num_samples: The number of samples produced    :param step_size: The step-size in the Leap Frog estimator    :param L: The number of steps in the Leap Frog Estimator    :param init: The initial position of the HMC    :param burn: Burn-in parameter    :param thin: Thinning parameter    :return: Samples of our posterior distribution using HMC. Shape: (D, (num_samples-burn)/thin)    Note: We imposed here the choice of mass m = 1 and a quadratic Kinetic Energy providing a Normal Gibbs Sampler    """    def U(W):        return -1 * (log_likelihood(W).flatten() + log_prior(W).flatten())    def K(W):        return np.sum(W ** 2) / 2    def K_gibbs_sampler():        D = init.flatten().shape[0]  # Dimensionality of our problem        return np.random.normal(0, 1, size=(1, D))    q_current = init    samples = [q_current]    accept = 0    # gradient U wrt to q    grad_U = grad(U)    # gradient of K wrt to p    grad_K = grad(K)    for i in range(num_samples):        if i % 100 == 0 and i > 0:            print(i, ':', accept * 1. / i)        p_current = K_gibbs_sampler()  # sample a random momentum from the Gibbs distribution        # calc position and momentum after L steps (initaliate intermediate vars)        q_proposal = q_current.copy()        p_proposal = p_current.copy()        # Leap-frog        for j in range(L):            # half-step update for momentum            p_step_t_half = p_proposal.flatten() - (step_size / 2.) * grad_U(q_proposal.flatten())            # full step update for position            q_proposal += step_size * p_step_t_half            # half-step update for momentum            p_proposal = p_step_t_half - (step_size / 2.) * grad_U(q_proposal)        p_proposal = - p_proposal.copy()  # reverse momentum to ensure detail balance/reversibility        # accept/reject new proposed position        H_proposal = U(q_proposal) + K(p_proposal)        H_current = U(q_current) + K(p_current)        proposal = np.exp(H_current - H_proposal)        alpha = min(1, proposal)        if np.random.uniform() <= alpha:            accept += 1  # you should keep track of your acceptances            q_current = q_proposal.copy()        samples.append(q_current.flatten())        i += 1    # burn and thin    burn_n = round(burn * num_samples)    return samples[burn_n::thin]def sigmoid(z):    return 1. / (1. + np.exp(-z))def HMC_Unit_test():    # Generate a toy dataset for classification    samples = 100    class_0 = np.random.multivariate_normal([-1, -1], 0.5 * np.eye(2), samples)    class_1 = np.random.multivariate_normal([1, 1], 0.5 * np.eye(2), samples)    x = np.vstack((class_0, class_1))    y = np.array([0] * 100 + [1] * 100)    mean = np.zeros(3)    cov = 10*np.eye(3)    D = 3    def log_likelihood(w):        theta = sigmoid(w[0] + np.dot(x, w[1:]))        return np.sum(np.log(theta[y==1])) + np.sum(np.log(1 - theta[y==0]))    def log_normal_prior(W):        logprior = -0.5 * (np.log(np.linalg.det(cov)) + D * np.log(2 * np.pi))        logprior += -0.5 * np.dot(np.dot(W-mean, np.linalg.inv(cov)), (W-mean).T)        return logprior    log_prior = log_normal_prior    log_likelihood = log_likelihood    lr = LogisticRegression(C=1., penalty='l2', solver='saga', tol=0.1)    lr.fit(x, y)    position_init = np.hstack((lr.coef_.flatten(), lr.intercept_))    position_init = position_init.reshape((1, 3))[0]    samples = hmc(log_prior, log_likelihood, 1000,  1e-1, 20, position_init, 0.1, 1)if __name__ == '__main__':    HMC_Unit_test()
=======
import autograd.numpy as np
from sklearn.linear_model import LogisticRegression
from autograd import grad


def hmc(log_prior, log_likelihood, num_samples, step_size, L, init, burn, thin):
    """
    :param log_prior: The log of the prior on our parameters
    :param log_likelihood: The log of the likelihood of our data under our model
    :param num_samples: The number of samples produced
    :param step_size: The step-size in the Leap Frog estimator
    :param L: The number of steps in the Leap Frog Estimator
    :param init: The initial position of the HMC
    :param burn: Burn-in parameter
    :param thin: Thinning parameter
    :return: Samples of our posterior distribution using HMC. Shape: (D, (num_samples-burn)/thin)
    Note: We imposed here the choice of mass m = 1 and a quadratic Kinetic Energy providing a Normal Gibbs Sampler
    """

    def U(W):
        return -1 * (log_likelihood(W).flatten() + log_prior(W).flatten())

    def K(W):
        return np.sum(W ** 2) / 2

    def K_gibbs_sampler():
        D = init.flatten().shape[0]  # Dimensionality of our problem
        return np.random.normal(0, 1, size=(1, D))

    q_current = init
    samples = [q_current]
    accept = 0

    # gradient U wrt to q
    grad_U = grad(U)
    # gradient of K wrt to p
    grad_K = grad(K)

    for i in range(num_samples):

        if i % 100 == 0 and i > 0:
            print(i, ':', accept * 1. / i)

        p_current = K_gibbs_sampler()  # sample a random momentum from the Gibbs distribution

        # calc position and momentum after L steps (initaliate intermediate vars)
        q_proposal = q_current.copy()
        p_proposal = p_current.copy()

        # Leap-frog
        for j in range(L):
            # half-step update for momentum
            p_step_t_half = p_proposal.flatten() - (step_size / 2.) * grad_U(q_proposal.flatten())
            # full step update for position
            q_proposal += step_size * p_step_t_half
            # half-step update for momentum
            p_proposal = p_step_t_half - (step_size / 2.) * grad_U(q_proposal)

        p_proposal = - p_proposal.copy()  # reverse momentum to ensure detail balance/reversibility

        # accept/reject new proposed position
        H_proposal = U(q_proposal) + K(p_proposal)
        H_current = U(q_current) + K(p_current)
        proposal = np.exp(H_current - H_proposal)

        alpha = min(1, proposal)

        if np.random.uniform() <= alpha:
            accept += 1  # you should keep track of your acceptances
            q_current = q_proposal.copy()

        samples.append(q_current.flatten())
        i += 1

    # burn and thin
    burn_n = round(burn * num_samples)
    return samples[burn_n::thin]




def sigmoid(z):
    return 1. / (1. + np.exp(-z))


def HMC_Unit_test():
    # Generate a toy dataset for classification
    samples = 100
    class_0 = np.random.multivariate_normal([-1, -1], 0.5 * np.eye(2), samples)
    class_1 = np.random.multivariate_normal([1, 1], 0.5 * np.eye(2), samples)
    x = np.vstack((class_0, class_1))
    y = np.array([0] * 100 + [1] * 100)
    mean = np.zeros(3)
    cov = 10*np.eye(3)
    D = 3

    def log_likelihood(w):
        theta = sigmoid(w[0] + np.dot(x, w[1:]))
        theta = np.clip(theta, 1e-15, 1-1e-15)
        loglkhd = y * np.log(theta) + (1 - y) * np.log(1 - theta)
        return np.sum(loglkhd)

    def log_normal_prior(W):
        logprior = -0.5 * (np.log(np.linalg.det(cov)) + D * np.log(2 * np.pi))
        logprior += -0.5 * np.dot(np.dot(W-mean, np.linalg.inv(cov)), (W-mean).T)
        return logprior

    log_prior = log_normal_prior
    log_likelihood = log_likelihood
    lr = LogisticRegression(C=1., penalty='l2', solver='saga', tol=0.1)
    lr.fit(x, y)
    position_init = np.hstack((lr.coef_.flatten(), lr.intercept_))
    position_init = position_init.reshape((1, 3))[0]
    samples = hmc(log_prior, log_likelihood, 1000,  1e-1, 20, position_init, 0.1, 1)


if __name__ == '__main__':
    HMC_Unit_test()
>>>>>>> 41711a589329220fc51b470a9449482e1ac38577
