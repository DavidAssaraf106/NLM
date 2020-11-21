import autograd.numpy as np
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