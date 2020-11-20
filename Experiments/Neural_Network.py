from FeedForwardNN import Feedforward
from Toy_Datasets import two_clusters_gaussian
import autograd.numpy as np
from autograd import grad
from autograd.misc.optimizers import adam
from Bayesian_pdf import get_log_prior, get_log_likelihood
from Hamiltonian_MC import hmc


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(y):   
            return np.exp(y)/(np.exp(y).sum())
class NLM:
    """
    This class implements the framework of training of the Neural Linear Model, as introduced in ...
    """

    def __init__(self, architecture, random=None, weights=None):
        """
        :param architecture: architecture is a dictionary which should contain the following keys:
        - width: the number of nodes inside every hidden layer (constant across the various hidden layers)
        - hidden_layers: the number of hidden layers
        - input_dim: the number of features of every training point
        - output_dim: the dimensionality of the output vector (=number of classes for a classification task)
        - activation_fn_type, activation_fn_params: related to the activation functions (not the output function)
        - prior: the type of prior distribution over the NN parameters. Currently supported : {beta, normal,
        None (no prior: MLE fit)}
        - prior_parameters: the parameters of the prior distribution. Should be a dictionary
        - likelihood: the type of likelihood distribution for the likelihood of the model. Currently supported:
        {Gaussian posterior, Logistic posterior, Categorical, None (for MLE fit)}
        - likelihood_parameters: the parameters of the likelihood distribution. Should be a dictionary
        """

        # check during the construction that the shapes are coherent, especially the shapes of the feature map and the
        # prior on the weights
        self.params = {'H': architecture['width'],
                       'L': architecture['hidden_layers'],
                       'D_in': architecture['input_dim'],
                       'D_out': architecture['output_dim'],
                       'activation_type': architecture['activation_fn_type'],
                       'activation_params': architecture['activation_fn_params'],
                       'prior_distribution': architecture.get('prior', None),
                       'prior_parameters': architecture.get('prior_parameters', None),
                       'likelihood_distribution': architecture.get('likelihood', None),
                       'likelihood_parameters': architecture.get('likelihood_parameters', None)}

        self.D = ((architecture['input_dim'] * architecture['width'] + architecture['width'])
                  + (architecture['output_dim'] * architecture['width'] + architecture['output_dim'])
                  + (architecture['hidden_layers'] - 1) * (architecture['width'] ** 2 + architecture['width'])
                  )  # in order: input, output, hidden. Take into account the biases

        if random is not None:
            self.random = random
        else:
            self.random = np.random.RandomState(0)

        #self.h = architecture['activation_fn'] ##where is it?
        self.h = lambda x: np.maximum(np.zeros(x.shape), x)

        if weights is None:
            self.weights = self.random.normal(0, 1, size=(1, self.D))
        else:
            self.weights = weights

        self.objective_trace = np.empty((1, 1))
        self.weight_trace = np.empty((1, self.D))

    def forward(self, weights, x, partial=False):
        ''' Forward pass given weights and input '''
        H = self.params['H']
        D_in = self.params['D_in']
        D_out = self.params['D_out']
        assert weights.shape[1] == self.D

        if len(x.shape) == 2:
            assert x.shape[0] == D_in
            x = x.reshape((1, D_in, -1))
        else:
            assert x.shape[1] == D_in

        weights = weights.T

        # input to first hidden layer
        W = weights[:H * D_in].T.reshape((-1, H, D_in))
        b = weights[H * D_in:H * D_in + H].T.reshape((-1, H, 1))
        input = self.h(np.matmul(W, x) + b)
        index = H * D_in + H

        assert input.shape[1] == H

        # additional hidden layers, except the last one
        for _ in range(self.params['L'] - 1):
            before = index
            W = weights[index:index + H * H].T.reshape((-1, H, H))
            index += H * H
            b = weights[index:index + H].T.reshape((-1, H, 1))
            index += H
            output = np.matmul(W, input) + b
            input = self.h(output)

            assert input.shape[1] == H

        if partial:  # post-training, we need the NLM to make partial forward passes.
            return input

        # output layer
        W = weights[index:index + H * D_out].T.reshape((-1, D_out, H))
        b = weights[index + H * D_out:].T.reshape((-1, D_out, 1))
        #output = softmax(np.matmul(W, input) + b)  # review that for training
        output = np.array([[softmax(np.matmul(W, input)[0][i] + b[0][i]) for i in range(D_out)]])
        assert output.shape[1] == self.params['D_out']

        return output


    def make_objective(self, x_train, y_train, reg_param):

        def objective(W, t):
            sigmoid_probability = self.forward(W, x_train)
            ## foutre une proba 
            sigmoid_p=sigmoid(sigmoid_probability)
            sigmoid_p = np.clip(sigmoid_probability, 1e-15, 1 - 1e-15)            
            bce = np.dot(np.log(sigmoid_p),y_train.flatten()).mean() #Cross-Entropy Loss 
            if reg_param is None:
                sum_error = bce
                return -sum_error
            else:
                mean_error = bce + reg_param * np.linalg.norm(W)
                return -mean_error

        return objective, grad(objective)

    def fit_MLE(self, x_train, y_train, params, reg_param=None):

        assert x_train.shape[0] == self.params['D_in']
        assert y_train.shape[0] == self.params['D_out']

        ### make objective function for training
        self.objective, self.gradient = self.make_objective(x_train, y_train, reg_param)

        ### set up optimization
        step_size = 0.01
        max_iteration = 5000
        check_point = 100
        weights_init = self.weights.reshape((1, -1))
        mass = None
        optimizer = 'adam'
        random_restarts = 5

        if 'step_size' in params.keys():
            step_size = params['step_size']
        if 'max_iteration' in params.keys():
            max_iteration = params['max_iteration']
        if 'check_point' in params.keys():
            self.check_point = params['check_point']
        if 'init' in params.keys():
            weights_init = params['init']
        if 'call_back' in params.keys():
            call_back = params['call_back']
        if 'mass' in params.keys():
            mass = params['mass']
        if 'optimizer' in params.keys():
            optimizer = params['optimizer']
        if 'random_restarts' in params.keys():
            random_restarts = params['random_restarts']

        def call_back(weights, iteration, g):
            ''' Actions per optimization step '''
            objective = self.objective(weights, iteration)
            self.objective_trace = np.vstack((self.objective_trace, objective))
            self.weight_trace = np.vstack((self.weight_trace, weights))
            if iteration % check_point == 0:
                print("Iteration {} lower bound {}; gradient mag: {}".format(iteration, objective, np.linalg.norm(
                    self.gradient(weights, iteration))))

        ### train with random restarts
        optimal_obj = 1e16
        optimal_weights = self.weights

        for i in range(random_restarts):
            if optimizer == 'adam':
                adam(self.gradient, weights_init, step_size=step_size, num_iters=max_iteration, callback=call_back)
            local_opt = np.min(self.objective_trace[-100:])
            if local_opt < optimal_obj:
                opt_index = np.argmin(self.objective_trace[-100:])
                self.weights = self.weight_trace[-100:][opt_index].reshape((1, -1))
            weights_init = self.random.normal(0, 1, size=(1, self.D))




        self.objective_trace = self.objective_trace[1:]
        self.weight_trace = self.weight_trace[1:]



    def fit_NLM(self, x_train, y_train, hmc, params_hmc):
        """
        :param self: a Neural Network that has been fitted via MLE. Also, the params of the NN should contain a key
        'prior' and a key 'likelihood'
        :param x_train: training features
        :param y_train: training labels
        :param hmc: HMC sampler.
        :param params_hmc: hyperparameters for HMC. Should be a dictionary with the following keys:
        - num_samples: total number of samples produced by the posterior
        - step_size:  The step-size in the Leap Frog estimator
        - L: The number of steps in the Leap Frog Estimator
        - init: The initial position of the HMC
        - burn: Burn-in parameter
<<<<<<< HEAD
=======
        - thin: Thinning factor
>>>>>>> main
        :return: Samples from the posterior distribution.
        """
        D = self.params['H']  # dimensionality of the feature map
        log_prior = get_log_prior(self.params['prior_distribution'], self.params['prior_parameters'], D)
        log_likelihood = get_log_likelihood(self.params['likelihood_distribution'], self.params['likelihood_parameters'], self, x_train, y_train, D) #we are supposed to take the output of the last layer
        samples = hmc(log_prior, log_likelihood, **params_hmc)
        return samples


    def get_feature_map_weights(self):
        """This function returns the weight of the last hidden layer. Those are the weights we will put
        our prior on in the NLM.
        """
        return self.weights.flatten()[-self.params['D_out']-self.params['H']:-self.params['D_out']].reshape(1, -1)


    def sample(self, x_train, y_train, hmc, params_fit, params_hmc):
        self.fit_MLE(x_train, y_train, params_fit)
        params_hmc['init'] = self.get_feature_map_weights()
        samples = self.fit_NLM(x_train, y_train, hmc, params_hmc)
        return samples


class Classifier:
    """
    This class implements the scikit-learn API for our Neural Network.
    """
    def __init__(self, weights, forward):
        self.weights = weights
        self.forward = forward

    def predict(self, x):
        p = self.forward(self.weights, x.T)
        return (p > 0.5).astype(np.int_)

    def predict_proba(self, x):
        return self.forward(self.weights, x.T)