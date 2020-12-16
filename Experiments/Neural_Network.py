import autograd.numpy as np
from pymc3 import Model
import pymc3 as pm
import theano
from autograd import grad
from autograd.misc.optimizers import adam
import theano.tensor as T
import random
from old.Bayesian_pdf import get_log_prior, get_log_likelihood
from old.Hamiltonian_MC import hmc

mac = False


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def softmax(y):  # checked, ok for softmax and the dimensions
    """
    This function is used to perform multi-class classification. This should be the activation function.
    We need to handle the cases where the shape of the input are going to be (1, K, batch_size)
    """
    D = y.shape[1]
    z = y.flatten().reshape(D, -1)
    z = np.exp(z)
    return z / np.sum(z, axis=0)


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

        self.h = architecture['activation_fn']  # where is it?, it is in the parameters where we define the NN

        if weights is None:
            self.weights = self.random.normal(0, 1, size=(1, self.D))
        else:
            self.weights = weights

        self.objective_trace = np.empty((1, 1))
        self.weight_trace = np.empty((1, self.D))

    def forward(self, weights, x, partial=False):
        """ Forward pass given weights and input """
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
        output = np.matmul(W, input) + b
        output = softmax(output).T
        assert output.shape[1] == self.params['D_out']

        return output

    def make_objective(self, x_train, y_train, reg_param):
        # We are in the case of multi-task classification. The labels need to be one-hot encoded and the loss function we
        # will use is the Categorical Cross Entropy. This needs to be done in the Output Layer. Therefore, the output layer
        # produces a vector of dimension (K, 1) for every input, where K is the number of classes. Every element of this
        # output vector is a probability
        # reference to categorical cross-entropy : https://gombru.github.io/2018/05/23/cross_entropy_loss/
        def objective(W, t):

            softmax_probability = self.forward(W, x_train)  # is of size (x_train, K)
            softmax_p = np.clip(softmax_probability, 1e-15, 1 - 1e-15)
            # here, y_train is of size (batch_size, K)
            # softmax_p is of size (batch_size, K)
            # in the single label classification (every training point has only one label)
            Cat_cross_entropy = np.sum(y_train.T * np.log(softmax_p), axis=1)  # vector of size(len(x_train, 1))
            total_cat_ce = np.mean(Cat_cross_entropy)
            if reg_param is None:
                sum_error = total_cat_ce
                return -sum_error
            else:
                mean_error = total_cat_ce - reg_param * np.linalg.norm(W)
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

    def get_feature_map_weights(self):  # convention: weights1, bias 1, weights 2, bias 2, weights3 bias 3
        """This function returns the weight of the last hidden layer. Those are the weights we will use in order
        to initialize the MCMC sampler for our posterior. The structure we decided in the pymc3 sampling is that
        the weights should be ordered as [(weights class i, bias class i) for i <= k] where k = num_classes.
        The actual ordering we defined in the forward mode was [weights class i for i <=k] + [bias class i for i <= k]
        for every layer. Therefore, we need to reshape our weights here. Wrong, WW does the same, all weights and
        all biases.
        NOTE: works for achitecture with > 1 hidden layer
        """
        index_output_layer = - self.params['H'] * self.params['D_out'] - self.params['D_out']
        weights_concerned = self.weights.flatten()[index_output_layer:]
        return weights_concerned.flatten()


    def pymc3_sampling(self, out_last_hidden_layer, output_dim, y, D, mac, mu_wanted=0, tau_wanted=1,
                       samples_wanted=1500,
                       number_chains=2):
        """
        :param out_last_hidden_layer: the feature map after the trained Neural network
        :param output_dim: the output dimension (= number of classes)
        :param y: your training labels
        :param D: the number of hidden nodes (is also the dimensionnality of the output of the feature map)
        :param mu_wanted: mu of the normal prior
        :param tau_wanted: precision of the normal prior
        :param samples_wanted: number of samples generated
        :param number_chains: number of chains ran
        :return: samples from the posterior of the Bayesian Logistic regression
        """
        initialization_pymc3 = self.get_feature_map_weights()
        if mac:
            theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"
        with pm.Model() as replacing_HMC:
            w = pm.Normal('w', mu=initialization_pymc3, tau=tau_wanted, shape=(D * output_dim + output_dim))
            linear_combinations = []
            for j in range(output_dim):
                dot = pm.math.dot(out_last_hidden_layer[0].T, w[j * D:j * D + D]) + w[
                    -output_dim + j]  # agreed and checked
                linear_combi = pm.Deterministic('s' + str(j), dot)
                linear_combinations.append(linear_combi)
            thetas = pm.Deterministic('theta', T.nnet.softmax(linear_combinations))
            y_obs = pm.Categorical('y_obs', p=thetas.T, observed=y)
            if mac:
                trace = pm.sample(samples_wanted, chains=number_chains, cores=1, init='advi')
            else:
                trace = pm.sample(samples_wanted, chains=number_chains, init='advi', cores=1)
        return trace

    def pymc3_sampling_modif(self, out_last_hidden_layer, output_dim, y, D, mac, mu_wanted=0, tau_wanted=1,
                       samples_wanted=1500,
                       number_chains=2):
        """
        :param out_last_hidden_layer: the feature map after the trained Neural network
        :param output_dim: the output dimension (= number of classes)
        :param y: your training labels
        :param D: the number of hidden nodes (is also the dimensionnality of the output of the feature map)
        :param mu_wanted: mu of the normal prior
        :param tau_wanted: precision of the normal prior
        :param samples_wanted: number of samples generated
        :param number_chains: number of chains ran
        :return: samples from the posterior of the Bayesian Logistic regression
        """
        initialization_pymc3 = self.get_feature_map_weights()
        weights_init = initialization_pymc3[:15].reshape(5, 3)
        bias = initialization_pymc3[-3:]
        if mac:
            theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"
        with pm.Model() as replacing_HMC:
            w = pm.Normal('w', mu=weights_init, tau=tau_wanted, shape=(D, output_dim))
            intercept = pm.Normal('intercept', mu=bias, tau=1.0, shape=output_dim)
            dot = pm.math.dot(out_last_hidden_layer[0].T,w) + intercept  # agreed and checked
            thetas = pm.Deterministic('theta', T.nnet.softmax(dot))
            y_obs = pm.Categorical('y_obs', p=thetas, observed=y)
            #if mac:
             #   trace = pm.sample(samples_wanted, chains=number_chains, cores=1, init='advi', tune=500)
            #else:
            trace = pm.sample(samples_wanted, chains=number_chains, init='advi', cores=1, tune=500)
        return trace

    def fit_NLM(self, x_train, y_train, mac):
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
        - thin: Thinning factor
        :return: Samples from the posterior distribution sampled via the NUTS pymc3.
        """
        D = self.params['H']  # dimensionality of the feature map
        samples = self.pymc3_sampling_modif(self.forward(self.weights, x_train, partial=True), self.params['D_out'], y_train,
                                      D, mac)
        print(samples['w'].shape)
        print(samples['intercept'].shape)
        return np.concatenate((samples['w'].reshape(samples['w'].shape[0], samples['w'].shape[1]*samples['w'].shape[2]), samples['intercept']), axis=1).flatten()   # modif david

    def sample_posterior(self, x_train, y_train, params_fit, mac):
        print('Currently fitting a Neural Network for the Classification task')
        self.fit_MLE(x_train, y_train, params_fit,reg_param=.01)
        print('NN trained ! Now, thanks to the feature map, we are going to sample the posterior weights')
        samples = self.fit_NLM(x_train, y_train, mac)
        print('Posterior samples sampled !')
        return samples

    def sample_models(self, x_train, y_train, params_fit, num_models, mac):
        posterior_weights = np.array(
            self.sample_posterior(x_train, y_train, params_fit, mac))  # size : (num_samples, num_weights)
        posterior_weights = posterior_weights[int(posterior_weights.shape[0]/2):, :]
        print('Now, thanks to the posterior, we are going to create ' + str(
            num_models) + ' different classification models')
        indexes_chosen = random.choices(range(posterior_weights.shape[0]), k=num_models)
        selected_weights = posterior_weights[indexes_chosen, :]  # size : (num_models, num_weights)
        index_output_layer = - self.params['H'] * self.params['D_out'] - self.params['D_out']
        weights_independent = self.weights[0][:index_output_layer]
        models = []
        for weight in selected_weights:
            complete_weight_model = np.array(
                list(weights_independent.flatten()) + list(weight.flatten()))  # make sure we get this right
            complete_weight_model = complete_weight_model.reshape((1, -1))
            models.append(Classifier(complete_weight_model, self.forward))
        return models

    def uncertainty_computation(self, models, points, test_points=None):
        """
        This function computes the epistemic uncertainty of points in test_points, as is done in ...
        For now, we do not use the entropy or the expectanty, we just get a rought approximate of the epistemic uncertainty
        by adding the variance of the probability predictions for every class
        :param test_points: points for which we want to calculate the uncertainty
        :param models: sample weights from the posterior (for the last hidden layer to output):
        this gives a NN with fixed weights, which to each input associates an output (vector of probabilities): this is one model
        models is a list [model1, model2, ...]. They should we created via the Classifier class
        :return: The epistemic uncertainty for each point intest_points, as a list
        """
        train_epistemic_uncer = []
        for point in points:
            list_p = []
            for i in range(len(models)):
                list_p.append(models[i].predict_proba(np.array(point).reshape(1, -1)))
            train_epistemic_uncer.append(np.sum(np.std(np.array(list_p), axis=1)))
        test_epistemic_uncer = []
        if test_points:
            for point in test_points:
                list_p = []
                for i in range(len(models)):
                    list_p.append(models[i].predict_proba(np.array(point).reshape(1, -1)))
                test_epistemic_uncer.append(np.sum(np.std(np.array(list_p), axis=1)))
        return train_epistemic_uncer, test_epistemic_uncer


class Classifier:
    """
    This class implements the scikit-learn API for our Neural Network.
    """

    def __init__(self, weights, forward):
        self.weights = weights
        self.forward = forward

    def predict(self, x):
        p = self.forward(self.weights, x.T)
        classes = []
        for i in range(p.shape[0]):
            classe = np.zeros(p.shape[1])
            biggest_probability = np.argmax(p[i]).flatten()
            classe[biggest_probability] = 1
            classes.append(classe)
        return np.array(classes)

    def predict_proba(self, x):
        return self.forward(self.weights, x.T)
