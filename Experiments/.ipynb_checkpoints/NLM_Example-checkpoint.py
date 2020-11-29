import warnings

warnings.filterwarnings('ignore')

from Neural_Network import NLM, Classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Toy_Datasets import two_clusters_gaussian, plot_decision_boundary
import autograd.numpy as np
from pandas import get_dummies
from Hamiltonian_MC import hmc
import matplotlib.pyplot as plt
from Bayesian_pdf import get_log_prior, get_log_likelihood
from sklearn.linear_model import LogisticRegression


# import tensorflow as tf


def fit_MLE_0(X, y, architecture, threshold_classification, params, random=0, exigence=True):
    """
    The role of this test is to first check that our Neural Network properly learns. The way our Neural Network is trained,
    we are estimating MLE parameters for W. This will enable us to subsequently extract the feature map. Note that one
    feature map is associated with one dataset. here, our Neural Network is made for Classification.
    :param X: the X coordinates, is going to be split into train and val
    :param y: the labels, is going to be split into train and val
    :param architecture: architecture of our Neural Network
    :param random: random_state, for reproduciility of learning
    :return:
    """
    nlm = NLM(architecture)
    y = get_dummies(y).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=random)
    nlm.fit_MLE(X_train.T, y_train.T, params)
    classifier = Classifier(nlm.weights, nlm.forward)
    y_pred_test = classifier.predict(X_test)
    print(np.mean(np.sum(y_pred_test == y_test, axis=1) == 3))
    accuracy = np.mean(np.sum(y_pred_test == y_test, axis=1) == 3)
    print(accuracy)
    if exigence:
        assert accuracy > threshold_classification, "The MLE of the model does not seem to have converged"
    return nlm, y_test, y_pred_test


def fit_MLE_1(max_iteration, n_samples, exigence=False):
    """
    The parameters of the last experiment we ran for this model are
    :param max_iteration: 20000
    :return: an accuracy on the validation set of 0.975.
    """
    params_1 = {'mean': [1, 1], 'covariance_matrix': 0.5 * np.eye(2)}
    params_2 = {'mean': [-1, -1], 'covariance_matrix': 0.5 * np.eye(2)}
    params_3 = {'mean': [-1, 1], 'covariance_matrix': 0.5 * np.eye(2)}
    params = [params_1, params_2, params_3]
    X, y = two_clusters_gaussian(params, n_samples)
    activation_fn_type = 'relu'
    activation_fn = lambda x: np.maximum(np.zeros(x.shape), x)
    width = 5
    hidden_layers = 2
    input_dim = 2
    output_dim = 3
    architecture = {'width': width,
                    'hidden_layers': hidden_layers,
                    'input_dim': input_dim,
                    'output_dim': output_dim,
                    'activation_fn_type': 'relu',
                    'activation_fn_params': 'rate=1',
                    'activation_fn': activation_fn}
    rand_state = 0
    random = np.random.RandomState(rand_state)
    params = {'step_size': 1e-3,
              'max_iteration': max_iteration,
              'random_restarts': 1}
    nlm, y_test, y_pred = fit_MLE_0(X, y, architecture, 0.8, params, random, exigence=False)
    return nlm, y_test, y_pred, X, y


def feature_map(max_iteration):
    """
    THis function tests the effectiveness of the partial forward function. It also tests the dimension of the output
    of the partial forward mode, ie the dimension of the feature map produced.
    """
    params_1 = {'mean': [1, 1], 'covariance_matrix': 0.5 * np.eye(2)}
    params_2 = {'mean': [-1, -1], 'covariance_matrix': 0.5 * np.eye(2)}
    params = [params_1, params_2]
    X, y = two_clusters_gaussian(params, 100)
    activation_fn_type = 'relu'
    activation_fn = lambda x: np.maximum(np.zeros(x.shape), x)
    width = 5
    hidden_layers = 1
    input_dim = 2
    output_dim = 1
    architecture = {'width': width,
                    'hidden_layers': hidden_layers,
                    'input_dim': input_dim,
                    'output_dim': output_dim,
                    'activation_fn_type': 'relu',
                    'activation_fn_params': 'rate=1',
                    'activation_fn': activation_fn}
    rand_state = 0
    random = np.random.RandomState(rand_state)
    params = {'step_size': 1e-3,
              'max_iteration': max_iteration,
              'random_restarts': 1}
    nlm = NLM(architecture)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=random)
    nlm.fit_MLE(X_train.T, y_train.reshape(1, -1), params)
    feature = nlm.forward(nlm.weights, X_test.T, partial=True)
    assert feature.shape[1] == nlm.params['H']
    print(nlm.weights.shape)
    print('The feature map is from ' + str(nlm.params['D_in']) + ' to ' + str(nlm.params['H']))
    return feature


def softmax_in_NN_2(max_iteration):
    """
    This function tests the validity of the forward pass + the softmax output when trained on K >=2 classes. It also tests the dimension of the output.
    """
    params_1 = {'mean': [1, 1], 'covariance_matrix': 0.5 * np.eye(2)}
    params_2 = {'mean': [-1, -1], 'covariance_matrix': 0.5 * np.eye(2)}
    params = [params_1, params_2]
    X, y = two_clusters_gaussian(params, 1000)
    activation_fn_type = 'relu'
    activation_fn = lambda x: np.maximum(np.zeros(x.shape), x)
    width = 5
    hidden_layers = 1
    input_dim = 2
    output_dim = 2
    architecture = {'width': width,
                    'hidden_layers': hidden_layers,
                    'input_dim': input_dim,
                    'output_dim': output_dim,
                    'activation_fn_type': 'relu',
                    'activation_fn_params': 'rate=1',
                    'activation_fn': activation_fn}
    rand_state = 0
    random = np.random.RandomState(rand_state)
    params = {'step_size': 1e-3,
              'max_iteration': max_iteration,
              'random_restarts': 1}
    nlm = NLM(architecture)
    y = get_dummies(y).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=random)
    nlm.fit_MLE(X_train.T, y_train.T, params)
    classifier = Classifier(nlm.weights, nlm.forward)
    y_pred_test = classifier.predict(X_test)
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred_test)  # ok for the transformations, accuracy not checked
    print(accuracy)


def softmax_in_NN_3(max_iteration):
    """
    This function tests the validity of the forward pass + the softmax output when trained on K >=2 classes. It also tests the dimension of the output.
    """
    params_1 = {'mean': [1, 1], 'covariance_matrix': 0.5 * np.eye(2)}
    params_2 = {'mean': [-1, -1], 'covariance_matrix': 0.5 * np.eye(2)}
    params_3 = {'mean': [8, 9], 'covariance_matrix': 0.5 * np.eye(2)}
    params = [params_1, params_2, params_3]
    X, y = two_clusters_gaussian(params, 1000)
    activation_fn_type = 'relu'
    activation_fn = lambda x: np.maximum(np.zeros(x.shape), x)
    width = 5
    hidden_layers = 1
    input_dim = 2
    output_dim = 3
    architecture = {'width': width,
                    'hidden_layers': hidden_layers,
                    'input_dim': input_dim,
                    'output_dim': output_dim,
                    'activation_fn_type': 'relu',
                    'activation_fn_params': 'rate=1',
                    'activation_fn': activation_fn}
    rand_state = 0
    random = np.random.RandomState(rand_state)
    params = {'step_size': 1e-3,
              'max_iteration': max_iteration,
              'random_restarts': 1}
    nlm = NLM(architecture)
    y = get_dummies(y).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=random)
    nlm.fit_MLE(X_train.T, y_train.T, params)
    classifier = Classifier(nlm.weights, nlm.forward)
    y_pred_test = classifier.predict(X_test)
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred_test)
    return accuracy


def softmax(y):  # checked, ok for softmax and the dimensions
    """
    This function is used to perform multi-class classification. This should be the activation function.
    We need to handle the cases where the shape of the input are going to be (1, K, batch_size)
    """
    # print(y)
    z = y.flatten().reshape(2, -1)
    # print(z)
    z = np.exp(z)
    return z / np.sum(z, axis=0)


def softmax_cat_ce():  # checked
    x = np.random.normal(0, 1, size=10)
    x = x.reshape((1, 2, -1))
    print(x.shape)
    print(x)
    print(softmax(x))
    print(np.log(softmax(x)))
    y = [1, 0, 0, 0, 1]
    y_train = get_dummies(y).values
    print(y_train.T)
    print(y_train.T * np.log(softmax(x)))
    Cat_cross_entropy = np.sum(y_train.T * np.log(softmax(x)), axis=0)  # vector of size(len(x_train, 1))
    print(Cat_cross_entropy)
    total_cat_ce = np.mean(Cat_cross_entropy)
    return total_cat_ce


def sample_NLM():
    params_1 = {'mean': [1, 1], 'covariance_matrix': 0.5 * np.eye(2)}
    params_2 = {'mean': [-1, -1], 'covariance_matrix': 0.5 * np.eye(2)}
    params = [params_1, params_2]
    X, y = two_clusters_gaussian(params, 100)
    activation_fn_type = 'relu'
    activation_fn = lambda x: np.maximum(np.zeros(x.shape), x)
    width = 5
    hidden_layers = 1
    input_dim = 2
    output_dim = 1
    architecture = {'width': width,
                    'hidden_layers': hidden_layers,
                    'input_dim': input_dim,
                    'output_dim': output_dim,
                    'activation_fn_type': 'relu',
                    'activation_fn_params': 'rate=1',
                    'activation_fn': activation_fn,
                    'prior': 'normal',
                    'prior_parameters': {'mean': np.zeros(5), 'covariance_matrix': np.eye(5)},
                    'likelihood': 'logistic',
                    'likelihood_parameters': None}
    rand_state = 0
    random = np.random.RandomState(rand_state)
    params_fit = {'step_size': 1e-3,
                  'max_iteration': 1000,
                  'random_restarts': 1}
    params_hmc = {'num_samples': 2000,
                  'step_size': 1e-2,
                  'L': 50,
                  'burn': 0.1,
                  'thin': 2}
    nlm = NLM(architecture)
    nlm.sample(X.T, y.reshape(1, -1), hmc, params_fit, params_hmc)


def test_decision_boundary(max_iteration):
    params_1 = {'mean': [1, 1], 'covariance_matrix': 0.5 * np.eye(2)}
    params_2 = {'mean': [-1, -1], 'covariance_matrix': 0.5 * np.eye(2)}
    params_3 = {'mean': [8, 9], 'covariance_matrix': 0.5 * np.eye(2)}
    params = [params_1, params_2, params_3]
    X, y = two_clusters_gaussian(params, 1000)
    activation_fn_type = 'relu'
    activation_fn = lambda x: np.maximum(np.zeros(x.shape), x)
    width = 10
    hidden_layers = 3
    input_dim = 2
    output_dim = 3
    architecture = {'width': width,
                    'hidden_layers': hidden_layers,
                    'input_dim': input_dim,
                    'output_dim': output_dim,
                    'activation_fn_type': 'relu',
                    'activation_fn_params': 'rate=1',
                    'activation_fn': activation_fn}
    rand_state = 0
    random = np.random.RandomState(rand_state)
    params = {'step_size': 1e-3,
              'max_iteration': max_iteration,
              'random_restarts': 1}
    nlm = NLM(architecture)
    y = get_dummies(y).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=random)
    nlm.fit_MLE(X_train.T, y_train.T, params)
    classifier = [Classifier(nlm.weights, nlm.forward)]
    fig, ax = plt.subplots(1, figsize=(20, 10))
    plot_decision_boundary(X, y, classifier, ax)
    plt.show()


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
    cov = 10 * np.eye(3)
    D = 3

    def log_likelihood(w):
        theta = sigmoid(w[0] + np.dot(x, w[1:]))
        theta = np.clip(theta, 1e-15, 1 - 1e-15)
        loglkhd = y * np.log(theta) + (1 - y) * np.log(1 - theta)
        return np.sum(loglkhd)

    def log_normal_prior(W):
        logprior = -0.5 * (np.log(np.linalg.det(cov)) + D * np.log(2 * np.pi))
        logprior += -0.5 * np.dot(np.dot(W - mean, np.linalg.inv(cov)), (W - mean).T)
        return logprior

    log_prior = log_normal_prior
    log_likelihood = log_likelihood
    lr = LogisticRegression(C=1., penalty='l2', solver='saga', tol=0.1)
    lr.fit(x, y)
    position_init = np.hstack((lr.coef_.flatten(), lr.intercept_))
    position_init = position_init.reshape((1, 3))[0]
    samples = hmc(log_prior, log_likelihood, 1000, 1e-3, 20, position_init, 0.1, 1)


def NLM_test():
    params_1 = {'mean': [1, 1], 'covariance_matrix': 0.5 * np.eye(2)}
    params_2 = {'mean': [-1, -1], 'covariance_matrix': 0.5 * np.eye(2)}
    params_3 = {'mean': [-1, 1], 'covariance_matrix': 0.5 * np.eye(2)}
    params = [params_1, params_2, params_3]
    X, y = two_clusters_gaussian(params, 100)
    activation_fn_type = 'relu'
    activation_fn = lambda x: np.maximum(np.zeros(x.shape), x)
    width = 5
    hidden_layers = 2
    input_dim = 2
    output_dim = 3
    architecture = {'width': width,
                    'hidden_layers': hidden_layers,
                    'input_dim': input_dim,
                    'output_dim': output_dim,
                    'activation_fn_type': 'relu',
                    'activation_fn_params': 'rate=1',
                    'activation_fn': activation_fn}
    rand_state = 0
    random = np.random.RandomState(rand_state)
    params = {'step_size': 1e-3,
              'max_iteration': 2000,
              'random_restarts': 1}
    nlm = NLM(architecture)
    y = get_dummies(y).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=random)
    nlm.fit_MLE(X_train.T, y_train.T, params)
    samples = nlm.fit_NLM(X_train.T, y_train.T)
    return samples


if __name__ == '__main__':
    samples = NLM_test()
    fig, ax = plt.subplots(1, figsize=(20, 10))
    burn_in = 0.2
    thinning_factor = 3
    print(samples['w'])
    w = []
    for i in range(len(samples['w'])):
        w.append(samples['w'][i][1])
    wp = w[int(burn_in * len(samples)):][::thinning_factor]
    fig, ax = plt.subplots(1, 1, figsize=(20, 5))
    ax.plot(range(len(wp)), wp)
    plt.show()

