import warnings

warnings.filterwarnings('ignore')
from Neural_Network import NLM, Classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Toy_Datasets.Toy_Datasets import two_clusters_gaussian, plot_decision_boundary, plot_uncertainty
from Toy_Datasets.Toy_Datasets_2D import create_two_circular_classes_outer
import autograd.numpy as np
from pandas import get_dummies
from old.Hamiltonian_MC import hmc
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from Entropy import epistemic_uncertainty, expected_aleatoric_uncertainty, total_uncertainty


# import tensorflow as tf



def fit_MLE(max_iteration, n_samples):
    """
    The role of this test is to first check that our Neural Network properly learns. The way our Neural Network is trained,
    we are estimating MLE parameters for W. This will enable us to subsequently extract the feature map. Note that one
    feature map is associated with one dataset. here, our Neural Network is made for Classification.
    The parameters of the last experiment we ran for this model are
    :param max_iteration: 20000
    :return: an accuracy on the validation set of 0.975.
    """
    params_1 = {'mean': [1, 1], 'covariance_matrix': 0.5 * np.eye(2)}
    params_2 = {'mean': [-1, -1], 'covariance_matrix': 0.5 * np.eye(2)}
    params_3 = {'mean': [-1, 1], 'covariance_matrix': 0.5 * np.eye(2)}
    params = [params_1, params_2, params_3]
    X, y = two_clusters_gaussian(params, n_samples)
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
    nlm = NLM(architecture)
    y = get_dummies(y).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=random)
    nlm.fit_MLE(X_train.T, y_train.T, params, reg_param=.01)
    classifier = Classifier(nlm.weights, nlm.forward)
    y_pred_test = classifier.predict(X_test)
    accuracy = np.mean(np.sum(y_pred_test == y_test, axis=1) == 3)
    fig, ax = plt.subplots(1, figsize=(20, 10))
    plot_decision_boundary(X_train, y_train, [classifier], ax)
    plt.savefig('Debug_David/MLE_train_NN_3_classes_1.png')
    plt.show()
    return nlm, y_test, y_pred_test, X, y


def decision_boundary(max_iteration):
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


def feature_map(max_iteration):
    """
    THis function tests the effectiveness of the partial forward function. It also tests the dimension of the output
    of the partial forward mode, ie the dimension of the feature map produced.
    """
    params_1 = {'mean': [1, 1], 'covariance_matrix': 0.5 * np.eye(2)}
    params_2 = {'mean': [-1, -1], 'covariance_matrix': 0.5 * np.eye(2)}
    params_3 = {'mean': [-1, 1], 'covariance_matrix': 0.5 * np.eye(2)}
    params = [params_1, params_2, params_3]
    X, y = two_clusters_gaussian(params, 100)
    y = get_dummies(y).values
    activation_fn = lambda x: np.maximum(np.zeros(x.shape), x)
    width = 5
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=random)
    nlm.fit_MLE(X_train.T, y_train.T, params)
    print(X_test.shape)
    feature = nlm.forward(nlm.weights, X_test.T, partial=True)
    print(feature.shape)
    print(feature)
    print(feature[0].shape)
    print(feature[0])
    weights = nlm.weights
    print(weights)
    feature_weights = nlm.get_feature_map_weights()
    print(feature_weights)
    print('The feature map is from ' + str(nlm.params['D_in']) + ' to ' + str(nlm.params['H']))
    return feature



def sample_NLM(max_iteration, n_samples):
    params_1 = {'mean': [1, 1], 'covariance_matrix': 0.5 * np.eye(2)}
    params_2 = {'mean': [-1, -1], 'covariance_matrix': 0.5 * np.eye(2)}
    params_3 = {'mean': [-1, 1], 'covariance_matrix': 0.5 * np.eye(2)}
    params = [params_1, params_2, params_3]
    X, y = two_clusters_gaussian(params, n_samples)
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
    nlm = NLM(architecture)
    y = get_dummies(y).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=random)
    nlm = NLM(architecture)
    print(y_train.shape)
    models = nlm.sample_models(X_train.T, y_train.T, params, 100, mac=False)
    accuracies = []
    for model in models:
        y_pred_test = model.predict(X_test)
        accuracy = np.mean(np.sum(y_pred_test == y_test, axis=1) == 3)
        accuracies.append(accuracy)
    print(accuracies)
    print('The mean accuracy for all of the models is ', np.mean(accuracies))
    fig, ax = plt.subplots(1, figsize=(20, 10))
    plot_decision_boundary(X_train, y_train, models, ax)
    plt.savefig('Debug_David/NLM_train_3_classes.png')
    plt.show()


def sample_NLM_Gael(max_iteration, n_samples):
    params_1 = {'mean': [4, 4], 'covariance_matrix': 0.5 * np.eye(2)}
    params_2 = {'mean': [-4, -4], 'covariance_matrix': 0.5 * np.eye(2)}
    params_3 = {'mean': [0, 0], 'covariance_matrix': 0.5 * np.eye(2)}
    params_4 = {'mean': [1, -5], 'covariance_matrix': 1 * np.eye(2)}

    params = [params_1, params_2, params_3, params_4]
    X, y = two_clusters_gaussian(params, n_samples)
    activation_fn = lambda x: np.maximum(np.zeros(x.shape), x)
    width = 5
    hidden_layers = 2
    input_dim = 2
    output_dim = 4
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
    y = get_dummies(y).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=random)
    nlm = NLM(architecture)
    models = nlm.sample_models(X_train.T, y_train.T, params, 100, mac=False)
    accuracies = []
    for model in models:
        y_pred_test = model.predict(X_test)
        y_pred_labels = np.argmax(y_pred_test, axis=1).flatten()
        y_test_labels = y = np.argmax(y_test, axis=1).flatten()
        accuracy = accuracy_score(y_test_labels, y_pred_labels)
        accuracies.append(accuracy)
        break
    print('The mean accuracy for all of the models is ', np.mean(accuracies))
    fig, ax = plt.subplots(1, figsize=(20, 10))
    plot_decision_boundary(X_train, y_train, models, ax)
    plt.savefig('Debug_David/NLM_train_3_classes_reproduce_Gael.png')
    plt.show()



def decision_uncertainty():
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
              'max_iteration': 7500,
              'random_restarts': 1}
    nlm = NLM(architecture)
    y = get_dummies(y).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=random)
    models = nlm.sample_models(X_train.T, y_train.T, params, num_models=100, mac=False)
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    plot_uncertainty(X_test, y_test, models, axes[0], epistemic_uncertainty)
    plot_uncertainty(X_test, y_test, models, axes[1], total_uncertainty)
    plot_uncertainty(X_test, y_test, models, axes[2], expected_aleatoric_uncertainty)
    plt.show()
    return models, X_test, y_test


def bacoun_1():
    boundary, class1, class2 = create_two_circular_classes_outer(n=1000, noise_input=0.05, plot=True, distance=2)
    X_try = np.hstack((boundary, class1, class2)).T
    y_try = np.array([[k] * 500 for k in range(3)])
    y_try = np.array(y_try).flatten()
    ###relu activation
    activation_fn_type = 'relu'
    activation_fn = lambda x: np.maximum(np.zeros(x.shape), x)

    ###neural network model design choices
    width = 5
    hidden_layers = 3
    input_dim = 2
    output_dim = 3

    architecture = {'width': width,
                    'hidden_layers': hidden_layers,
                    'input_dim': input_dim,
                    'output_dim': output_dim,
                    'activation_fn_type': 'relu',
                    'activation_fn_params': 'rate=1',
                    'prior': 'normal',
                    'prior_parameters': {'mean': np.zeros(5), 'covariance_matrix': np.eye(5)},
                    'likelihood': 'logistic',
                    'activation_fn': activation_fn}

    # set random state to make the experiments replicable
    rand_state = 0
    random = np.random.RandomState(rand_state)

    # instantiate a Feedforward neural network object
    nlm2 = NLM(architecture, random=random)
    y_try_ = get_dummies(y_try).values
    X_train_try, X_test_try, y_train_try, y_test_try = train_test_split(X_try, y_try_, train_size=0.9,
                                                                        random_state=random)
    params = {'step_size': 1e-3,
              'max_iteration': 5000,
              'random_restarts': 1}

    # fit my neural network to minimize MSE on the given data
    # nlm.fit_MLE(x.T, y.reshape(1,-1), params)

    nlm2.fit_MLE(X_train_try.T, y_train_try.T, params)
    classifier = [Classifier(nlm2.weights, nlm2.forward)]
    fig, ax = plt.subplots(1, figsize=(20, 10))
    plot_decision_boundary(X_try, y_try_, classifier, ax, shaded=False)
    plt.show()


if __name__ == '__main__':
    sample_NLM(5000, 200)

