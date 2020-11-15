from Neural_Network import NLM, Classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Toy_Datasets import two_clusters_gaussian
import autograd.numpy as np

def fit_MLE_0(X, y, architecture, threshold_classification, params, random, exigence):
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=random)
    nlm.fit_MLE(X_train.T, y_train.reshape(1, -1), params)
    classifier = Classifier(nlm.weights, nlm.forward)
    y_pred_test = classifier.predict(X_test)
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred_test.flatten())
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
    params = [params_1, params_2]
    X, y = two_clusters_gaussian(params, n_samples)
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
    print('The feature map is from '+str(nlm.params['D_in']) + ' to ' + str(nlm.params['H']))
    return feature





if __name__ == '__main__':
    pass





