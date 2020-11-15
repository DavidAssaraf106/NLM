"""

Create a toy dataset with two cluster Gaussian well separated by a linear boundary between them

params_1 = {'mean': [1, 1], 'covariance_matrix': 0.5 * np.eye(2)}
params_2 = {'mean': [-1, -1], 'covariance_matrix': 0.5 * np.eye(2)}
params = [params_1, params_2]
X, y = two_clusters_gaussian(params, 100)


"""

"""
Create a NLM for classification. input_dim should be the number of features your training points have an output_dim should 
be your number of classes (NB:by convention, for 2 classes, our output_dim is 1)


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

nn = NLM(architecture, random=random)


"""

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
"""