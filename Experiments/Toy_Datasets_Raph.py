import autograd.numpy as np
import autograd.numpy as np
from sklearn import cluster, datasets
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


def two_clusters_gaussian(params, n_samples, test_points=None):
    """
    :param params: should be a list of length K, K being the number of classes you wish to create
    for every class 0 <= k <=K-1, params[k] should be a dictionnary containing two keys: mean and covariance_matrix.
    The shapes expected for mean are D and covariance_matrix are D*D where D is the number of features for every
    datapoint.
    :param n_samples: number of samples you wish to create for every cluster
    :param test_points: OOD points
    :return: x of len(K*n_samples, n_features) and y of shape (K*n_samples). For both x and y, the features pertain
    sequentially to every class 0 <= k <= K-1
    """
    if params:
        if isinstance(params, list):  # params is a list
            K = len(params)
        else:  # params is a numpy array
            K = params.shape[0]
        x = np.array([0, 0])
        for k, param in enumerate(params):
            param_k = params[k]
            try:
                mean_k, cov_k = param_k['mean'], param_k['covariance_matrix']
            except KeyError:
                raise KeyError('The parameters for class ' + str(
                    k) + 'are not in the right dictionnary format. Please use mean and covariance_matrix')
            assert len(mean_k) == cov_k.shape[0] == cov_k.shape[1], 'Wrong shapes for the parameters of class ' + str(k)
            samples_class_k = np.random.multivariate_normal(mean_k, cov_k, n_samples)
            x = np.vstack((x, samples_class_k))
        y = np.array([[k] * n_samples for k in range(K)])
        return x[1:, :], np.array(y).flatten()
    else:
        raise BaseException().args


def create_two_circular_classes(n=1500, noise_input=0.05, plot=False):
    """
    INPUT:
    n is the target number of points in each circle (note that this is a target)
    noise is the noise used in sklearn make_circles function
    plot: if TRUE, will return a plot of the two classes of points (in red and yellow)
    as well as the boundary class (in blue)
    
    OUPUT: boundary, class1, class2
    the two classes of points
    the boundary class
    
    """
    # Generating the data using sklearn built-in function
    noisy_circles_1 = datasets.make_circles(n_samples=n, factor=.0, noise=noise_input)
    noisy_circles_2 = datasets.make_circles(n_samples=n, factor=.3, noise=noise_input)
    noisy_circles_3 = datasets.make_circles(n_samples=n, factor=.5, noise=noise_input)
    noisy_circles_4 = datasets.make_circles(n_samples=n, factor=.7, noise=noise_input)

    X_1 = []
    Y_1 = []
    for i in range(len(noisy_circles_1[0])):
        # make_circles creates two circles, we only want to create one
        if (noisy_circles_1[0][i][0]) ** 2 + noisy_circles_1[0][i][1] ** 2 < .7:
            X_1.append(noisy_circles_1[0][i][0])
            Y_1.append(noisy_circles_1[0][i][1])

    X_2 = []
    Y_2 = []
    for i in range(len(noisy_circles_2[0])):
        # make_circles creates two circles, we only want to create one
        if (noisy_circles_2[0][i][0]) ** 2 + noisy_circles_2[0][i][1] ** 2 < .7:
            X_2.append(noisy_circles_2[0][i][0])
            Y_2.append(noisy_circles_2[0][i][1])
    X_3 = []
    Y_3 = []
    for i in range(len(noisy_circles_3[0])):
        # make_circles creates two circles, we only want to create one
        if (noisy_circles_3[0][i][0]) ** 2 + noisy_circles_3[0][i][1] ** 2 < .7:
            X_3.append(noisy_circles_3[0][i][0])
            Y_3.append(noisy_circles_3[0][i][1])

    X_4 = []
    Y_4 = []
    for i in range(len(noisy_circles_4[0])):
        # make_circles creates two circles, we only want to create one
        if (noisy_circles_4[0][i][0]) ** 2 + noisy_circles_4[0][i][1] ** 2 < .7:
            X_4.append(noisy_circles_4[0][i][0])
            Y_4.append(noisy_circles_4[0][i][1])

    if plot:
        plt.plot(X_1, Y_1, 'x', c='r')
        plt.plot(X_2, Y_2, 'x', c='b')
        plt.plot(X_3, Y_3, 'x', c='y')
        plt.plot(X_4, Y_4, 'x', c='b')
        plt.show()

    boundary = [X_2 + X_4, Y_2 + Y_4]
    class1 = [X_1, Y_1]
    class2 = [X_3, Y_3]

    return boundary, class1, class2


def plot_decision_boundary(x, y, models, ax, poly_degree=1, test_points=None, shaded=True):
    '''
    plot_decision_boundary plots the training data and the decision boundary of the classifier.
    input:
       x - a numpy array of size N x 2, each row is a patient, each column is a biomarker
       y - a numpy array of length N, each entry is either 0 (no cancer) or 1 (cancerous)
       models - an array of classification models
       ax - axis to plot on
       poly_degree - the degree of polynomial features used to fit the model
       test_points - test data
       shaded - whether or not the two sides of the decision boundary are shaded
    returns:
       ax - the axis with the scatter plot

    '''
    # Plot data
    # from one-hot encode to array
    if y.shape[1] > 1:
        y = np.argmax(y, axis=1).flatten()
    num_classes = np.max(y) + 1
    for k in range(num_classes):
        ax.scatter(x[y == k, 0], x[y == k, 1], alpha=0.2, label='class ' + str(k))

    # Create mesh
    xmin = np.min(x.flatten()) - 3
    xmax = np.max(x.flatten()) + 3
    interval = np.arange(xmin, xmax, 0.1)
    n = np.size(interval)
    x1, x2 = np.meshgrid(interval, interval)
    x1 = x1.reshape(-1, 1)
    x2 = x2.reshape(-1, 1)
    xx = np.concatenate((x1, x2), axis=1)

    if len(models) > 1:
        alpha_line = 0.1
        linewidths = 0.1
    else:
        alpha_line = 0.8
        linewidths = 0.5

    i = 0

    for model in models:
        yy = model.predict(xx)
        yy = np.array([np.argmax(y) for y in yy])
        yy = yy.reshape((n, n))

        # Plot decision surface
        x1 = x1.reshape(n, n)
        x2 = x2.reshape(n, n)
        if shaded:
            ax.contourf(x1, x2, yy, alpha=0.1 * 1. / (i + 1) ** 2, cmap='bwr')
        ax.contour(x1, x2, yy, colors='black', linewidths=linewidths, alpha=alpha_line)

        i += 1

    if test_points is not None:
        for i in range(len(test_points)):
            pt = test_points[i]
            if i == 0:
                ax.scatter(pt[0], pt[1], alpha=1., s=50, color='black', label='test data')
            else:
                ax.scatter(pt[0], pt[1], alpha=1., s=50, color='black')

    ax.set_xlim((xmin, xmax))
    ax.set_ylim((xmin, xmax))
    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')
    ax.legend(loc='best')
    return ax


if __name__ == '__main__':
    pass
