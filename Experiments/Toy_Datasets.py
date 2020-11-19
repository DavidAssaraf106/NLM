import autograd.numpy as np

# To use sklearn datasets:
from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
# further packages needed
from itertools import cycle, islice




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
                raise KeyError('The parameters for class ' + str(k) + 'are not in the right dictionnary format. Please use mean and covariance_matrix')
            assert len(mean_k) == cov_k.shape[0] == cov_k.shape[1], 'Wrong shapes for the parameters of class ' + str(k)
            samples_class_k = np.random.multivariate_normal(mean_k, cov_k, n_samples)
            x = np.vstack((x, samples_class_k))
        y = np.array([[k] * n_samples for k in range(K)])
        return x[1:, :], np.array(y).flatten()
    else:
        raise BaseException().args




if __name__ == '__main__':
    params_1 = {'mean': [1, 1], 'covariance_matrix': 0.5*np.eye(2)}
    params_2 = {'mean': [-1, -1], 'covariance_matrix': 0.5 * np.eye(2)}
    params = [params_1, params_2]
    X, y = two_clusters_gaussian(params, 100)



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
    
    X_1=[]
    Y_1=[]
    for i in range(len(noisy_circles_1[0])):
        # make_circles creates two circles, we only want to create one
        if (noisy_circles_1[0][i][0])**2+noisy_circles_1[0][i][1]**2<.7:
            X_1.append(noisy_circles_1[0][i][0])
            Y_1.append(noisy_circles_1[0][i][1])
            
    X_2=[]
    Y_2=[]
    for i in range(len(noisy_circles_2[0])):
        # make_circles creates two circles, we only want to create one
        if (noisy_circles_2[0][i][0])**2+noisy_circles_2[0][i][1]**2<.7:
            X_2.append(noisy_circles_2[0][i][0])
            Y_2.append(noisy_circles_2[0][i][1])
    X_3=[]
    Y_3=[]
    for i in range(len(noisy_circles_3[0])):
        # make_circles creates two circles, we only want to create one
        if (noisy_circles_3[0][i][0])**2+noisy_circles_3[0][i][1]**2<.7:
            X_3.append(noisy_circles_3[0][i][0])
            Y_3.append(noisy_circles_3[0][i][1])
            
    X_4=[]
    Y_4=[]
    for i in range(len(noisy_circles_4[0])):
        # make_circles creates two circles, we only want to create one
        if (noisy_circles_4[0][i][0])**2+noisy_circles_4[0][i][1]**2<.7:
            X_4.append(noisy_circles_4[0][i][0])
            Y_4.append(noisy_circles_4[0][i][1])
            
    if plot:
        plt.plot(X,Y,'x',c='r')
        plt.plot(X_2,Y_2,'x',c='b')
        plt.plot(X_3,Y_3,'x',c='y')
        plt.plot(X_4,Y_4,'x',c='b')
        plt.show()
        
    boundary=[X_2+X_4,Y_2+Y_4]
    class1=[X_1,Y_1]
    class2=[X_3,Y_3]
    
    return boundary, class1, class2

