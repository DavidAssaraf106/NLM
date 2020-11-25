# Description: creating functions to quantify the uncertainty
# Based on HW7


import autograd.numpy as np
import autograd.numpy as np
from sklearn import cluster, datasets
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


def uncertainty_test(test_points, models, printing=False):
    """
    Gives the epistemic uncertainty of points in test_points

    INPUT:
    test_points: points for which we want to calculate the uncertainty
    models: sample weights from the posterior (for the last hidden layer to output):
    this gives a NN with fixed weights, which to each input associates an output (vector of probabilities): this is one model
    models is a list [model1, model2, ...]
    NEED TO HAVE A PREDICT_PROBA FUNCTIONALITY 
    printin: if True, will print the uncertainty of each point
    
    OUTPUT:
    The epistemic uncertainty for each point intest_points, as a list
    """
    epistemic_uncer=[]
    for point in test_points:
        list_p=[]
        for i in range(len(models)):
            list_p.append(models[i].predict_proba(np.array(point).reshape(1,-1)))
        epistemic_uncer.append(np.var(list_p))
    if printing==True:
        for i in range(len(epistemic_uncer)):
            print('Test point: {}'.format(test_points[i]))
            print('The epistemic uncertainty for test point: {} is {}'.format(test_points[i], epistemic_uncer[i]))
    return epistemic_uncer





