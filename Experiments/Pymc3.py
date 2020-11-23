import autograd.numpy as np
import autograd.numpy as np
import matplotlib.pyplot as plt
import warnings
from pymc3 import Model
import pymc3 as pm


def pymc3_sampling():
	with pm.Model() as replacing_HMC:  
	    # w has a prior: N(0,10) TO CHANGE
	    w = pm.Normal('w', mu=0, sigma=np.sqrt(10), shape=(D+1))
	    # compute w_0 + w^TX 
	    linear_combi = w[0] + pm.math.dot(X_subset, w[1:])
	    theta = pm.Deterministic('theta', pm.math.sigmoid(linear_combi))
	    # Y commes from a Bernouilli(theta)
	    y_obs = pm.Bernoulli('y_obs', p=theta, observed=y_subset)
	    trace = pm.sample(5000,chains=2)
	return trace

