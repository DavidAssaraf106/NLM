import autograd.numpy as np
import autograd.numpy as np
import matplotlib.pyplot as plt
import warnings
from pymc3 import Model
import pymc3 as pm


def pymc3_sampling(D, sigma_in, out):
	"""
	In
	"""
	with pm.Model() as replacing_HMC:  
	    # w has a prior: N(0,1) 
	    w = pm.Normal('w', mu=0, sigma=sigma_in, shape=(D+1)) 
	    linear_combi = pm.math.dot(X_subset,w)
	    # TO CHECK SOFTMAX
	    thetas = pm.Deterministic('theta', pm.math.softmax(linear_combi))
	    # Y commes from a Categorical(theta)
	    y_obs = pm.Categorical('y_obs', p=thetas, observed=y_subset)
	    trace = pm.sample(5000,chains=2)
	return trace





###### A faire #############
# tensorflow: tenosrflowprobability OU PytorchPyro 
# Ou sinon: numpyro
# Aussi faire BBVI, 