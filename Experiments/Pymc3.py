import autograd.numpy as np
import autograd.numpy as np
import matplotlib.pyplot as plt
import warnings
from pymc3 import Model
import pymc3 as pm
import theano.tensor as T


def pymc3_sampling(D, sigma_in, out_last_hidden_layer):
	"""
	In
	"""
	with pm.Model() as replacing_HMC:  
	    # w has a prior: N(0,1) 
	    w = pm.Normal('w', mu=0, sigma=sigma_in, shape=(D+1)) 
	    linear_combi = pm.math.dot(X_subset,w)
	    thetas = pm.Deterministic('theta', T.nnet.softmax(linear_combi))
	    # or thetas = pm.Deterministic('theta', pm.math.softmax(linear_combi))?
	    # Y commes from a Categorical(thetas)
	    y_obs = pm.Categorical('y_obs', p=thetas, observed=out_last_hidden_layer)
	    trace = pm.sample(5000,chains=2)
	return trace





###### A faire #############
# fix HMC et faire pymc3 (NUTs) mais apres le mieux c'est tensorflow ou Pytorch


# tensorflow: tenosrflowprobability OU PytorchPyro 
# Ou sinon: numpyro


# Aussi faire BBVI 