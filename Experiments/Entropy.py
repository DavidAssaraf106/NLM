import autograd.numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.graphics.tsaplots import plot_acf


def entropy_vec(v):
	D=len(v)
	su=0
	for i in range(D):
		if v[i]==0:
			su+=0
		else:
			su+=np.log(v[i])*v[i]
	return -su


def total_uncertainty(x, models, n_iter=100):
	"""
	models is a list of objects of class classifiers
	"""
	# Step 1: estimate the expectation
   	s=0
   	for mod in models:
		s+=mod.predict_proba(x) # a vector in R^(output_dim)
    # Step 2: entropy
	return entropy_vec(s/len(models))


def expected_aleatoric_uncertainty(x, models, n_iter=100):
	"""
	models is a list of objects of class classifiers
	"""
	m=len(models)
 	s=0
	for mod in models:
		a=mod.predict_proba(x)
		s+=entropy_vec(a)
	return s/m


def epistemic_uncertainty(x, models, n_iter=100):
	"""
	models is a list of objects of class classifiers

	OUTPUT:
	The Mutual information (MI) 
	"""
	return total_uncertainty(x, models,n_iter=100)-expected_aleatoric_uncertainty(x, models,n_iter=100)
