import autograd.numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.graphics.tsaplots import plot_acf

def entropy_vec(v):
	D=len(v)
	su=0
	for i in range(D):
		su+=np.log(v[i])*v[i]
	return -su


# TO DO
def total_uncertainty(y,trace,x):
#   trace_subset=take subset of trace
#   m number of weights samples in trace)subset
#   s=0
#   for weights in trace_subset:
#			s+=p(y|x^*,W)/m
	return entropy_vec(s)

# TO DO
def expected_aleatoric_uncertainty(y,trace,x):
#   trace_subset=take subset of trace
#   m number of weights samples in trace)subset
# 	s=0
#	for weight in trace_subset:
#			a=calculaer p(y|x^*,W)
#			s+=entropy_vec(a)
	return s/m


def epistemic_uncertainty(y,trace,x):
	return total_uncertainty(y,trace,x)-expected_aleatoric_uncertainty(y,trace,x)