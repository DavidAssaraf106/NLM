import autograd.numpy as np
import matplotlib.pyplot as plt
import warnings
from pymc3 import Model
import pymc3 as pm
import theano.tensor as T
from statsmodels.graphics.tsaplots import plot_acf



def pymc3_sampling(out_last_hidden_layer, output_dim, out_y, D, mu_wanted=0, tau_wanted=1, samples_wanted=1000, number_chains=2):
    """
    INPUTS:
    D: dimension of the last hidden layer
    sigma_in: std of the prior (a normal, with mean 0)
    I AM USING THE CONVENTION THAT THE BIAS COME AT THE END

    OUTPUTS:
    the trace
    """
    with pm.Model() as replacing_HMC:  
        # w has a prior: N(0,1) 
        # Output dim number of bias
        w = pm.Normal('w', mu=mu_wanted, tau=tau_wanted, shape=(D*output_dim+output_dim)) 
        linear_combinations=[]
        for j in range(output_dim):
            dot=pm.math.dot(out_last_hidden_layer[0].T,w[j*D:j*D+D])+w[-j]
            linear_combi = pm.Deterministic('s'+str(j),dot)
            linear_combinations.append(linear_combi)
        thetas = pm.Deterministic('theta', T.nnet.softmax(linear_combinations))
        # Y commes from a Categorical(thetas)
        y_obs = pm.Categorical('y_obs', p=thetas.T, observed=out_y)
        trace = pm.sample(samples_wanted,chains=number_chains)
    return trace




##### A FINIR ###########################################
# def plot_trace(traces, weight_number=1, thin_wanted=3, burn_wanted=0.2, colour='b', lag=20):
#     """
#     INPUTS:
#     the trace, the weight number we want to plot, the thin and burn_in rates, the colour,
#     the lag for the autocorrelation plots


#     OUTPUTS:
#     the plots of the trace
#     the autocorrelation plot

#     """

#     burn_in=burn_wanted
#     thinning_factor=thin_wanted
#     w=[]
#     for i in range(len(traces['w'])):    
#         w.append(traces['w'][i][weight_number])
#     wp = w[int(burn_in*len(traces)):][::thinning_factor]  
#     fig, ax=plt.subplots(1,1,figsize=(20,5))
#     ax.plot(range(len(wp)), wp, color=colour)
#     ax.set_title('Trace plot for weight {}'.format(weight_number))
#     plot_acf(wp, lags=lag, label='w{}'.format(weight_number))




###### A faire #############
# fix HMC et faire pymc3 (NUTs) mais apres le mieux c'est tensorflow ou Pytorch


# tensorflow: tenosrflowprobability OU PytorchPyro 
# Ou sinon: numpyro


# Aussi faire BBVI 