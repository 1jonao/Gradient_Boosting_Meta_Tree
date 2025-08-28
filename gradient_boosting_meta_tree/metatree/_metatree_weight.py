# Code Writer : Ryota Maniwa 2023/8/15
import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp


def _weight_learning_rate(
        weight_size:int,
        learning_rate_value:float
        ):
    '''make meta-tree weight vector 

        Parameters
        ----------
	    weight_size : int
            A positive integer
        learning_rate_value : float
		    A real number in :math:`[0, 1]`

        Returns
        ----------
        prob_vec : numpy ndarray
    '''
    return np.full(weight_size,learning_rate_value)

def _proba_uniform(
        weight_size:int,
)->ArrayLike:
    '''make meta-tree weight vector 

        Parameters
        ----------
	    weight_size : int
            A positive integer

        Returns
        ----------
        prob_vec : numpy ndarray
    '''
    try:
        metatree_prob_vec = np.full(weight_size,1/weight_size)
    except:
        metatree_prob_vec = np.array([])
    return metatree_prob_vec

def _proba_posterior(
        log_likelihoods: ArrayLike,
)->ArrayLike:
    '''make meta-tree weight vector 

        Parameters
        ----------
        log_likelihoods: ArrayLike
            1 dimensional np.array of size=weight_size with positive floats, 

        Returns
        ----------
        prob_vec : numpy ndarray
    '''
    normalization_term = logsumexp(log_likelihoods)
    return np.exp(log_likelihoods - normalization_term), normalization_term

def _proba_posterior_smooth_dirichletprior(
        log_likelihoods: ArrayLike,
        smoothness_param_vecs: ArrayLike,
)->ArrayLike:
    '''make meta-tree weight vector 

        Parameters
        ----------
        log_likelihoods: ArrayLike
            1 dimensional np.array of size=weight_size with positive floats, 
        smoothness_param_vecs: ArrayLike,
            1 dimensional np.array of size=weight_size with positive floats. This vector serves as a hyperparameter of dirichlet prior distribuion over the prior weights of meta-trees.

        Returns
        ----------
        prob_vec : numpy ndarray
    '''
    smoothed_log_likelihoods = np.logaddexp(np.exp(log_likelihoods),np.log(smoothness_param_vecs))
    if (smoothness_param_vecs == -0).all():
        raise(RuntimeError(
            '''h0_metatree_weight_smoothness_pred is set to 0.(Theoretically this performs as metatree_weight_type_pred==\'proba_posterior\' but it doesn't performs as expected due to the round-offs).
Use metatree_weight_type_pred=\'proba_posterior\' or set h0_metatree_weight_smoothness_pred > 0.
            '''
        ))
    normalization_term = logsumexp(smoothed_log_likelihoods)
    return np.exp(smoothed_log_likelihoods - normalization_term), normalization_term

def _proba_posterior_smooth_exp_tilting(
        log_likelihoods: ArrayLike,
        smoothness_param_vecs: ArrayLike,
)->ArrayLike:
    '''make meta-tree weight vector 

        Parameters
        ----------
        log_likelihoods: ArrayLike
            1 dimensional np.array of size=weight_size with positive floats, 
        smoothness_param_vecs: ArrayLike,
            1 dimensional np.array of size=weight_size with positive floats. This vector serves as a hyperparameter of dirichlet prior distribuion over the prior weights of meta-trees.

        Returns
        ----------
        prob_vec : numpy ndarray
    '''
    smoothed_log_likelihoods = log_likelihoods * smoothness_param_vecs
    normalization_term = logsumexp(smoothed_log_likelihoods)
    return np.exp(smoothed_log_likelihoods - normalization_term), normalization_term
