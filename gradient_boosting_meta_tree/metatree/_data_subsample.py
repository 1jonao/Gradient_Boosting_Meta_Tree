# Code Writer : Ryota Maniwa 2023/8/15

import numpy as np
from .. import bernoulli, categorical, normal
from typing import Dict, List, Tuple, Union, Optional, Generator, Any
from numpy.typing import ArrayLike

CLF_MODELS = {
    bernoulli,
    categorical,
    }
REG_MODELS = {
    normal,
    }

def _data_subsample_bootstrap(
        y_vec: np.ndarray,
        num_sample: Optional[int],
        rng: np.random.Generator,
    ):
    '''N_sub*100% subsampling (Bootstrap sampling)

        Parameters
        ----------
        x_continuous_vecs : numpy ndarray, optional
            2 dimensional float array whose size is ``(sample_size,c_dim_continuous)``, 
            by default None.
        x_categorical_vecs : numpy ndarray, optional
            2 dimensional int array whose size is ``(sample_size,c_dim_categorical)``, 
            by default None. Each element x_categorical[i,j] must satisfy 
            0 <= x_categorical[i,j] < self.c_num_children_vec[self.c_dim_continuous+i].
        y_vec : numpy ndarray
            values of objective variable whose dtype may be int or float
        num_sample: float
            A positive integer
        rng : Generator
            A generator from np.random.default_rng()
            
        return
        ----------
        indices : numpy ndarray
            Sampled indices of data.
        '''
    if type(num_sample) is int:
        _num_sample = num_sample
    elif num_sample == None:
        _num_sample = y_vec.shape[0]
    indices = rng.choice(y_vec.shape[0],size=_num_sample,replace=True)
    return indices

def _data_subsample_goss(
        learn_model,
        x_continuous_vecs:ArrayLike,
        x_categorical_vecs:ArrayLike,
        y_vec:ArrayLike,
        y_pred_vec:ArrayLike,
        num_sample:int,
        top_rate:float,
        other_rate:float,
        rng:Generator,
        ):
    '''N_sub*100% subsampling (Bootstrap sampling)

        Parameters
        ----------
        x_continuous_vecs : numpy ndarray, optional
            2 dimensional float array whose size is ``(sample_size,c_dim_continuous)``, 
            by default None.
        x_categorical_vecs : numpy ndarray, optional
            2 dimensional int array whose size is ``(sample_size,c_dim_categorical)``, 
            by default None. Each element x_categorical[i,j] must satisfy 
            0 <= x_categorical[i,j] < self.c_num_children_vec[self.c_dim_continuous+i].
        y_vec : numpy ndarray
            values of objective variable whose dtype may be int or float
        y_pred_vec : numpy ndarray
            values of objective variable whose dtype may be int or float
        top_rate : float
            A real number in :math:`[0, 1]`
        other_rate : float
            A real number in :math:`[0, 1]`
        seed : int
            A positive integer
        
        return
        ----------
        x_continuous_vecs : numpy ndarray, optional
            2 dimensional float array whose size is ``(sample_size,c_dim_continuous)``, 
        x_categorical_vecs : numpy ndarray, optional
            2 dimensional int array whose size is ``(sample_size,c_dim_categorical)``, 
            by default None. Each element x_categorical[i,j] must satisfy 
            0 <= x_categorical[i,j] < self.c_num_children_vec[self.c_dim_continuous+i].
        y : numpy ndarray
            values of objective variable whose dtype may be int or float.
        '''
    if learn_model.SubModel in CLF_MODELS:
        if learn_model.criterion == 'logloss':
            gradient = -np.sum(np.eye(np.unique(y_vec).size)[y_vec]*np.log(y_pred_vec),axis=1)
    elif learn_model.SubModel in REG_MODELS:
        if learn_model.criterion == 'squared':
            gradient = (y_vec-y_pred_vec)**2

    c_dim_continuous = x_continuous.shape[1]
    c_dim_categorical = x_categorical.shape[1]
    c_dim_features = c_dim_continuous+c_dim_categorical
    data = np.empty([y.shape[0],c_dim_features+1])
    data[:,:c_dim_continuous] = x_continuous
    data[:,c_dim_continuous:c_dim_features] = x_categorical
    data[:,c_dim_features] = y

    indices = np.argsort(gradient)[::-1]
    top_indices = indices[:int(top_rate * y.shape[0])]
    top_data = data[top_indices]
    top_y_pred = y_pred_vec[top_indices]
    other_data = np.delete(data,top_indices,axis=0)
    other_y_pred = np.delete(y_pred_vec,top_indices,axis=0)
    random_indices = rng.choice(other_data.shape[0], size=int(other_data.shape[0] * other_rate), replace=False)
    random_other_data = other_data[random_indices]
    random_other_y_pred = other_y_pred[random_indices]
    select_data = np.vstack((top_data,random_other_data))
    if learn_model.SubModel in CLF_MODELS:
        if learn_model.criterion == 'logloss':
            select_y_pred = np.vstack((top_y_pred,random_other_y_pred))
    elif learn_model.SubModel in REG_MODELS:
        if learn_model.criterion == 'squared':
            select_y_pred = np.concatenate((top_y_pred,random_other_y_pred))
    
    return select_data[:,:c_dim_features], select_data[:,c_dim_features].astype(int), select_y_pred

# simpler function for residual algorithm
def _data_subsample_goss_residual(
        residual_vec:ArrayLike,
        top_rate:float,
        other_rate:float,
        rng:Generator,
        ):
    num_tops = int(top_rate * residual_vec.shape[0])
    sorted_indices = np.flipud(np.argsort(residual_vec))
    top_indices = sorted_indices[:num_tops]
    indices_to_sample = sorted_indices[num_tops:]
    num_others = int(other_rate * residual_vec.shape[0])
    random_indices = rng.choice(indices_to_sample, size=num_others, replace=False)
    indices_concat = np.concatenate((top_indices,random_indices))
    weight_others = (1-top_rate) / other_rate
    sample_weight = np.concatenate((np.ones(num_tops), np.full(num_others, weight_others)))
    return indices_concat, sample_weight

if __name__ == '__main__':
    print('debug')
    print('')
    
    rng = np.random.default_rng(0)
    datasize = 5
    samplesize = 3
    dim_continuous = 4
    dim_categorical = 5


    print('_data_subsample_bootstrap')
    x_continuous = rng.random(size=(datasize,dim_continuous))
    x_categorical = rng.integers(low=0, high=2, size=(datasize,dim_categorical))
    y = rng.random(size=(datasize))
    indices = _data_subsample_bootstrap(
        y_vec=y,
        num_sample=samplesize,
        rng=rng)
    print('x_select_continuous.shape:',x_continuous[indices].shape)
    print('x_select_categorical.shape:',x_categorical[indices].shape)
    print('y_select.shape:',y[indices].shape)
    print(x_continuous)
    print(x_categorical)
    print(y)
    print(x_continuous[indices])
    print(x_categorical[indices])
    print(y[indices])
    print('')

    print('_data_subsample_goss_residual')
    residuals = rng.random(size=(datasize))
    indices, sample_weight = _data_subsample_goss_residual(
        residual_vec=residuals,
        top_rate=0.2,
        other_rate=0.5,
        rng=rng)
    print('x_select_continuous.shape:',x_continuous[indices].shape)
    print('x_select_categorical.shape:',x_categorical[indices].shape)
    print('y_select.shape:',y[indices].shape)
    print(x_continuous)
    print(x_categorical)
    print(y)
    print(x_continuous[indices])
    print(x_categorical[indices])
    print(y[indices])
    print(residuals[indices])
    print(sample_weight)
    print('')
