# Code Writer : Ryota Maniwa 2023/8/14
# Code Writer : Keito Tajima 2023/11/17
# Code Writer : Naoki ICHIJO

import numpy as np
import math
from typing import Optional, Union, Generator, List
from .._exceptions import (CriteriaError, DataFormatError, ParameterFormatError,
                         ParameterFormatWarning, ResultWarning)

def feature_subsample_random(features_vec,num_features,seed):
    ''' The list of features is randomly selected.

    Parameters
	----------
	tuple_feature : numpy ndarray
        Each element of the list is a positive integer.
    num_features : int
        A positive integer
    seed : int
        A positive integer
    
	Returns
	-------
	feature_vec : numpy ndarray
        Each element of the list is a positive integer and the list is sorted in ascending.
    
    '''
    rng = np.random.default_rng(seed)
    indices = sorted(rng.choice(len(features_vec), size=num_features, replace=False))
    return indices

def _feature_subsample(feature_candidates_vec:list,max_features:Union[str,int,None],dim_features:int,rng:Generator):
    features_vec = list(set(feature_candidates_vec))
    if max_features is None:
        pass
    elif max_features == "sqrt":
        num_features = int(np.ceil(np.sqrt(dim_features)))
        features_vec = rng.choice(list(set(features_vec)),size=num_features,replace=False)
    elif max_features == "log2":
        num_features = int(np.ceil(math.log2(dim_features)))
        features_vec = rng.choice(list(set(features_vec)),size=num_features,replace=False)
    elif type(max_features) == int:
        if len(features_vec) <= max_features:
            pass
        else:
            features_vec = rng.choice(list(set(features_vec)),size=max_features,replace=False)
    else:
        raise(ParameterFormatError('max_features can only take None(all features), \'log2\', \'sqrt\', or an int number no bigger than dim_features.'))
    return sorted(features_vec)
# より厳密に書くなら，num_features > |set(feature_candidates_vec)|の場合分けが必要

def _feature_subsample_by_corrcoef_y(
    feature_candidates_vec: List[int],
    x_continuous_vecs: np.ndarray,
    x_categorical_vecs: np.ndarray,
    y_vec: np.ndarray,
    m: int # 取り出す特長量のインデックス数
):
    x_vecs = np.empty([y_vec.shape[0],x_continuous_vecs.shape[1]+x_categorical_vecs.shape[1]])
    x_vecs[:,:x_continuous_vecs.shape[1]] = x_continuous_vecs
    x_vecs[:,x_continuous_vecs.shape[1]:] = x_categorical_vecs

    feature_candidates_vec = list(set(feature_candidates_vec))
    
    r = np.array([np.abs(np.corrcoef(x_vecs[:,i],y_vec)[0][1]) for i in range(x_vecs.shape[1])])
    indices_sort = np.argsort(r)[::-1]
    return [index for index in indices_sort if index in feature_candidates_vec][:m]

def _feature_subsample_by_corrcoef_x(
    feature_candidates_vec: List[int],
    x_continuous_vecs: np.ndarray,
    x_categorical_vecs: np.ndarray,
    k_parent: int, # (直前の)親ノードのk
    m: int # 取り出す特長量のインデックス数
):
    x_vecs = np.empty([x_continuous_vecs.shape[0],x_continuous_vecs.shape[1]+x_categorical_vecs.shape[1]])
    x_vecs[:,:x_continuous_vecs.shape[1]] = x_continuous_vecs
    x_vecs[:,x_continuous_vecs.shape[1]:] = x_categorical_vecs

    feature_candidates_vec = list(set(feature_candidates_vec))

    r = np.array([np.corrcoef(x_vecs[:,k_parent],x_vecs[:,i])[0][1] for i in range(x_vecs.shape[1])])
    print(r)
    indices_sort = np.argsort(r)
    return [index for index in indices_sort if index in feature_candidates_vec][:m]

if __name__ == '__main__':
    # DIM_FEATURES = 9
    # a = [i for i in range(DIM_FEATURES-3)]
    # b = [1,1,2,3,4,4,4,5,5,5,6,7,8,9]

    # print(feature_subsample_random(b,3,None))
    rng = np.random.default_rng(0)
    DIM_FEATURES = 9
    a = [i for i in range(DIM_FEATURES-3)]
    b = [1,1,2,3,4,4,4,5,5,5,6,7,8,9]
    print(a)
    print(b)

    print(feature_subsample_random(b,3,None))

    # print(_feature_subsample(b,None,DIM_FEATURES,0))
    # print(f"log2K:{np.ceil(math.log2(DIM_FEATURES))}")
    # print(_feature_subsample(b,"log2",DIM_FEATURES,0))
    # print(f"sqrt:{np.ceil(np.sqrt(DIM_FEATURES))}")
    # print(_feature_subsample(b,"sqrt",DIM_FEATURES,0))
    # print(f"max_features:{5}")
    # print(_feature_subsample(b,5,DIM_FEATURES,0))
    NUM_DATA = 10
    DIM_CONTINUOUS = 3
    DIM_CATEGORICAL = 2

    np.random.seed(0)
    x_continuous = np.random.random((NUM_DATA,DIM_CONTINUOUS))
    x_categorical = np.random.randint(0,2,(NUM_DATA,DIM_CATEGORICAL))
    y = np.random.random(NUM_DATA)
    x_continuous[:,1] = 2*y # 相関が1の説明変数

    # 相関の絶対値: [0.32878093 1.         0.03077294 0.46942874 0.02022813]
    indices = _feature_subsample_by_corrcoef_y(
        feature_candidates_vec=[0,1,2,3,4],
        x_continuous_vecs=x_continuous,
        x_categorical_vecs=x_categorical,
        y_vec=y,
        m=3
    )
    print(indices) # [1,3,0]のはず

    indices = _feature_subsample_by_corrcoef_y(
        feature_candidates_vec=[0,2,3,4],
        x_continuous_vecs=x_continuous,
        x_categorical_vecs=x_categorical,
        y_vec=y,
        m=3
    )
    print(indices) # [3,0,2]のはず

    # 相関係数: [-0.32878093  1.         -0.03077294 0.46942874  0.02022813]
    indices = _feature_subsample_by_corrcoef_x(
        feature_candidates_vec=[0,1,2,3,4],
        x_continuous_vecs=x_continuous,
        x_categorical_vecs=x_categorical,
        k_parent=1,
        m=3
    )
    print(indices) # [0,2,4]のはず

    indices = _feature_subsample_by_corrcoef_x(
        feature_candidates_vec=[0,1,2,3],
        x_continuous_vecs=x_continuous,
        x_categorical_vecs=x_categorical,
        k_parent=1,
        m=3
    )
    print(indices) # [0,2,3]のはず
