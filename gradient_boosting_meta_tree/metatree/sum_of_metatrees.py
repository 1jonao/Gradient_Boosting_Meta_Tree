import sys, os

import numpy as np
from copy import deepcopy
from collections import deque, defaultdict
from scipy.special import logsumexp
from tqdm import tqdm
import re
import tempfile
import json

import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex

from typing import Dict, List, Tuple, Union, Optional, Any
from numpy.typing import ArrayLike
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier

from .. import _check
from .. import base
from .._exceptions import (CriteriaError, DataFormatError, ParameterFormatError,
                         ParameterFormatWarning, ResultWarning)

# from metatree_single_GenModel import MetaTreeGenModel
from .metatree_single_LearnModel import MetaTreeLearnModel # WARNING: 他のnormalとかのimportに則って.をつけたら動くが，取り除くとインポートできない．なぜかわからないので誰か教えて．
from ._metatree_weight import _weight_learning_rate, _proba_uniform, _proba_posterior, _proba_posterior_smooth_dirichletprior, _proba_posterior_smooth_exp_tilting
from ._feature_subsample import _feature_subsample
from ._data_subsample import _data_subsample_bootstrap, _data_subsample_goss_residual

from . import _constants
_CMAP = _constants._CMAP

MODELS = _constants.MODELS
DISCRETE_MODELS = _constants.DISCRETE_MODELS
CONTINUOUS_MODELS = _constants.CONTINUOUS_MODELS
CLF_MODELS = _constants.CLF_MODELS
REG_MODELS = _constants.REG_MODELS
    
def parse_json_tree(tree, path):
    """
    再帰的にJSON形式のツリーを探索し、文字列パスを割り当てる。
    
    Args:
        tree (dict): ノードの辞書。
        path (str): 現在のノードのパス。

    Returns:
        dict: 文字列パスをキーとして、ノード情報を値とする辞書。
    """
    tree_to_dict = {}
    feature_str = tree.get('split')
    if feature_str is not None:
        feature_int = int(feature_str[1:])
        threshold = float(tree.get('split_condition'))
    else:
        feature_int = None
        threshold = None
    node_info = {
        'nodeid': tree.get('nodeid'),
        'depth': tree.get('depth'),
        'feature': feature_int,
        'threshold': threshold,
        'leaf': tree.get('leaf'),
    }
    tree_to_dict[path] = node_info

    # 子ノードがある場合、再帰的に処理
    children = tree.get('children', [])
    if children:
        tree_to_dict.update(parse_json_tree(children[0], path + '0'))  # 左の子
        tree_to_dict.update(parse_json_tree(children[1], path + '1'))  # 右の子

    return tree_to_dict

def parse_xgb_json_model(file_path):
    """
    JSON形式のXGBoostモデルを読み込み、全てのツリーのノードに文字列パスを割り当てる。
    
    Args:
        file_path (str): JSONファイルのパス。

    Returns:
        dict: 各ブースターに対して、文字列パスをキーとする辞書の辞書。
    """
    with open(file_path, 'r') as f:
        model_data = json.load(f)

    result = {}
    for booster_id, tree in enumerate(model_data):
        result[booster_id] = parse_json_tree(tree, path='')

    return result

class SumOfMetaTreeLearnModel(base.Posterior,base.PredictiveMixin):
    def __init__(
        self,
        c_num_metatrees: int,
        SubModel: Any, # TODO サブモデルクラスでの型アノテーション
        c_dim_continuous: int,
        c_dim_categorical: int,
        c_max_depth: int, # TODO c_max_depth=0 causes error during broadcasting self.h0_split = np.ones(self.c_max_depth,dtype=float), it creates empty array.
        c_num_children_vec: List or int = 2,
    c_num_assignment_vec: Optional[np.ndarray] = None,
        h0_constants_SubModel: Dict = {},
        sub_h0_params: Dict = {},
        h0_split: List or float = 0.5,
    h0_feature_weight_vec: Optional[np.ndarray] = None,
        threshold_params: Optional[Dict] = None,
        data_subsample_params: Optional[Dict] = None,
        h0_metatree_list:Optional[List[MetaTreeLearnModel]]=None,
        metatree_weight_type_build:str='learning_rate',
        metatree_weight_build_is_compress:bool=False,
        metatree_weight_build_newtree:Union[str,float]='num_tree',
        metatree_weight_type_pred:str='learning_rate',
        learning_rate:float=0.1,
    h0_metatree_weight_smoothness: Union[np.ndarray, float] = 0.,
        feature_subsample:Optional[str]=None,
        target_function:Optional[str]=None,
        tree_criterion:str='squared_error_leaf', # NOTE: sklearnではfriedman_mseがあるが，まだsquared_errorのみで進める．
        # FIXME: target_functionとtree_criterionで'squared_error'を共有していてややこしい．
        lambda_xgb:float = 1.,
        gamma_xgb:float = 0.,
        seed: Optional[int] = None,
    ):
        # Static valuables which should not be updated, from the input
        self.c_num_metatrees = _check.pos_int(c_num_metatrees,'c_num_metatrees',ParameterFormatError)
        self.c_dim_continuous = _check.nonneg_int(c_dim_continuous,'c_dim_continuous',ParameterFormatError)
        self.c_dim_categorical = _check.nonneg_int(c_dim_categorical,'c_dim_categorical',ParameterFormatError)
        self.c_dim_features = _check.pos_int(c_dim_continuous+c_dim_categorical,'c_dim_continuous+c_dim_categorical',ParameterFormatError)
        self.c_feature_candidates = list(range(self.c_dim_features))
        self.c_max_depth = _check.nonneg_int(c_max_depth,'c_max_depth',ParameterFormatError)

        _check.pos_ints(c_num_children_vec,'c_num_children_vec',ParameterFormatError)
        if np.any(c_num_children_vec<2):
            raise(ParameterFormatError(
                'All the elements of c_num_children_vec must be greater than or equal to 2: '
                +f'c_num_children_vec={c_num_children_vec}.'
            ))
        self.c_num_children_vec = np.ones(self.c_dim_features,dtype=int)*2
        self.c_num_children_vec[:] = c_num_children_vec

        self.c_num_assignment_vec = np.ones(self.c_dim_features,dtype=int)
        self.c_num_assignment_vec[:self.c_dim_continuous] *= self.c_max_depth
        if c_num_assignment_vec is not None:
            _check.nonneg_ints(c_num_assignment_vec,'c_num_assignment_vec',ParameterFormatError)
            if np.any(c_num_assignment_vec>self.c_max_depth):
                raise(ParameterFormatError(
                    'All the elements of c_num_assignment_vec must be less than or equal to self.c_max_depth: '
                    +f'c_num_assignment_vec={c_num_assignment_vec}.'
                ))
            self.c_num_assignment_vec[:] = c_num_assignment_vec

        if type(h0_split) == float:
            self.h0_split = _check.float_in_closed01(h0_split,'h0_split',ParameterFormatError)
            self.h0_split_list = np.full(self.c_max_depth, h0_split, dtype=float)
        else:
            self.h0_split_list = _check.float_in_closed01_vec(h0_split,'h0_split',ParameterFormatError)

        # valuables which can be updated, from the input
        if SubModel not in MODELS:
            raise(ParameterFormatError(
                "SubModel must be "
                +"normal"
                # +"bernoulli, poisson, normal, exponential, linearregression."
            ))
        self.SubModel = SubModel
        self.sub_h0_params = self.SubModel.LearnModel(**sub_h0_params).get_h0_params()
        self.h0_constants_SubModel = self.SubModel.LearnModel(**h0_constants_SubModel).get_constants()

        if h0_feature_weight_vec is None:
            self.h0_feature_weight_vec = np.full(self.c_dim_features, 1/self.c_dim_features)
        else:
            self.h0_feature_weight_vec = _check.proba_vec(h0_feature_weight_vec,'h0_feature_weight_vec',ParameterFormatError)

        # thresholds
        self.threshold_params = threshold_params # TODO: dictのParameterFormat判定

        # data_subsample
        if data_subsample_params is None:
            self.data_subsample_params = {'type':'all'}
        else:
            self.data_subsample_params = data_subsample_params

        if h0_metatree_list is None: # FIXME: []を入れるとそのまま入力した空配列が使われるが，メモリ共有の問題が起こる．
            self.h0_metatree_list = []
        else:
            self.h0_metatree_list = h0_metatree_list
        
        self.metatree_weight_type_build = metatree_weight_type_build
        self.metatree_weight_build_is_compress = metatree_weight_build_is_compress
        if metatree_weight_build_newtree == 'num_tree':
            self.metatree_weight_build_newtree = metatree_weight_build_newtree
        else:
            self.metatree_weight_build_newtree = _check.float_in_closed01(metatree_weight_build_newtree,'metatree_weight_build_newtree_constant',ParameterFormatError)
        self.metatree_weight_type_pred = metatree_weight_type_pred
        self.learning_rate = _check.nonneg_float(learning_rate,'learning_rate',ParameterFormatError)
        if type(h0_metatree_weight_smoothness) == float:
            self.h0_metatree_weight_smoothness = _check.nonneg_float(h0_metatree_weight_smoothness,'h0_metatree_weight_smoothness',ParameterFormatError)
            self.h0_metatree_weight_smoothness_vec = np.full(self.c_num_metatrees, h0_metatree_weight_smoothness, dtype=float)
        elif type(h0_metatree_weight_smoothness) == int:
            self.h0_metatree_weight_smoothness = _check.nonneg_int(h0_metatree_weight_smoothness,'h0_metatree_weight_smoothness',ParameterFormatError)
            self.h0_metatree_weight_smoothness_vec = np.full(self.c_num_metatrees, h0_metatree_weight_smoothness, dtype=float)
        else:
            self.h0_metatree_weight_smoothness_vec = _check.nonneg_float_vec(h0_metatree_weight_smoothness,'h0_metatree_weight_smoothness',ParameterFormatError)

        self.feature_subsample = feature_subsample
        if target_function is None:
            if self.SubModel in REG_MODELS:
                self.target_function = 'squared_error'
            elif self.SubModel in CLF_MODELS:
                self.target_function = 'log_loss'
        else:
            self.target_function = target_function
        
        if tree_criterion == 'squared_error_leaf':
            pass
        elif tree_criterion == 'xgb_gain_leaf':
            pass
        else:
            raise(ParameterFormatError,'tree_criterion=\'squared_error_leaf\', \'xgb_gain_leaf\' are only supported in this moment.')
        self.tree_criterion = tree_criterion

        self.lambda_xgb = _check.nonneg_float(lambda_xgb,'lambda_xgb',ParameterFormatError)
        self.gamma_xgb = _check.nonneg_float(gamma_xgb,'gamma_xgb',ParameterFormatError)

        if seed is None:
            self.seed = seed
        else:
            self.seed = _check.int_(seed, 'seed', ParameterFormatError)
        self.rng = np.random.default_rng(self.seed)
        return
    
    def _check_sample_x(
            self, 
            x_continuous: np.ndarray,
            x_categorical: np.ndarray,
            ):
        if self.c_dim_continuous > 0 and self.c_dim_categorical > 0:
            _check.float_vecs(x_continuous,'x_continuous',DataFormatError)
            _check.shape_consistency(
                x_continuous.shape[-1],'x_continuous.shape[-1]',
                self.c_dim_continuous,'self.c_dim_continuous',
                ParameterFormatError
                )
            x_continuous = x_continuous.reshape([-1,self.c_dim_continuous])
            _check.nonneg_int_vecs(x_categorical,'x_categorical',DataFormatError) # NOTE: allow bool type input ?
            _check.shape_consistency(
                x_categorical.shape[-1],'x_categorical.shape[-1]',
                self.c_dim_categorical,'self.c_dim_categorical',
                ParameterFormatError
                )
            x_categorical = x_categorical.reshape([-1,self.c_dim_categorical])
            for i in range(self.c_dim_categorical):
                if x_categorical[:,i].max() >= self.c_num_children_vec[self.c_dim_continuous+i]:
                    raise(DataFormatError(
                        f"x_categorical[:,{i}].max() must smaller than "
                        +f"self.c_num_children_vec[{self.c_dim_continuous+i}]: "
                        +f"{self.c_num_children_vec[self.c_dim_continuous+i]}"))
            _check.shape_consistency(
                x_continuous.shape[0],'x_continuous.shape[0]',
                x_categorical.shape[0],'x_categorical.shape[0]',
                ParameterFormatError
                )

        elif self.c_dim_continuous > 0:
            _check.float_vecs(x_continuous,'x_continuous',DataFormatError)
            _check.shape_consistency(
                x_continuous.shape[-1],'x_continuous.shape[-1]',
                self.c_dim_continuous,'self.c_dim_continuous',
                ParameterFormatError
                )
            x_continuous = x_continuous.reshape([-1,self.c_dim_continuous])
            x_categorical = np.empty([x_continuous.shape[0],0]) # dummy

        elif self.c_dim_categorical > 0:
            _check.nonneg_int_vecs(x_categorical,'x_categorical',DataFormatError)
            _check.shape_consistency(
                x_categorical.shape[-1],'x_categorical.shape[-1]',
                self.c_dim_categorical,'self.c_dim_categorical',
                ParameterFormatError
                )
            x_categorical = x_categorical.reshape([-1,self.c_dim_categorical])
            for i in range(self.c_dim_categorical):
                if x_categorical[:,i].max() >= self.c_num_children_vec[self.c_dim_continuous+i]:
                    raise(DataFormatError(
                        f"x_categorical[:,{i}].max() must smaller than "
                        +f"self.c_num_children_vec[{self.c_dim_continuous+i}]: "
                        +f"{self.c_num_children_vec[self.c_dim_continuous+i]}"))
            x_continuous = np.empty([x_categorical.shape[0],0]) # dummy

        return x_continuous,x_categorical
    def _check_sample_y(self,x_continuous,y):
        if self.SubModel is linearregression:
            self.SubModel.LearnModel(**self.h0_constants_SubModel)._check_sample(x_continuous,y)
        else:
            self.SubModel.LearnModel(**self.h0_constants_SubModel)._check_sample(y)
        return np.ravel(y)
    def _check_sample(self,x_continuous,x_categorical,y):
        x_continuous, x_categorical = self._check_sample_x(x_continuous,x_categorical)
        y = self._check_sample_y(x_continuous,y)
        _check.shape_consistency(
            x_continuous.shape[0],'x_continuous.shape[0] and x_categorical.shape[0]',
            y.shape[0],'y.shape[0]',
            ParameterFormatError
            )
        return x_continuous,x_categorical,y

    ##########################
    # Required for LearnModel
    ##########################
    def calc_pred_dist(self):
        for tree in self.hn_metatree_list:
            tree.calc_pred_dist()
        self.metatree_weights_pred = self.calc_weight_pred()
        return
    def estimate_params(self):
        return
    def get_h0_params(self):
        return
    def get_hn_params(self):
        return
    def get_p_params(self):
        return
    def make_prediction(
        self,
    x_continuous_new: np.ndarray, 
    x_categorical_new: np.ndarray,
        loss:str='squared', 
        ):
        x_continuous_new, x_categorical_new = self._check_sample_x(x_continuous_new, x_categorical_new)
        self.calc_pred_dist()
        if loss in ['squared']:
            hat_y_vec = sum(map(lambda i: self.metatree_weights_pred[i] * self.hn_metatree_list[i].make_prediction(x_continuous_new, x_categorical_new, loss=loss), range(self.c_num_metatrees)))
        elif loss in ['0-1']:
            hat_theta_vec = sum(map(lambda i: self.metatree_weights_pred[i] * self.hn_metatree_list[i].make_prediction(x_continuous_new, x_categorical_new, loss='squared'), range(self.c_num_metatrees)))
            hat_y_vec = np.where(hat_theta_vec > 0.5, 1, 0)
        else:
            raise(ParameterFormatError,'loss=\'squared\', \'0-1\' is only supported in this moment.')
        if self.centerize_y:
            hat_y_vec += self.y_mean
        return hat_y_vec
    def calc_weight_pred(self)->np.ndarray:
        if self.metatree_weight_type_pred == 'learning_rate':
            return _weight_learning_rate(self.c_num_metatrees, self.learning_rate)
        elif self.metatree_weight_type_pred == 'proba_uniform':
            self.update_trees_from_y()
            return _proba_uniform(self.c_num_metatrees)
        elif self.metatree_weight_type_pred == 'proba_posterior':
            self.update_trees_from_y()
            metatree_log_marginal_likelihoods = np.array(self.calc_metatree_log_marginal_likelihoods())
            posterior, _ = _proba_posterior(metatree_log_marginal_likelihoods)
            return posterior
        elif self.metatree_weight_type_pred == 'proba_posterior_smooth_dirichletprior':
            self.update_trees_from_y()
            metatree_log_marginal_likelihoods = np.array(self.calc_metatree_log_marginal_likelihoods())
            smoothed_posterior, _ = _proba_posterior_smooth_dirichletprior(metatree_log_marginal_likelihoods, self.h0_metatree_weight_smoothness_vec)
            return smoothed_posterior
        elif self.metatree_weight_type_pred == 'proba_posterior_smooth_exp_tilting':
            self.update_trees_from_y()
            metatree_log_marginal_likelihoods = np.array(self.calc_metatree_log_marginal_likelihoods())
            smoothed_posterior, _ = _proba_posterior_smooth_exp_tilting(metatree_log_marginal_likelihoods, self.h0_metatree_weight_smoothness_vec)
            return smoothed_posterior
        else:
            raise(ParameterFormatError,'metatree_weight_type_pred=\'learning_rate\', \'proba_uniform\', \'proba_posterior\', \'proba_posterior_smooth_dirichletprior\', \'proba_posterior_smooth_exp_tilting\' is only supported in this moment.')
    def calc_metatree_log_marginal_likelihoods(self)->List[float]:
        return list(map(
            lambda tree: tree.calc_tree_log_marginal_likelihood(),
            self.hn_metatree_list
        ))
    def pred_and_update(self):
        return
    def set_h0_params(self,
        h0_feature_weight_vec = None,
        h0_split=None,
        sub_h0_params=None,
        ):
        """Set the hyperparameters of the prior distribution.

        Parameters
        ----------
        h0_k_weight_vec : numpy.ndarray, optional
            A vector of real numbers in :math:`[0, 1]`, 
            by default None
            Sum of its elements must be 1.
        h0_g : float, optional
            A real number in :math:`[0, 1]`, by default None
        sub_h0_params : dict, optional
            h0_params for self.SubModel.LearnModel, by default None
        """
        return
    def set_hn_params(self):
        return
    def update_posterior(self):
        return
    def visualize_posterior(self):
        return
    ##########################

    ##########################
    # visualization
    ##########################
    def visualize_model(self,num_tree,filename=None,dirname=None,format=None,view=False,save=False,hn_params=True,h_params=False,p_params=False):
        if num_tree > self.c_num_metatrees:
            raise(ParameterFormatError,'num_tree must be smaller than or equal to self.c_num_metatrees.')
        try:
            import graphviz
            tree = self.hn_metatree_list[num_tree]
            filename = f'tree_{num_tree}' if filename is None else filename
            tree_graph, root_node_label = tree.visualize_model(filename=None,format=format,view=False,hn_params=hn_params,h_params=h_params,p_params=p_params)
            root_node_label = root_node_label[:]
            root_node_label = f'predictive_weight={self.metatree_weights_pred[num_tree]}\\l\\l' + root_node_label
            tree_graph.node(name='',label=root_node_label,fillcolor=f'{rgb2hex(_CMAP(1-tree.root_node.hn_split))}')
            if view:
                tree_graph.view()
            else:
                pass
            if save:
                os.makedirs(dirname,exist_ok=True) # TODO: enable absolute path option for dirnam
                tree_graph.render(os.path.join(dirname,filename),format=format,cleanup=True)
            else:
                pass
            return tree_graph
        except ImportError as e:
            print(e)
        except graphviz.CalledProcessError as e:
            print(e)
        return tree_graph

    def build_metatrees(
        self,
    x_continuous_vecs: np.ndarray,
    x_categorical_vecs: np.ndarray,
    y_vec: np.ndarray,
        split_strategy: str = 'best', # 'best', 'given_rule', 'copy_sklearn_ensemble', 'copy_xgb'
        building_scheme: str = 'depth_first',
        centerize_y: bool = False,
        calc_residual: bool = True,
        progress_bar: bool = False,
        max_leaf_nodes: Optional[int] = None,
        SklearnEnsembleObj: Optional[BaseEstimator] = None,
        params_sklearn_ensemble: dict = {},
        update_leaf_with_residual: bool = False, # True is supported only when split_strategy='copy_sklearn_ensemble' and GradinetBoostingRegressor is used for SklearnEnsembleObj.
        XGBRegressorObj: Optional[BaseEstimator] = None,
        params_xgb: dict = {},
        given_split_rules_list: Optional[List[Dict]] = None,
    ):
        # input data
        self.x_continuous_vecs, self.x_categorical_vecs, self.y_vec = self._check_sample(x_continuous_vecs,x_categorical_vecs,y_vec)
        self.centerize_y = centerize_y
        if self.centerize_y:
            self.y_mean = self.y_vec.mean()
            self.y_vec = self.y_vec - self.y_mean
        if bool(given_split_rules_list) + bool(SklearnEnsembleObj) + bool(calc_residual) > 1:
            raise(ParameterFormatError,'given_split_rules_list, SklearnEnsembleObj, calc_residual are exclusive to each other.')
        if split_strategy == 'given_rule':
            if given_split_rules_list is None:
                raise(ParameterFormatError,'split_strategy=\'given_rule\' requires the input of given_split_rules_list.')
            else:
                training_error_list = self._build_metatrees_given_split_rules(given_split_rules_list, progress_bar=progress_bar)
                return training_error_list
        elif split_strategy == 'copy_sklearn_ensemble':
            if SklearnEnsembleObj is None:
                raise(ParameterFormatError,'split_strategy=\'copy_sklearn_ensemble\' requires the input of SklearnEnsembleObj.')
            training_error_list, self.fitted_sklearn_estimator = self._copy_trees_from_sklearn_ensemble(SklearnEnsembleObj, params_sklearn_ensemble, update_leaf_with_residual)
            return training_error_list
        elif split_strategy == 'copy_xgb':
            if XGBRegressorObj is None:
                raise(ParameterFormatError,'split_strategy=\'copy_xgb\' requires the input of XGBRegressor object.')
            training_error_list, self.fitted_sklearn_estimator = self._copy_trees_from_xgb(XGBRegressorObj, params_xgb)
            return training_error_list
        elif split_strategy == 'best':
            if calc_residual:
                training_error_list = self._build_metatrees_by_residual(split_strategy=split_strategy,building_scheme=building_scheme,progress_bar=progress_bar,max_leaf_nodes=max_leaf_nodes)
                return training_error_list
            else:
                raise(ParameterFormatError, 'In this moment, split_strategy=\'best\' requires calc_residual=True.')
        else:
            raise(ParameterFormatError,'split_strategy=\'best\', \'given_rule\', \'copy_sklearn_ensemble\' are only supported in this moment.')

    def _build_metatrees_given_split_rules(
        self,
        given_split_rules_list: List[Dict],
        progress_bar: bool = False,
    ):
        training_error_list = []
        self.pred_new_tree_list = []
        # calculate residual from given metatrees
        num_trees_built = len(self.h0_metatree_list)
        if num_trees_built > 0:
            raise(ParameterFormatError, 'hot-start(initializing with built meta-trees) is currently not enabled.')
        if len(given_split_rules_list) != self.c_num_metatrees:
            raise(ParameterFormatError,'The length of given_split_rules_list must be equal to self.c_num_metatrees.')
        # initiate metatree_list
        self.hn_metatree_list = [None for _ in range(self.c_num_metatrees)]
        iterator = range(num_trees_built, self.c_num_metatrees)
        for i in tqdm(iterator, leave=False, desc='constructing metatrees...') if progress_bar else iterator:
            # data_subsample
            if self.data_subsample_params['type'] == 'all':
                tmp_indices = np.arange(self.y_vec.shape[0])
                tmp_sample_weight = None
            elif self.data_subsample_params['type'] == 'bootstrap':
                tmp_indices = _data_subsample_bootstrap(self.y_vec,self.data_subsample_params['num_sample'],self.rng)
                tmp_sample_weight = None
            else:
                raise(ParameterFormatError,'data_subsample[\'type\'] is only supported for \'all\' or \'bootstrap\' are only supported when split_strategy=\'given_rule\'.')
            
            self.hn_metatree_list[i] = MetaTreeLearnModel(
                SubModel=self.SubModel,
                c_dim_continuous = self.c_dim_continuous,
                c_dim_categorical = self.c_dim_categorical,
                c_max_depth = self.c_max_depth,
                c_num_children_vec = self.c_num_children_vec,
                c_feature_candidates = self.c_feature_candidates,
                h0_constants_SubModel = self.h0_constants_SubModel,
                sub_h0_params = self.sub_h0_params,
                h0_split = self.h0_split_list,
                h0_feature_weight_vec = self.h0_feature_weight_vec,
                threshold_params = self.threshold_params,
                lambda_xgb = self.lambda_xgb,
                gamma_xgb = self.gamma_xgb,
                seed = self.seed, # TODO: rngをそのまま渡したい．
            )
            self.hn_metatree_list[i].build(
                self.x_continuous_vecs[tmp_indices],
                self.x_categorical_vecs[tmp_indices],
                self.y_vec[tmp_indices],
                'given_rule',
                self.tree_criterion,
                tmp_sample_weight,
                'depth_first',
                ignore_check=True,
                max_features = self.feature_subsample,
                max_leaf_nodes = None,
                sklearn_tree = None,
                given_split_rules = given_split_rules_list[i],
                )
            self.hn_metatree_list[i].calc_pred_dist()
            # if self.target_function == 'squared_error':
            #     loss = 'squared'
            # elif self.target_function == 'reg_xgb':
            #     loss = 'reg_xgb'
            # else:
            #     raise(ParameterFormatError,'target_function=\'squared_error\', \'reg_xgb\' is only supported in this moment.')
            # if self.metatree_weight_type_build.startswith('proba'):
            #     self.hn_metatree_list[i]._update_posterior_all_nodes_given(
            #         self.x_continuous_vecs,
            #         self.x_categorical_vecs,
            #         self.y_vec,
            #         reset_param=True,
            #     )
            # pred_built, residual_vec, training_error = self.update_residual(residual_vec,pred_built,num_trees_built=i+1,loss=loss)
            # training_error_list.append(training_error)
            
        return training_error_list

        

    def _build_metatrees_by_residual(
        self,
        split_strategy: str = 'best',
        building_scheme: str = 'depth_first',
        progress_bar: bool = False,
        max_leaf_nodes: Optional[int] = None,
    ):
        training_error_list = []
        self.pred_new_tree_list = []
        # calculate residual from given metatrees
        num_trees_built = len(self.h0_metatree_list)
        if num_trees_built > 0:
            raise(ParameterFormatError, 'hot-start(initializing with built meta-trees) is currently not enabled.')
        # initiate weight vector
        self.hn_metatree_list = []
        self._init_weight(num_trees_built)
        pred_built = self.calc_pred_from_metatree_list(num_trees_built)
        if self.SubModel in REG_MODELS:
            residual_vec = self.y_vec - pred_built # WARNING: たぶん回帰;二乗誤差のみでしか動かない．
        else:
            raise(ParameterFormatError,'Regression models are only supported at this moment.')
        
        # copy metatrees from h0_metatree_list
        self.hn_metatree_list = [None for _ in range(self.c_num_metatrees)]
        for i, metatree in enumerate(self.h0_metatree_list):
            self.hn_metatree_list[i] = metatree

        # sequentially constructing metatrees
        iterator = range(num_trees_built, self.c_num_metatrees)
        for i in tqdm(iterator, leave=False, desc='constructing metatrees...') if progress_bar else iterator:
            # data_subsample
            if self.data_subsample_params['type'] == 'all':
                tmp_indices = np.arange(self.y_vec.shape[0])
                tmp_sample_weight = None
            elif self.data_subsample_params['type'] == 'bootstrap':
                tmp_indices = _data_subsample_bootstrap(self.y_vec,self.data_subsample_params['num_sample'],self.rng)
                tmp_sample_weight = None
            elif self.data_subsample_params['type'] == 'goss':
                tmp_indices, tmp_sample_weight = _data_subsample_goss_residual(residual_vec,self.data_subsample_params['top_rate'],self.data_subsample_params['other_rate'],self.rng)
            else:
                raise(ParameterFormatError,'data_subsample[\'type\'] is only supported for \'all\', \'goss\' or \'bootstrap\' are only supported in this moment.')
            # feature_subsample
            # if self.feature_subsample == 'all_features':
            #     tmp_feature_candidates = self.c_feature_candidates
            # elif (self.feature_subsample in ['log2', 'sqrt']) or (type(self.feature_subsample) == int):
            #     tmp_feature_candidates = _feature_subsample(self.c_feature_candidates, self.feature_subsample, self.c_dim_features, self.rng)
                
            # else:
            #     raise(ParameterFormatError,'feature_subsample=\'all_features\', \'log2\', \'sqrt\', or an int number are only supported in this moment.')
            self.hn_metatree_list[i] = MetaTreeLearnModel(
                SubModel=self.SubModel,
                c_dim_continuous = self.c_dim_continuous,
                c_dim_categorical = self.c_dim_categorical,
                c_max_depth = self.c_max_depth,
                c_num_children_vec = self.c_num_children_vec,
                c_feature_candidates = self.c_feature_candidates,
                h0_constants_SubModel = self.h0_constants_SubModel,
                sub_h0_params = self.sub_h0_params,
                h0_split = self.h0_split_list,
                h0_feature_weight_vec = self.h0_feature_weight_vec,
                threshold_params = self.threshold_params,
                lambda_xgb = self.lambda_xgb,
                gamma_xgb = self.gamma_xgb,
                seed = self.seed, # TODO: rngをそのまま渡したい．
            )
            self.hn_metatree_list[i].build(
                self.x_continuous_vecs[tmp_indices],
                self.x_categorical_vecs[tmp_indices],
                residual_vec[tmp_indices],
                split_strategy,
                self.tree_criterion,
                tmp_sample_weight,
                building_scheme,
                ignore_check=True,
                max_features = self.feature_subsample,
                max_leaf_nodes = max_leaf_nodes,
                )
            self.hn_metatree_list[i].calc_pred_dist()
            if self.target_function == 'squared_error':
                loss = 'squared'
            elif self.target_function == 'reg_xgb':
                loss = 'reg_xgb'
            else:
                raise(ParameterFormatError,'target_function=\'squared_error\', \'reg_xgb\' is only supported in this moment.')
            if self.metatree_weight_type_build.startswith('proba'):
                self.hn_metatree_list[i]._update_posterior_all_nodes_given(
                    self.x_continuous_vecs,
                    self.x_categorical_vecs,
                    self.y_vec,
                    reset_param=True,
                )
            pred_built, residual_vec, training_error = self.update_residual(residual_vec,pred_built,num_trees_built=i+1,loss=loss)
            training_error_list.append(training_error)
            
        return training_error_list
    
    def update_trees_from_y(
            self,
    ) -> None:
        """
        残渣に対して学習した木でのサブモデルと分岐確率事後分布をyに対して計算し直す．
        単一決定木モデルの出力へのメソッド．
        """
        self.hn_metatree_list = list(map(
            lambda tree: tree._update_posterior_all_nodes_given(
                self.x_continuous_vecs,
                self.x_categorical_vecs,
                self.y_vec,
                reset_param=True,
                ),
            self.hn_metatree_list
        ))
        return
    def _init_weight(
            self,
            num_trees_built: int,
    )->None:
        if self.metatree_weight_type_build == 'learning_rate':
            self.metatree_weights_build = _weight_learning_rate(self.c_num_metatrees, self.learning_rate)
        elif self.metatree_weight_type_build == 'proba_uniform':
            pass
        elif self.metatree_weight_type_build.startswith('proba_posterior'):
            self.metatree_log_marginal_likelihoods = [None for _ in range(self.c_num_metatrees)]
            if num_trees_built == 0:
                self.posterior_normalization_term = -np.infty
            else:
                self.metatree_log_marginal_likelihoods[:self.c_num_metatrees] = self.calc_metatree_log_marginal_likelihoods()
                self.posterior_normalization_term = logsumexp(self.metatree_log_marginal_likelihoods)
        return
    def calc_pred_from_metatree_list(
            self,
            num_trees_built: int,
    ) -> ArrayLike:
        if self.SubModel in REG_MODELS:
            pred_metatree_list = [np.zeros((self.y_vec.shape[0])) for _ in range(self.c_num_metatrees)]
        else:
            raise(ParameterFormatError,'Regression models are only supported at this moment.')
        pred_metatree_list[:num_trees_built] = list(map(lambda i: self.hn_metatree_list[i].make_prediction(self.x_continuous_vecs, self.x_categorical_vecs, loss='squared'), range(num_trees_built)))
        if num_trees_built == 0:
            pred_built = np.zeros(self.y_vec.shape[0])
        else:
            pred_built = np.sum(self.metatree_weights_build[:num_trees_built].reshape(num_trees_built,1) * pred_metatree_list[:num_trees_built])
        return pred_built
    
    def update_pred_proba(
            self,
            pred_built: np.ndarray,
            pred_new_tree: np.ndarray,
            metatree_weight_type_build: str,
            num_trees_built: int,
    ) -> np.ndarray:
        if metatree_weight_type_build == 'proba_uniform':
            pred_new = pred_built*(num_trees_built-1)/(num_trees_built) + pred_new_tree/(num_trees_built)
        elif metatree_weight_type_build.startswith('proba_posterior'):
            newtree_log_likelihood = self.hn_metatree_list[num_trees_built-1].calc_tree_log_marginal_likelihood()
            if metatree_weight_type_build == 'proba_posterior':
                self.metatree_log_marginal_likelihoods[num_trees_built-1] = newtree_log_likelihood
            elif metatree_weight_type_build == 'proba_posterior_smooth_dirichletprior':
                self.metatree_log_marginal_likelihoods[num_trees_built-1] = np.logaddexp(newtree_log_likelihood,np.log(self.h0_metatree_weight_smoothness_vec[num_trees_built-1])) # NOTE: ln(w+a) = ln(exp(ln(w)) + a)
            elif metatree_weight_type_build == 'proba_posterior_smooth_exp_tilting':
                self.metatree_log_marginal_likelihoods[num_trees_built-1] =  newtree_log_likelihood * self.h0_metatree_weight_smoothness_vec[num_trees_built-1] # NOTE: ln(w^a) = a*ln(w)
            posterior_normalization_term_new = np.logaddexp(self.posterior_normalization_term, self.metatree_log_marginal_likelihoods[num_trees_built-1])
            tmp1 = np.exp(self.posterior_normalization_term - posterior_normalization_term_new)
            tmp2 = np.exp(self.metatree_log_marginal_likelihoods[num_trees_built-1] - posterior_normalization_term_new)
            pred_new = (pred_built * tmp1) + (pred_new_tree * tmp2)
            self.posterior_normalization_term = posterior_normalization_term_new
        return pred_new
    def update_residual(
            self,
            residual_vec: np.ndarray,
            pred_built: np.ndarray,
            num_trees_built: int,
            loss: str,
    ) -> np.ndarray:
        # WARNING: yに対しての予測を行うべきでは？
        pred_new_tree = self.hn_metatree_list[num_trees_built-1].make_prediction(self.x_continuous_vecs,self.x_categorical_vecs,loss=loss)
        self.pred_new_tree_list.append(pred_new_tree)
        # pred_built_debug = np.array([np.sum([self.pred_new_tree_list[b][i] for b in range(num_trees_built)])/(num_trees_built+1) for i in range(self.y_vec.shape[0])])
        if self.metatree_weight_type_build == 'learning_rate':
            residual_vec -= self.metatree_weights_build[num_trees_built-1] * pred_new_tree
            training_error = (residual_vec**2).mean()
        elif self.metatree_weight_type_build.startswith('proba'): # NOTE: 'proba' in __のほうが速いが，前方一致のほうがバグの混入少ないと判断．
            pred_built = self.update_pred_proba(pred_built, pred_new_tree, self.metatree_weight_type_build, num_trees_built=num_trees_built)
            if self.metatree_weight_build_is_compress: # TODO: is_compressまわりは議論が必要
                training_error = ((self.y_vec - pred_built)**2).mean()
                if self.metatree_weight_build_newtree == 'num_tree':
                    pred_built_compressed = pred_built * (num_trees_built) / (num_trees_built+1)
                    residual_vec = (self.y_vec - pred_built_compressed) * (num_trees_built+1) # NOTE: もともとのyに近い大きさになるように分母を掛け直す．特に議論が必要．
                    # residual_vec = (self.y_vec - pred_built)
                else:
                    pred_built_compressed = pred_built * (1 - self.metatree_weight_build_newtree)
                    residual_vec = (self.y_vec - pred_built_compressed) / self.metatree_weight_build_newtree # NOTE: もともとのyに近い大きさになるように分母を掛け直す．特に議論が必要．
                    # residual_vec = (self.y_vec - pred_built)
            else:
                residual_vec = (self.y_vec - pred_built) * (num_trees_built)
                training_error = ((self.y_vec - pred_built)**2).mean()
        else:
            raise(ParameterFormatError('metatree_weight_type_build=\'learning_rate\', \'proba_uniform\', \'proba_posterior\', \'proba_posterior_smooth_dirichletprior\' are only supported in this moment.'))
        # print(np.isclose(pred_built_debug,pred_built).all())
        return pred_built, residual_vec, training_error

    def _copy_trees_from_sklearn_ensemble(
        self,
        SklearnEnsembleObj: BaseEstimator,
        params_sklearn_ensemble: dict,
        update_leaf_with_residual: bool = False,
    ):

        # calculate residual from given metatrees
        num_trees_built = len(self.h0_metatree_list)
        if num_trees_built > 0:
            raise(ParameterFormatError, 'hot-start(initializing with built meta-trees) is currently not enabled.')
        # initiate weight vector
        self.hn_metatree_list = []
        self._init_weight(num_trees_built)
        
        # copy metatrees from h0_metatree_list
        self.hn_metatree_list = [None for _ in range(self.c_num_metatrees)]
        for i, metatree in enumerate(self.h0_metatree_list):
            self.hn_metatree_list[i] = metatree

        # build sklearn-trees
        self.x_vecs = np.concatenate([self.x_continuous_vecs, self.x_categorical_vecs],axis=1)
        sklearn_estimator = SklearnEnsembleObj(**params_sklearn_ensemble)
        sklearn_estimator.fit(self.x_vecs,self.y_vec)

        residual_vecs = [self.y_vec] # for the first tree
        if update_leaf_with_residual: # get residual vectors (= target values for the next tree), residual_vecs[i] is used to construct the i-th tree.
            if SklearnEnsembleObj in [GradientBoostingRegressor]:
                pass
            else:
                raise(ParameterFormatError,'update_leaf_with_residual=True is only supported when SklearnEnsembleObj is GradientBoostingRegressor.')
            for hat_y_train in sklearn_estimator.staged_predict(self.x_vecs):
                residual = self.y_vec - hat_y_train
                residual_vecs.append(residual)

        # sequentially copy metatrees
        for i in range(num_trees_built, self.c_num_metatrees):
            if SklearnEnsembleObj in [RandomForestRegressor, RandomForestClassifier]:
                sklearn_tree = sklearn_estimator.estimators_[i-num_trees_built]
            elif SklearnEnsembleObj in [GradientBoostingRegressor, GradientBoostingClassifier]:
                sklearn_tree = sklearn_estimator.estimators_[i-num_trees_built][0] # NOTE: なぜかGradientBoostingRegressor.estimators_は二重配列
            self.hn_metatree_list[i] = MetaTreeLearnModel(
                SubModel=self.SubModel,
                c_dim_continuous = self.c_dim_continuous,
                c_dim_categorical = self.c_dim_categorical,
                c_max_depth = self.c_max_depth,
                c_num_children_vec = self.c_num_children_vec,
                c_feature_candidates = self.c_feature_candidates,
                h0_constants_SubModel = self.h0_constants_SubModel,
                sub_h0_params = self.sub_h0_params,
                h0_split = self.h0_split_list,
                h0_feature_weight_vec = self.h0_feature_weight_vec,
                threshold_params = self.threshold_params,
                lambda_xgb = self.lambda_xgb,
                gamma_xgb = self.gamma_xgb,
                seed = self.seed, # TODO: rngをそのまま渡したい．
            )

            if update_leaf_with_residual:
                target = residual_vecs[i]
            else:
                target = self.y_vec

            self.hn_metatree_list[i].build(
                x_continuous_vecs=self.x_continuous_vecs,
                x_categorical_vecs=self.x_categorical_vecs,
                y_vec=target,
                split_strategy='copy_from_sklearn_tree',
                criterion=self.tree_criterion,
                sample_weight=None,
                building_scheme='depth_first', # nothing to do
                ignore_check=True,
                max_leaf_nodes = None,
                max_features = None,
                sklearn_tree=sklearn_tree
                )
            self.hn_metatree_list[i].calc_pred_dist()

        return [None], sklearn_estimator
    
    def _copy_trees_from_xgb(
        self,
        XGBRegressorObj: BaseEstimator,
        params_xgb: dict,
    ):
        num_trees_built = 0
        self.hn_metatree_list = [None for _ in range(self.c_num_metatrees)]
        # build xgb-trees
        self.x_vecs = np.concatenate([self.x_continuous_vecs, self.x_categorical_vecs],axis=1)
        xgb_model = XGBRegressorObj(**params_xgb)
        xgb_model.fit(self.x_vecs,self.y_vec)
        # create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            xgb_file_path = os.path.join(tmpdir,'xgb_model.json')
            # xgb_file_path = 'xgb_model.json' # for debug
            # save xgb model txtfile
            xgb_model.get_booster().dump_model(xgb_file_path, dump_format='json')
            # load xgb model json and get list of model_dict with binary node ids
            # with open(xgb_file_path) as f:
            #     list_of_model_dict = json.load(f)
            list_of_model_dict = parse_xgb_json_model(xgb_file_path)
            # model_dict = parse_xgb_model(xgb_file_path)
            # sequentially copy metatrees
            for i in range(num_trees_built, self.c_num_metatrees): # NOTE: early_stopping (xgbの木の本数がc_num_metatreesよりも少ない場合) には非対応
                self.hn_metatree_list[i] = MetaTreeLearnModel(
                    SubModel=self.SubModel,
                    c_dim_continuous = self.c_dim_continuous,
                    c_dim_categorical = self.c_dim_categorical,
                    c_max_depth = self.c_max_depth,
                    c_num_children_vec = self.c_num_children_vec,
                    c_feature_candidates = self.c_feature_candidates,
                    h0_constants_SubModel = self.h0_constants_SubModel,
                    sub_h0_params = self.sub_h0_params,
                    h0_split = self.h0_split_list,
                    h0_feature_weight_vec = self.h0_feature_weight_vec,
                    threshold_params = self.threshold_params,
                    lambda_xgb = self.lambda_xgb,
                    gamma_xgb = self.gamma_xgb,
                    seed = self.seed, # TODO: rngをそのまま渡したい．
                )
                self.hn_metatree_list[i].build(
                    x_continuous_vecs=self.x_continuous_vecs,
                    x_categorical_vecs=self.x_categorical_vecs,
                    y_vec=self.y_vec,
                    split_strategy='copy_from_xgb',
                    criterion=self.tree_criterion,
                    sample_weight=None,
                    building_scheme='depth_first', # nothing to do
                    ignore_check=True,
                    max_leaf_nodes = None,
                    max_features = None,
                    xgb_tree_info = list_of_model_dict[i]
                    )
                self.hn_metatree_list[i].calc_pred_dist()
        

        return [None], xgb_model