import sys, os

import numpy as np
from copy import deepcopy
from collections import deque, defaultdict

import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex

from typing import Dict, List, Tuple, Union, Optional, Generator, Any
from numpy.typing import ArrayLike

from .. import _check
from .. import base
from .. import bernoulli, categorical, normal, linearregression
from .._exceptions import (CriteriaError, DataFormatError, ParameterFormatError,
                         ParameterFormatWarning, ResultWarning)

from ._threshold_candidates import GenThresholdCandidates
from ._feature_subsample import _feature_subsample
import sklearn
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from . import _constants
_CMAP = _constants._CMAP

MODELS = _constants.MODELS
DISCRETE_MODELS = _constants.DISCRETE_MODELS
CONTINUOUS_MODELS = _constants.CONTINUOUS_MODELS
CLF_MODELS = _constants.CLF_MODELS
REG_MODELS = _constants.REG_MODELS

def _squared_error_with_weight(
        y_vec:ArrayLike, 
        y_pred_vec:Union[float,ArrayLike],
        sample_weight:ArrayLike
        ):
    return (((y_vec-y_pred_vec)**2)*sample_weight).sum()
    
def _calc_xgb_impurity_with_weight(
        y_vec:ArrayLike, 
        y_pred_vec:Union[float,ArrayLike],
        sample_weight:ArrayLike,
        lambda_:float,
        ):
    # gradient = (2 * (sample_weight * (y_pred_vec - y_vec)**2)).sum()
    gradient = np.sum(-(2 * (sample_weight * y_vec)))
    hessian = np.sum(sample_weight)
    return np.square(gradient) / (hessian + lambda_)

class _LearnNode:
    def __init__(
            self,
            rng: np.random.Generator,
            SubModel: Any,
            c_id: str,
            c_depth: int,
            h0_split: float,
            c_ancestor_nodes: list,
            num_samples: Optional[int],
            sample_indices: Optional[np.ndarray],
            is_leaf: bool,
            c_feature_candidates: List,
            c_num_assignment_vec: List,
            c_data_region: List,
            children: List['_LearnNode'] = [],
            feature: Optional[int] = None,
            threshold: Optional[float] = None,
            h0_constants_SubModel: Dict = {},
            sub_h0_params: Dict = {},
            threshold_candidates: Optional[List] = None,
            ):
        
        # Static valuables which should not be updated
        self.c_id = c_id
        self.c_depth = c_depth
        self.c_ancestor_nodes = c_ancestor_nodes
        self.h0_split = h0_split
        self.rng = rng
        self.c_data_region = c_data_region

        # valuables which can be updated
        self.num_samples = num_samples
        self.sample_indices = sample_indices
        self.is_leaf = is_leaf
        self.children = children
        self.c_feature_candidates = c_feature_candidates
        self.c_num_assignment_vec = c_num_assignment_vec
        self._split_rule = [feature, threshold]
        self.h0_constants_SubModel = h0_constants_SubModel
        self.sub_h0_params = sub_h0_params
        self.sub_model = SubModel.LearnModel(
                **self.h0_constants_SubModel,
                **self.sub_h0_params
                )
        self.threshold_candidates = threshold_candidates
        
    def _set_h0_params(
            self,
            h0_split: Optional[int] = None,
            sub_h0_params: Optional[Dict] = None,
            ):
        if h0_split is not None:
            self.h0_split = h0_split
        if sub_h0_params is not None:
            self.sub_model.set_h0_params(**self.sub_h0_params)
        return self

class MetaTreeLearnModel(base.Posterior,base.PredictiveMixin):
    """The posterior distribution and the predictive distribution of MetaTree.
    c_num_children_vec : int or numpy.ndarray, optional
        number of children of a node, when the node is split by the feature.
        A positive integer or a vector of positive integers whose length is 
        ``c_dim_features``, by default 2.
        If a single integer is input it will be broadcasted for every feature.
    c_max_depth : int, optional
        A positive integer, by default 2
    c_feature_candidates : 'all' or List, optional
        Str or list of possible features for split, by default 'all'
    h0_constants_SubModel : dict, optional
        constants for self.SubModel.LearnModel, by default {}
    h0_feature_weight_vec : numpy.ndarray, optional
        A vector of positive real numbers whose length is 
        ``c_dim_continuous+c_dim_categorical``, 
        by default [1/c_num_assignment_vec.sum(),...,1/c_num_assignment_vec.sum()].
    h0_split : list or float, optional
        A real number or a vector in :math:`[0, 1]`, by default 0.5.
        If a single integer is input it will be broadcasted for every depth.
        If a vector is input, for every node of the depth d the value is set to the d-th element of the vector.
    seed : {None, int}, optional
        A seed to initialize numpy.random.default_rng(),
        by default None

    Attributes
    ----------
    ''not yet updated''
    c_dim_features: int
        c_dim_continuous + c_dim_categorical
    hn_k_weight_vec : numpy.ndarray
        A vector of positive real numbers whose length is 
        ``c_dim_continuous+c_dim_categorical``
    hn_g : float
        A real number in :math:`[0, 1]`
    sub_hn_params : dict
        hn_params for self.SubModel.LearnModel
    hn_metatree_list : list of metatree._Node
        Root nodes of meta-trees
    hn_metatree_prob_vec : numpy.ndarray
        A vector of real numbers in :math:`[0, 1]` 
        that represents prior distribution of h0_metatree_list.
        Sum of its elements is 1.0.
    """
    def __init__(
            self,
            SubModel: Any, # TODO サブモデルクラスでの型アノテーション
            c_dim_continuous: int,
            c_dim_categorical: int,
            c_max_depth: int, # TODO c_max_depth=0 causes error during broadcasting self.h0_split = np.ones(self.c_max_depth,dtype=float), it creates empty array.
            c_num_children_vec: List or int = 2,
            c_feature_candidates: str or ArrayLike = 'all',
            c_num_assignment_vec: Optional[ArrayLike] = None,
            h0_constants_SubModel: Dict = {},
            sub_h0_params: Dict = {},
            h0_split: List or float = 0.5,
            h0_feature_weight_vec: Optional[ArrayLike] = None,
            threshold_params: Optional[Dict] = None,
            lambda_xgb:float = 1.,
            gamma_xgb:float = 0.,
            seed: Optional[int] = None,
        ):
    
        # Static valuables which should not be updated, from the input
        self.c_dim_continuous = _check.nonneg_int(c_dim_continuous,'c_dim_continuous',ParameterFormatError)
        self.c_dim_categorical = _check.nonneg_int(c_dim_categorical,'c_dim_categorical',ParameterFormatError)
        self.c_dim_features = _check.pos_int(c_dim_continuous+c_dim_categorical,'c_dim_continuous+c_dim_categorical',ParameterFormatError)

        self.c_max_depth = _check.nonneg_int(c_max_depth,'c_max_depth',ParameterFormatError)

        _check.pos_ints(c_num_children_vec,'c_num_children_vec',ParameterFormatError)
        if np.any(c_num_children_vec<2):
            raise(ParameterFormatError(
                'All the elements of c_num_children_vec must be greater than or equal to 2: '
                +f'c_num_children_vec={c_num_children_vec}.'
            ))
        self.c_num_children_vec = np.ones(self.c_dim_features,dtype=int)*2
        self.c_num_children_vec[:] = c_num_children_vec

        if c_feature_candidates == 'all':
            self.c_feature_candidates = list(range(self.c_dim_features))
        else:
            _check.unique_list(c_feature_candidates,'c_feature_candidates',ParameterFormatError)
            _check.floats_in_closedrange(c_feature_candidates,'c_feature_candidates',0,self.c_dim_features-1,ParameterFormatError)
            self.c_feature_candidates = c_feature_candidates

        self.c_num_assignment_vec = np.ones(self.c_dim_features,dtype=int)
        self.c_num_assignment_vec[:self.c_dim_continuous] = self.c_max_depth
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
        self.h0_constants_SubModel = self.SubModel.LearnModel(**h0_constants_SubModel).get_constants()
        self.sub_h0_params = self.SubModel.LearnModel(**sub_h0_params).get_h0_params()

        if h0_feature_weight_vec is None:
            self.h0_feature_weight_vec = np.full(self.c_dim_features, 1/self.c_dim_features)
        else:
            self.h0_feature_weight_vec = _check.proba_vec(h0_feature_weight_vec,'h0_feature_weight_vec',ParameterFormatError)

        self.lambda_xgb = _check.nonneg_float(lambda_xgb,'lambda_xgb',ParameterFormatError)
        self.gamma_xgb = _check.nonneg_float(gamma_xgb,'gamma_xgb',ParameterFormatError)

        if seed is None:
            self.seed = seed
        else:
            self.seed = _check.int_(seed, 'seed', ParameterFormatError)
        self.rng = np.random.default_rng(self.seed)

        # automatically defined
        # tree structure
        self.root_node = _LearnNode(
            self.rng,
            self.SubModel,
            c_id = '',
            c_depth = 0,
            h0_split = self.h0_split_list[0],
            c_ancestor_nodes = [],
            num_samples = None,
            sample_indices = None,
            is_leaf = True,
            c_feature_candidates=self.c_feature_candidates,
            c_num_assignment_vec=self.c_num_assignment_vec,
            c_data_region = [],
            children = [],
            feature = None,
            threshold = None,
            h0_constants_SubModel = self.h0_constants_SubModel,
            sub_h0_params = self.sub_h0_params,
            )
        self.nodes_list = [self.root_node]
        self.leaf_nodes_list = [self.root_node]
        self.inner_nodes_list = []

        # hn_params
        self.hn_feature_weight_vec = deepcopy(self.h0_feature_weight_vec)
        self.hn_split_list = deepcopy(self.h0_split_list)
        self.sub_hn_params = {}

        self.x_continuous_vecs = None
        self.x_categorical_vecs = None
        self.y_vec = None

        # thresholds
        self.threshold_params = {
            'type': None,
            'num_thresholds': None,
        }
        if threshold_params is None:
            self.threshold_params['type'] = 'all'
            self.threshold_params['num_thresholds'] = None
        else: # TODO: dictのParameterFormat判定
            self.threshold_params['type'] = threshold_params['type']
            self.threshold_params['num_thresholds'] = threshold_params['num_thresholds']
        self.threshold_generator = GenThresholdCandidates(self.rng) # WARNING: pythonのデータ構造としてのGeneratorではないため，名前が良くないかもしれない．

        self.threshold_candidates = None

    def _check_sample_x(
            self, 
            x_continuous: ArrayLike or List,
            x_categorical: ArrayLike or List,
            ):
        if self.c_dim_continuous > 0 and self.c_dim_categorical > 0:
            _check.float_vecs(x_continuous,'x_continuous',DataFormatError)
            _check.shape_consistency(
                x_continuous.shape[-1],'x_continuous.shape[-1]',
                self.c_dim_continuous,'self.c_dim_continuous',
                ParameterFormatError
                )
            x_continuous = x_continuous.reshape([-1,self.c_dim_continuous])
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
    # not yet finished
    ##########################
    def calc_pred_dist(self): # build, update が終わったあとで，predictiveの計算を行う．まだ正規分布だけ．TODO:カテゴリカル，線形回帰への対応
        for node in self.nodes_list:
            node.sub_model.calc_pred_dist()
            node.proba = np.prod([anc.hn_split for anc in node.c_ancestor_nodes]) * (1 - node.hn_split)
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
        x_continuous_new: ArrayLike, 
        x_categorical_new: ArrayLike,
        loss:str='squared',
        method:str='recursive'  # 'recursive', 'iterative', or 'original'
        ):
        """
        予測を行う関数（複数実装対応版）
        
        この実装では、3つの異なる方法で予測を行うことができます：
        
        - original: 元の実装（各ノードで根ノードからの全分岐ルールを毎回適用）
        - recursive: 再帰的最適化実装（木を辿りながら効率的に計算）
        - iterative: 反復的最適化実装（再帰オーバーヘッドを排除）
        
        計算効率の比較：
        - original: O(N×D×M) - N:サンプル数, D:木の深さ, M:ノード数
        - recursive/iterative: O(N×D) - 大幅な効率改善
        
        Parameters
        ----------
        x_continuous_new : ArrayLike
            予測対象の連続特徴量データ
        x_categorical_new : ArrayLike
            予測対象のカテゴリカル特徴量データ
        loss : str, optional
            損失関数の種類, by default 'squared'
        method : str, optional
            実装方法, by default 'recursive'
            - 'original': 元の実装（比較用・教育用）
            - 'recursive': 再帰的最適化実装
            - 'iterative': 反復的最適化実装
            
        Returns
        -------
        ArrayLike
            予測値のベクトル
        """
        x_continuous_new, x_categorical_new = self._check_sample_x(x_continuous_new, x_categorical_new)
        hat_y_vec = np.zeros(x_continuous_new.shape[0])
        
        if method == 'original':
            # 元の実装を使用（比較・教育目的）
            self._make_prediction_original(
                x_continuous_new, 
                x_categorical_new, 
                hat_y_vec, 
                loss
            )
        elif method == 'recursive':
            # 再帰的最適化実装を使用
            self._make_prediction_recursive(
                self.root_node, 
                np.ones(x_continuous_new.shape[0], dtype=bool),  # 全サンプルから開始
                x_continuous_new, 
                x_categorical_new, 
                hat_y_vec, 
                loss
            )
        elif method == 'iterative':
            # 反復的最適化実装を使用
            self._make_prediction_iterative(
                x_continuous_new, 
                x_categorical_new, 
                hat_y_vec, 
                loss
            )
        else:
            raise ValueError(f"Unknown method: {method}. Choose 'original', 'recursive', or 'iterative'")
        
        return hat_y_vec
        
    def _make_prediction_original(
        self,
        x_continuous_new: ArrayLike,
        x_categorical_new: ArrayLike,
        hat_y_vec: ArrayLike,
        loss: str
    ):
        """
        元の予測実装（比較・教育目的）
        
        この実装は元のコードと同じロジックを使用しています。
        各ノードに対して、根ノードからの全ての分岐ルールを毎回適用する方法です。
        
        計算効率：O(N×D×M) - N:サンプル数, D:木の深さ, M:ノード数
        
        利点：
        - シンプルで理解しやすい
        - 各ノードが独立して処理される
        - デバッグが容易
        
        欠点：
        - 計算効率が悪い（同じ分岐ルールの重複適用）
        - 木が深い・ノード数が多い場合に遅くなる
        - メモリ効率も劣る
        
        Parameters
        ----------
        x_continuous_new : ArrayLike
            連続特徴量の新しいデータ
        x_categorical_new : ArrayLike
            カテゴリカル特徴量の新しいデータ
        hat_y_vec : ArrayLike
            予測値を蓄積する配列（in-place更新）
        loss : str
            損失関数の種類
        """
        # 元の実装：各ノードで根ノードからの全ての分岐ルールを毎回適用
        for node in self.nodes_list:
            sample_indices_new, num_sample_new = self._get_data_indices_node_original(
                node.c_data_region, 
                x_continuous_new, 
                x_categorical_new
            )
            if num_sample_new > 0:
                node_prediction = node.sub_model.make_prediction(loss=loss)
                hat_y_vec += node_prediction * node.proba * sample_indices_new

    def _get_data_indices_node_original(
        self,
        data_region: List,
        x_continuous: ArrayLike,
        x_categorical: ArrayLike,
    ):
        """
        元の実装：根ノードからの全ての分岐ルールを毎回適用してデータインデックスを取得
        
        この関数は、指定されたdata_regionに基づいて、どのサンプルが
        特定のノードに割り当てられるかを判定します。
        
        計算効率：O(D×N) - D:根からの深さ, N:サンプル数
        全ノードで呼び出されるため、全体では O(N×D×M) となります。
        
        アルゴリズム：
        1. 全サンプルから開始（sample_indices = True）
        2. data_region内の各分岐ルールを順次適用
        3. 条件を満たすサンプルのみを残す
        4. 最終的に残ったサンプルの数とマスクを返す
        
        Parameters
        ----------
        data_region : List
            根ノードから現在のノードまでの分岐ルールのリスト
            各要素は ((feature_index, threshold), child_direction) の形式
        x_continuous : ArrayLike
            連続特徴量データ
        x_categorical : ArrayLike
            カテゴリカル特徴量データ
            
        Returns
        -------
        sample_indices : ArrayLike
            該当ノードに割り当てられるサンプルのブールマスク
        num_samples : int
            該当ノードに割り当てられるサンプル数
            
        Notes
        -----
        この実装では、各ノードで根ノードからの全ての分岐ルールを
        毎回適用するため、計算の重複が発生します。
        最適化版では、木を辿りながら増分的にマスクを更新することで
        この重複を回避しています。
        """
        sample_indices = np.full(x_continuous.shape[0], True)  # 全サンプルから開始

        for reg in data_region:
            # 親ノードでの分岐ルールを'全て'の学習データに適用し，
            # 該当ノードに割り当てられる学習データのインデックスを得る
            split_rule = reg[0]  # (feature_index, threshold)
            child_direction = reg[1]  # 左の子=0, 右の子=1, カテゴリカルなら値そのもの
            
            if split_rule[1] is None:  # threshold is None -> categorical feature
                categorical_feature_idx = split_rule[0] - self.c_dim_continuous
                condition = (x_categorical[:, categorical_feature_idx] == child_direction)
            else:  # continuous feature
                if child_direction == 0:  # 左の子（閾値以下）
                    condition = (x_continuous[:, split_rule[0]] <= split_rule[1])
                else:  # 右の子（閾値より大きい）  
                    condition = (x_continuous[:, split_rule[0]] > split_rule[1])
            
            # 現在の条件と新しい条件の両方を満たすサンプルのみを保持
            sample_indices = sample_indices & condition
        
        num_samples = np.count_nonzero(sample_indices)
        return sample_indices, num_samples
        
    def _make_prediction_iterative(
        self,
        x_continuous_new: ArrayLike,
        x_categorical_new: ArrayLike,
        hat_y_vec: ArrayLike,
        loss: str
    ):
        """
        反復的（スタックベース）な予測関数 - 再帰のオーバーヘッドを排除
        
        この実装では、スタックを使用して再帰呼び出しを排除し、
        関数呼び出しのオーバーヘッドを削減します。特に深いツリーや
        大量のノードがある場合に効果的です。
        
        アルゴリズム：
        1. 根ノードをスタックに初期化
        2. スタックが空になるまで以下を繰り返し：
           a. ノードをスタックから取り出し
           b. 現在のノードでの予測寄与を計算
           c. 内部ノードの場合、子ノードをスタックに追加
        
        メリット：
        - 再帰呼び出しのオーバーヘッド排除
        - スタックオーバーフローの回避
        - 深いツリーでの安定動作
        - メモリ使用量の制御が容易
        
        Parameters
        ----------
        x_continuous_new : ArrayLike
            連続特徴量の新しいデータ
        x_categorical_new : ArrayLike
            カテゴリカル特徴量の新しいデータ
        hat_y_vec : ArrayLike
            予測値を蓄積する配列（in-place更新）
        loss : str
            損失関数の種類
        """
        # スタックを使用して反復的に処理
        # 各要素: (node, sample_mask)
        stack = [(self.root_node, np.ones(x_continuous_new.shape[0], dtype=bool))]
        
        while stack:
            node, current_sample_mask = stack.pop()
            
            # 現在のノードに割り当てられるサンプルが存在しない場合はスキップ
            if not np.any(current_sample_mask):
                continue
                
            # 現在のノードでの予測への寄与を計算
            node_contribution = node.proba * current_sample_mask
            if np.any(node_contribution):
                node_prediction = node.sub_model.make_prediction(loss=loss)
                hat_y_vec += node_prediction * node_contribution
            
            # 内部ノードの場合、子ノードをスタックに追加
            if not node.is_leaf:
                # 子ノードを逆順でスタックに追加（元の再帰順序を保持するため）
                # スタックは後入先出なので、逆順で追加することで
                # 再帰実装と同じ順序で処理される
                for i in reversed(range(len(node.children))):
                    child = node.children[i]
                    child_mask = self._apply_split_rule_to_mask(
                        node._split_rule, 
                        i, 
                        current_sample_mask, 
                        x_continuous_new, 
                        x_categorical_new
                    )
                    stack.append((child, child_mask))
        
    def _make_prediction_recursive(
        self,
        node: '_LearnNode',
        current_sample_mask: ArrayLike,
        x_continuous_new: ArrayLike,
        x_categorical_new: ArrayLike,
        hat_y_vec: ArrayLike,
        loss: str
    ):
        """
        根ノードから順番に辿ることで分岐ルールの適用回数を減らした最適化された予測関数
        
        Parameters
        ----------
        node : _LearnNode
            現在のノード
        current_sample_mask : ArrayLike
            現在のノードに到達するサンプルのマスク
        x_continuous_new : ArrayLike
            連続特徴量の新しいデータ
        x_categorical_new : ArrayLike
            カテゴリカル特徴量の新しいデータ
        hat_y_vec : ArrayLike
            予測値を蓄積する配列（in-place更新）
        loss : str
            損失関数の種類
        """
        # 現在のノードに割り当てられるサンプルが存在しない場合は何もしない
        if not np.any(current_sample_mask):
            return
            
        # 現在のノードでの予測への寄与を計算
        node_contribution = node.proba * current_sample_mask
        if np.any(node_contribution):
            node_prediction = node.sub_model.make_prediction(loss=loss)
            hat_y_vec += node_prediction * node_contribution
        
        # 葉ノードの場合は終了
        if node.is_leaf:
            return
            
        # 内部ノードの場合、子ノードに再帰的に処理を委譲
        for i, child in enumerate(node.children):
            # 現在のノードの分岐ルールを適用して子ノードのマスクを計算
            child_mask = self._apply_split_rule_to_mask(
                node._split_rule, 
                i, 
                current_sample_mask, 
                x_continuous_new, 
                x_categorical_new
            )
            # 子ノードで再帰的に処理
            self._make_prediction_recursive(
                child, 
                child_mask, 
                x_continuous_new, 
                x_categorical_new, 
                hat_y_vec, 
                loss
            )
    
    def _apply_split_rule_to_mask(
        self,
        split_rule: List,
        child_index: int,
        parent_mask: ArrayLike,
        x_continuous: ArrayLike,
        x_categorical: ArrayLike,
    ) -> ArrayLike:
        """
        分岐ルールを親ノードのマスクに適用して子ノードのマスクを生成
        
        Parameters
        ----------
        split_rule : List
            分岐ルール [feature_index, threshold]
        child_index : int
            子ノードのインデックス
        parent_mask : ArrayLike
            親ノードのサンプルマスク
        x_continuous : ArrayLike
            連続特徴量データ
        x_categorical : ArrayLike
            カテゴリカル特徴量データ
            
        Returns
        -------
        ArrayLike
            子ノードのサンプルマスク
        """
        feature_idx = split_rule[0]
        threshold = split_rule[1]
        
        # 親ノードに割り当てられていないサンプルは除外
        child_mask = parent_mask.copy()
        
        if threshold is None:  # カテゴリカル特徴量
            categorical_feature_idx = feature_idx - self.c_dim_continuous
            condition = (x_categorical[:, categorical_feature_idx] == child_index)
        else:  # 連続特徴量
            if child_index == 0:  # 左の子（閾値以下）
                condition = (x_continuous[:, feature_idx] <= threshold)
            else:  # 右の子（閾値より大きい）
                condition = (x_continuous[:, feature_idx] > threshold)
        
        # 親ノードの条件と子ノードの条件を両方満たすサンプルのみを選択
        child_mask = child_mask & condition
        
        return child_mask
    
    def benchmark_prediction_methods(
        self,
        x_continuous_new: ArrayLike,
        x_categorical_new: ArrayLike,
        loss: str = 'squared',
        methods: List[str] = ['original', 'recursive', 'iterative'],
        n_trials: int = 5
    ) -> Dict[str, float]:
        """
        3つの予測実装のパフォーマンスを比較するベンチマーク関数
        
        各実装の特徴：
        - original: 元の実装（O(N×D×M)）- 各ノードで全分岐ルールを適用
        - recursive: 再帰的最適化（O(N×D)）- 木を辿りながら効率的に計算  
        - iterative: 反復的最適化（O(N×D)）- 再帰オーバーヘッドも排除
        
        Parameters
        ----------
        x_continuous_new : ArrayLike
            予測対象の連続特徴量データ
        x_categorical_new : ArrayLike
            予測対象のカテゴリカル特徴量データ
        loss : str, optional
            損失関数の種類, by default 'squared'
        methods : List[str], optional
            測定する手法のリスト, by default ['original', 'recursive', 'iterative']
        n_trials : int, optional
            各手法の実行回数（平均値を算出）, by default 5
            
        Returns
        -------
        Dict[str, float]
            各手法の平均実行時間（秒）
        """
        import time
        
        results = {}
        x_continuous_new, x_categorical_new = self._check_sample_x(x_continuous_new, x_categorical_new)
        
        print(f"ベンチマーク開始: {len(methods)}つの手法を{n_trials}回ずつ実行")
        print(f"データサイズ: {x_continuous_new.shape[0]} サンプル")
        print(f"木の情報: {len(self.nodes_list)} ノード, 最大深度: {self.c_max_depth}")
        print("-" * 60)
        
        for method in methods:
            try:
                times = []
                
                # 複数回実行して平均を取る
                for trial in range(n_trials):
                    start_time = time.time()
                    _ = self.make_prediction(x_continuous_new, x_categorical_new, loss, method)
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                avg_time = np.mean(times)
                std_time = np.std(times)
                min_time = np.min(times)
                max_time = np.max(times)
                
                results[method] = {
                    'avg_time': avg_time,
                    'std_time': std_time,
                    'min_time': min_time,
                    'max_time': max_time,
                    'all_times': times
                }
                
                print(f"{method:>10} method: {avg_time:.6f} ± {std_time:.6f} seconds")
                print(f"{'':>10}          (min: {min_time:.6f}, max: {max_time:.6f})")
                
            except Exception as e:
                results[method] = {'error': str(e)}
                print(f"{method:>10} method failed: {e}")
        
        # 相対性能の比較
        if len([k for k in results.keys() if 'error' not in results[k]]) >= 2:
            print("-" * 60)
            print("相対性能比較:")
            
            # 基準となる最も遅い手法を見つける
            valid_methods = {k: v for k, v in results.items() if 'error' not in v}
            if valid_methods:
                slowest_time = max(v['avg_time'] for v in valid_methods.values())
                
                for method, result in valid_methods.items():
                    if 'avg_time' in result:
                        speedup = slowest_time / result['avg_time']
                        print(f"{method:>10}: {speedup:.2f}倍高速")
        
        return results
    
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
    def visualize_model(self,filename=None,format=None,view=True,hn_params=True,h_params=False,p_params=False):
        """Visualize the stochastic data generative model and generated samples.

        Note that values of categorical features will be shown with jitters.

        Parameters
        ----------
        filename : str, optional
            Filename for saving the figure, by default ``None``
        format : str, optional
            Rendering output format (``\"pdf\"``, ``\"png\"``, ...).
        sample_size : int, optional
            A positive integer, by default 100
        x_continuous : numpy ndarray, optional
            2 dimensional float array whose size is ``(sample_size,c_dim_continuous)``, 
            by default None.
        x_categorical : numpy ndarray, optional
            2 dimensional int array whose size is ``(sample_size,c_dim_categorical)``, 
            by default None. Each element x_categorical[i,j] must satisfy 
            0 <= x_categorical[i,j] < self.c_num_children_vec[self.c_dim_continuous+i].

        Examples
        --------
        >>> from bayesml import metatree
        >>> model = metatree.GenModel(
        >>>     c_dim_continuous=1,
        >>>     c_dim_categorical=1)
        >>> model.gen_params(threshold_type='random')
        >>> model.visualize_model()

        .. image:: ./images/metatree_example1.png

        .. image:: ./images/metatree_example2.png

        See Also
        --------
        graphviz.Digraph
        """

        try:
            import graphviz
            tree_graph = graphviz.Digraph(filename=filename,format=format)
            tree_graph.attr("node",shape="box",fontname="helvetica",style="rounded,filled")
            root_node_label = self._visualize_model_recursion(
                tree_graph,
                self.root_node,
                None,
                None,
                None,
                1.0,
                hn_params=hn_params,
                h_params=h_params,
                p_params=p_params,
                )
            # Can we show the image on the console without saving the file?
            if view:
                tree_graph.view()
        except ImportError as e:
            print(e)
        except graphviz.CalledProcessError as e:
            print(e)
        return tree_graph, root_node_label

    def _visualize_model_recursion(self,tree_graph,node:_LearnNode,parent_id,parent_split_rule,sibling_num,p_v,hn_params,h_params,p_params):
        tmp_id = node.c_id
        tmp_p_v = p_v
        proba_node = tmp_p_v*(1-node.hn_split)
        label_string = f"node_id='{tmp_id}'\\l"
        label_string += f"samples={node.num_samples}\\l"
        # add node information
        # if node.is_leaf:
            # label_string += 'k=None\\l'
        # else:
            # label_string += f'k={node._split_rule[0]}\\l'
            # if node._split_rule[0] < self.c_dim_continuous:
                # label_string += f'threshold=\\l{node._split_rule[1]:.2f}\\l'
        label_string += f'hn_split={node.hn_split:.2f}\\lproba_node={proba_node:.2f}\\l'
        sub_params = {}
        if node.sub_model is not None:
            # print model name
            model_names = [
                'bernoulli',
                'categorical',
                'normal',
                'multivariate_normal',
                'linearregression',
                'poisson',
                'exponential',
            ]
            model_list = [eval(modelname) for modelname in model_names]
            for model, model_name in zip(model_list, model_names):
                if self.SubModel == model:
                    label_string += f'model={model_name}\\l'
                    break
            if hn_params + h_params + p_params > 1:
                raise(ParameterFormatError(
                    'Only one of hn_params, h_params, and p_params can be True.'
                ))
            if hn_params:
                label_string += 'sub_hn_params={'
                sub_params = node.sub_model.get_hn_params()
            if h_params:
                label_string += 'sub_params={'
                try:
                    sub_params = node.sub_model.estimate_params(loss='0-1',dict_out=True)
                except:
                    sub_params = node.sub_model.estimate_params(dict_out=True)
            if p_params:
                label_string += 'sub_p_params={'
                sub_params = node.sub_model.get_p_params()
        for key,value in sub_params.items():
            try:
                label_string += f'\\l    {key}:{value:.2f}'
            except:
                try:
                    label_string += f'\\l    {key}:{np.array2string(value,precision=2,max_line_width=1)}'
                except:
                    label_string += f'\\l    {key}:{value}'
        label_string += '}\\l'
            
        tree_graph.node(name=f'{tmp_id}',label=label_string,fillcolor=f'{rgb2hex(_CMAP(proba_node))}')
        if proba_node > 0.65:
            tree_graph.node(name=f'{tmp_id}',fontcolor='white')
        
        # add edge information
        if parent_id is not None:
            if parent_split_rule[0] < self.c_dim_continuous:
                if sibling_num == 0:
                    tree_graph.edge(f'{parent_id}', f'{tmp_id}', label=f'x_{parent_split_rule[0]} ≦ {parent_split_rule[1]:.2f}')
                else:
                    tree_graph.edge(f'{parent_id}', f'{tmp_id}', label=f'{parent_split_rule[1]:.2f} < x_{parent_split_rule[0]}')
            else:
                tree_graph.edge(f'{parent_id}', f'{tmp_id}', label=f'x_{parent_split_rule[0]} = {sibling_num}')
        
        if node.is_leaf != True:
            for i in range(self.c_num_children_vec[node._split_rule[0]]):
                self._visualize_model_recursion(tree_graph,node.children[i],tmp_id,node._split_rule,i,tmp_p_v*node.hn_split,hn_params,h_params,p_params)
        return label_string
    
    ##########################################
    # update posterior
    ##########################################
    
    ##########################################
    # update posterior on given nodes
    ##########################################
    def _update_posterior_all_nodes_given(
            self,
            x_continuous_vecs:ArrayLike,
            x_categorical_vecs:ArrayLike,
            y_vec:ArrayLike,
            reset_param:bool = False,
        ):

        self._update_posterior_submodel(self.nodes_list_sorted_by_depth, x_continuous_vecs, x_categorical_vecs, y_vec,reset_param)
        self._update_posterior_hn_split(self.nodes_list_sorted_by_depth)
        return self
    def _update_posterior_hn_split_node(
            self,
            node: _LearnNode,
        ): 
        # TODO: この計算方法ではほとんどhn_split=1になってしまうのでは？平滑化を考える必要あり．
        # FIXME: 連続特徴量ではhn_splitが0に近くなる一方で，離散特徴量ではhn_splitがかなり大きくなる．なぜ？
        # this also to be used by parent node
        id = node.c_id
        if node.is_leaf:
            node._log_marginal_likelihood = node.sub_log_marginal_likelihood
            node.hn_split = 0.
        else:
            log_marginal_likelihood_children = sum(map(lambda child: child._log_marginal_likelihood, node.children))
            tmp = np.log(node.h0_split) + log_marginal_likelihood_children
            node._log_marginal_likelihood = np.logaddexp((np.log(1-node.h0_split)+node.sub_log_marginal_likelihood), tmp)
            node.log_hn_split = tmp - node._log_marginal_likelihood
            node.hn_split = np.exp(node.log_hn_split)
        return
    def _update_posterior_hn_split(
            self,
            nodes_list_sorted_by_depth: List, #list of nodes to be updated
        ):
        for node in nodes_list_sorted_by_depth:
            self._update_posterior_hn_split_node(node)
        return
    def _update_posterior_submodel(
            self,
            nodes_list: List,
            x_continuous_vecs: ArrayLike,
            x_categorical_vecs: ArrayLike,
            y_vec: ArrayLike,
            reset_param:bool = False,
        ):
        for node in nodes_list: # TODO: 全てのノードでの事後分布更新は無駄
            self._update_posterior_submodel_node(node, x_continuous_vecs, x_categorical_vecs, y_vec,reset_param)
        return
    def _update_posterior_submodel_node(
            self,
            node: _LearnNode,
            x_continuous_vecs:ArrayLike,
            x_categorical_vecs:ArrayLike,
            y_vec:ArrayLike,
            reset_param:bool = False,
        ):
        id = node.c_id
        if reset_param:
            node.sub_model.reset_hn_params()
        node.sample_indices, node.num_samples = self._get_data_indices_node(node.c_data_region, x_continuous_vecs, x_categorical_vecs)
        if node.num_samples > 0:
            node.sub_model._update_posterior(y_vec[node.sample_indices])
            node.sub_model.calc_pred_dist()
            # node.sub_log_marginal_likelihood = node.sub_model.calc_log_marginal_likelihood()
        node.sub_log_marginal_likelihood = node.sub_model.calc_log_marginal_likelihood() # REVIEW: データ数が0の場合にエラーが出るかもしれない．
        return
    def _get_data_indices_node(
            self,
            data_region: List,
            x_continuous:ArrayLike,
            x_categorical:ArrayLike,
        ):
        """
        現在も使用される方法：根ノードからの全ての分岐ルールを毎回適用してデータインデックスを取得
        
        注意：この関数は学習時（build関数など）で使用されており、予測時の最適化とは
        異なる目的で使用されています。予測時の最適化版は以下の関数を参照：
        - _make_prediction_original: この関数と同等のロジックを使用（比較用）
        - _make_prediction_recursive: 最適化された再帰的実装
        - _make_prediction_iterative: 最適化された反復的実装
        
        Parameters
        ----------
        data_region : List
            根ノードから現在のノードまでの分岐ルールのリスト
        x_continuous : ArrayLike
            連続特徴量データ
        x_categorical : ArrayLike
            カテゴリカル特徴量データ
            
        Returns
        -------
        sample_indices : ArrayLike
            該当ノードに割り当てられるサンプルのブールマスク
        num_samples : int
            該当ノードに割り当てられるサンプル数
        """
        sample_indices =  np.full(x_continuous.shape[0], True) # np.onesよりも速い

        for reg in data_region:
            # 親ノードでの分岐ルールを'全て'の学習データに適用し，該当ノードに割り当てられる学習データのインダイスを得る．
            # TODO: 分岐ルールの適用数をすでに割り当てた分岐ルールによって減らせるか？ → _make_prediction_recursive関数で最適化済み
            if reg[0][1] is None: # threshold is None -> categorical
                sample_indices = sample_indices * (x_categorical[:,reg[0][0]-self.c_dim_continuous] == reg[1])
            else: # continuous
                sample_indices = sample_indices * ((x_continuous[:,reg[0][0]] > reg[0][1]) == reg[1])
        num_samples = np.count_nonzero(sample_indices)
        return sample_indices, num_samples
    # def _get_data_indices_node_for_children(
    #         self,
    #         data_region: List,
    #         x_continuous:ArrayLike,
    #         x_categorical:ArrayLike,
    #     ):
    #     sample_indices =  np.full(x_continuous.shape[0], True) # np.onesよりも速い
    #     for reg in data_region:
    #         # 親ノードでの分岐ルールを'全て'の学習データに適用し，該当ノードに割り当てられる学習データのインダイスを得る．
    #         # TODO: 分岐ルールの適用数をすでに割り当てた分岐ルールによって減らせるか？ ただ，現状でもnp.arrayでの計算で済むのでボトルネックにはならなそう．
    #         if reg[0][1] is None: # threshold is None -> categorical
    #             sample_indices = sample_indices * (x_categorical[:,reg[0][0]-self.c_dim_continuous] == reg[1])
    #         else: # continuous
    #             sample_indices = sample_indices * ((x_continuous[:,reg[0][0]] >= reg[0][1]) == reg[1])
    #     num_samples = np.count_nonzero(sample_indices)
    #     return sample_indices, num_samples
    def _node_get_sub_data(
            self,
            node: _LearnNode,
            x_continuous_vecs:ArrayLike,
            x_categorical_vecs:ArrayLike,
            y_vec:ArrayLike,
        ): # ノードごとに学習データ保持するのはメモリ的に非効率．この関数は使わないでindiceのみを保持．
        node.sub_x_continuous_vecs = x_continuous_vecs[node.sample_indices]
        node.sub_x_categorical_vecs = x_categorical_vecs[node.sample_indices]
        node.sub_y_vec = y_vec[node.sample_indices]
        return
    
    ############################################################
    ## build tree
    ############################################################
    def build(
            self,
            x_continuous_vecs:ArrayLike,
            x_categorical_vecs:ArrayLike,
            y_vec: np.ndarray, 
            split_strategy: str, # 'random', 'best', 'copy_from_sklearn_tree', 'copy_from_xgb', or 'given_rule'
            criterion: str, # 'squared_error_leaf' or 'xgb_gain_leaf' or 'marginal_likelihood'
            sample_weight:Optional[ArrayLike] = None, 
            building_scheme: str = 'depth_first', # 'depth_first', 'breadth_first'
            ignore_check: bool = False,
            max_leaf_nodes: Optional[int] = None,
            max_features: Union[int,str,None]=None,
            sklearn_tree: Optional[Any]=None,
            given_split_rules: Optional[Dict]=None,
            xgb_tree_info: Optional[Dict]=None,
            ):
        if ignore_check:
            self.x_continuous_vecs, self.x_categorical_vecs, self.y_vec = x_continuous_vecs, x_categorical_vecs, y_vec
        else:
            self.x_continuous_vecs, self.x_categorical_vecs, self.y_vec = self._check_sample(x_continuous_vecs,x_categorical_vecs,y_vec)
        if sample_weight is None:
            self.sample_weight = sample_weight
        else:
            self.sample_weight = _check.float_vec(sample_weight, 'sample_weight', ParameterFormatError)
        self.max_features = max_features
        if max_leaf_nodes is None:
            self.num_leaf_nodes_rest = -1
        else:
            self.num_leaf_nodes_rest = _check.int_(max_leaf_nodes,'max_leaf_nodes',ParameterFormatError) - 1

        self.given_split_rules = given_split_rules
        # パラメータ入力値 None の処理
        # if (random_state is None) or type(random_state) == int:
        #     rng = np.random.default_rng(random_state)
        # else:
        #     rng = random_state
        # if h0_split is None:
        #     h0_split = self.h0_split
        # if (type(max_features) is int) or (type(max_features) is None):
        #     max_features_int = max_features
        # elif type(max_features) is float:
        #     max_features_int = int(self.c_dim_features * _check.float_in_closed01(max_features,'max_features',ParameterFormatError)) # TODO: エラー文を専用のものに修正
        # elif type(max_features) is str:
        #     if max_features == 'sqrt':
        #         max_features_int = int(np.sqrt(self.c_dim_features))
        #     elif max_features == 'log2': # TODO: 本当はlog2+1だから名前変えたほうが良いかも
        #         max_features_int = int(np.log2(self.c_dim_features)) + 1\
        # # TODO: x_train_mat, y_train_vec, criterion, max_featuresのerror判定
        # if self.threshold_candidates is None:
        #     self.threshold_candidate = self._generate_threshold()
        
        if split_strategy == 'copy_from_sklearn_tree':
            if sklearn_tree is None:
                raise(ParameterFormatError('split_strategy is "copy_from_sklearn_tree", but sklearn_tree is None.'))
            if type(sklearn_tree) not in [DecisionTreeClassifier, DecisionTreeRegressor]:
                raise(ParameterFormatError('sklearn_tree must be DecisionTreeClassifier or DecisionTreeRegressor.'))
            self._copy_tree_from_sklearn_tree(original_tree=sklearn_tree)
            self._update_posterior_submodel(
                self.nodes_list,
                self.x_continuous_vecs,
                self.x_categorical_vecs,
                self.y_vec,
                reset_param=True)
            self.nodes_list_sorted_by_depth = sorted(self.nodes_list, key=(lambda node: node.c_depth), reverse=True)
            self.dict_node_id = defaultdict()
            for node in self.nodes_list:
                self.dict_node_id[node.c_id] = node
            self._update_posterior_hn_split(self.nodes_list_sorted_by_depth)
            return
        elif split_strategy == 'copy_from_xgb':
            if xgb_tree_info is None:
                raise(ParameterFormatError('split_strategy is "copy_from_xgb", but xgb_tree_info_list is None.'))
            # if type(xgb_tree_info_list) is not list:
            #     raise(ParameterFormatError('xgb_tree_info_list must be a list.'))
            self._copy_singletree_from_xgb(tree_info=xgb_tree_info)
            self._update_posterior_submodel(
                self.nodes_list,
                self.x_continuous_vecs,
                self.x_categorical_vecs,
                self.y_vec,
                reset_param=True)
            self.nodes_list_sorted_by_depth = sorted(self.nodes_list, key=(lambda node: node.c_depth), reverse=True)
            self.dict_node_id = defaultdict()
            for node in self.nodes_list:
                self.dict_node_id[node.c_id] = node
            self._update_posterior_hn_split(self.nodes_list_sorted_by_depth)
            return
        elif split_strategy == 'given_rule':
            if given_split_rules is None:
                raise(ParameterFormatError('split_strategy is "given_rule", but given_split_rules is None.'))
        elif split_strategy == 'random':
            pass
        elif split_strategy == 'best':
            pass
        else:
            raise(ParameterFormatError(f'split_strategy:{split_strategy} is not yet supported.'))


        # append data into root node
        self._update_posterior_submodel_node(self.root_node, x_continuous_vecs, x_categorical_vecs, y_vec,reset_param=True)
        if building_scheme == 'depth_first':
            self._build_depth_first(
                split_strategy,
                # rng = rng,
                # x_train_mat = x_train_mat,
                # y_train_vec = y_train_vec,
                criterion = criterion,
                # splitter = splitter,
                # max_depth = max_depth,
                # min_samples_split = min_samples_split,
                # min_samples_leaf = min_samples_leaf,
                # max_features_int = max_features_int,
                # max_leaf_nodes = max_leaf_nodes,
                # min_criteria_decrease = min_criteria_decrease,
                # h0_split = h0_split,
                )
        elif building_scheme == 'breadth_first':
            self._build_breadth_first(
                split_strategy,
                # rng = rng,
                # x_train_mat = x_train_mat,
                # y_train_vec = y_train_vec,
                criterion = criterion,
                # splitter = splitter,
                # max_depth = max_depth,
                # min_samples_split = min_samples_split,
                # min_samples_leaf = min_samples_leaf,
                # max_features_int = max_features_int,
                # max_leaf_nodes = max_leaf_nodes,
                # min_criteria_decrease = min_criteria_decrease,
                # h0_split = h0_split,
                )
        else:
            raise(ParameterFormatError(f'building_scheme:{building_scheme} is not yet supported.'))
        self.nodes_list_sorted_by_depth = sorted(self.nodes_list, key=(lambda node: node.c_depth), reverse=True)
        self.dict_node_id = defaultdict()
        for node in self.nodes_list:
            self.dict_node_id[node.c_id] = node
        self._update_posterior_submodel(
            self.nodes_list,
            self.x_continuous_vecs,
            self.x_categorical_vecs,
            self.y_vec,
            reset_param=True)
        self._update_posterior_hn_split(self.nodes_list_sorted_by_depth)
        return

    def _generate_threshold(self, x_continuous_vecs):
        if type(self.threshold_params['num_thresholds']) is int:
            _num_thresholds = self.threshold_params['num_thresholds']
        elif self.threshold_params['num_thresholds'] == 'log2':
            _num_thresholds = int(np.log2(x_continuous_vecs.shape[0])+1)
        elif self.threshold_params['num_thresholds'] == 'sqrt':
            _num_thresholds = int(np.sqrt(x_continuous_vecs.shape[0]))
        else:
            _num_thresholds = x_continuous_vecs.shape[0]
        threshold_candidates = [None for _ in range(self.c_dim_continuous)]
        # if x_continuous_vecs.shape[0] <= _num_thresholds: # NOTE: 書いていて気づいたが，featureごとにall_thresholdsをするよりはx_continuous_vecs.Tするほうが速そう←試してみたら誤差レベルだった
        #     for feature in range(self.c_dim_continuous):
        #         threshold_candidates[feature] = self.threshold_generator.all_thresholds(x_continuous_vecs, feature, None)
        #     return threshold_candidates
        if self.threshold_params['type'] == 'all' or self.threshold_params['type'] is None:
            func = self.threshold_generator.all_thresholds
        elif self.threshold_params['type'] == 'hist':
            func = self.threshold_generator.by_histogram
        elif self.threshold_params['type'] == 'quantile':
            func = self.threshold_generator.by_quantile
        elif self.threshold_params['type'] == 'random':
            func = self.threshold_generator.by_random
        else:
            threshold_type = self.threshold_params['type']
            raise(ParameterFormatError(f'threshold_params[\'type\']: {threshold_type} is not supported. \
                                       It only allows \'all\', \'hist\', \'quantile\', and \'random\'.'))
        for feature in range(self.c_dim_continuous):
            threshold_candidates[feature] = func(x_continuous_vecs, feature, _num_thresholds)
        return threshold_candidates
    
    # NOTE: ノード数の制約が未実装なので，現状では探索順に意味はない．
    def _build_depth_first(
            self,
            split_strategy: str,
            # rng: Generator,
            # x_train_mat: ArrayLike,
            # y_train_vec: ArrayLike,
            criterion: str,
            # splitter: str,
            # max_depth: Optional[int],
            # min_samples_split: int,
            # min_samples_leaf: int,
            # max_features_int: int or float or Optional[str],
            # min_criteria_decrease: Optional[float],
            ): # TODO: enable these options
        leaf_nodes_to_split = deque(self.leaf_nodes_list)
        leaf_nodes_end = []
        while bool(leaf_nodes_to_split) & (self.num_leaf_nodes_rest!=0): # repeat while leaf_nodes are not empty
            node = leaf_nodes_to_split.pop() # pop top-right node (latest, stack)
            if node.c_depth == self.c_max_depth:
                node.is_leaf = True
            else:
                node.is_leaf = False
            if node.is_leaf: # stop splitting and go to next node in leaf_nodes_to_split
                leaf_nodes_end.append(node)
                continue
            val = self._split_node(node,self.x_continuous_vecs[node.sample_indices],self.x_categorical_vecs[node.sample_indices],self.y_vec[node.sample_indices],split_strategy,criterion)
            if val == -1:
                node.is_leaf = True
                leaf_nodes_end.append(node)
            else:
                self.inner_nodes_list.append(node)
            for child in node.children: # 左側のノード(インデックス番号小)から先に降りる（過去の実装と同様）になるように逆順にleaf_nodesに追加
                # if node.children is empty, i.e. node is a leaf_node, this loop won't be processed  
                leaf_nodes_to_split.append(child) # add children to top right in the que
                self.nodes_list.append(child)
            if node.is_leaf == False:
                self.num_leaf_nodes_rest -= 1
            
        leaf_nodes_end = leaf_nodes_end + list(leaf_nodes_to_split)
        self.leaf_nodes_list = list(leaf_nodes_end)
        return

    def _build_breadth_first(
            self,
            split_strategy: str,
            # rng: Generator,
            # x_train_mat: ArrayLike,
            # y_train_vec: ArrayLike,
            criterion: str,
            # splitter: str,
            # max_depth: Optional[int],
            # min_samples_split: int,
            # min_samples_leaf: int,
            # max_features_int: int or float or Optional[str],
            # min_criteria_decrease: Optional[float],
            ): # TODO: enable these options
        leaf_nodes_to_split = deque(self.leaf_nodes_list)
        leaf_nodes_end = []
        while bool(leaf_nodes_to_split) & (self.num_leaf_nodes_rest!=0): # repeat while leaf_nodes are not empty
            node = leaf_nodes_to_split.popleft() # pop top-left node (oldest, que)
            if node.c_depth == self.c_max_depth:
                node.is_leaf = True
            else:
                node.is_leaf = False
            if node.is_leaf: # stop splitting and go to next node in leaf_nodes_to_split
                leaf_nodes_end.append(node)
                continue
            val = self._split_node(node,self.x_continuous_vecs[node.sample_indices],self.x_categorical_vecs[node.sample_indices],self.y_vec[node.sample_indices],split_strategy,criterion)
            if val == -1:
                node.is_leaf = True
                leaf_nodes_end.append(node)
            else:
                self.inner_nodes_list.append(node)
            for child in node.children: # 左側のノード(インデックス番号小)から先に降りる（過去の実装と同様）になるように逆順にleaf_nodesに追加
                # if node.children is empty, i.e. node is a leaf_node, this loop won't be processed  
                leaf_nodes_to_split.append(child) # add children to top right in the que
                self.nodes_list.append(child)
            if node.is_leaf == False:
                self.num_leaf_nodes_rest -= 1
        self.leaf_nodes_list = list(leaf_nodes_end)
        return
    

    ############################################################
    ## split a single node
    ############################################################
    def _node_split_rule_random(
            self,
            node: _LearnNode,
            sub_x_continuous_vecs:ArrayLike,
            sub_x_categorical_vecs:ArrayLike,
            y_vec: ArrayLike,
        ): # TODO: 同じ連続特徴量での分岐が複数回起こる場合に意味のない分岐（絶対に訪れることのない葉ノード）が発生する.

        tmp = [self.h0_feature_weight_vec[i] for i in node.c_feature_candidates]
        prob_feature = tmp / np.sum(tmp)
        feature = self.rng.choice(a=node.c_feature_candidates, p=prob_feature)
        if feature < self.c_dim_continuous:
            # 学習データの範囲内で閾値を取る．
            feature_data = sub_x_continuous_vecs[:,node._split_rule[0]]
            if feature_data.size == 0:
                return None, None
            threshold = self.rng.uniform(feature_data.min(), feature_data.max())
        else:
            threshold = None
        return (feature, threshold)

    def _node_split_rule_best(
            self,
            node: _LearnNode,
            criterion: str,
            # sub_x_continuous_vecs:ArrayLike,
            # sub_x_categorical_vecs:ArrayLike,
            # y_vec: ArrayLike,
        ): # TODO: 同じ連続特徴量での分岐が複数回起こる場合に意味のない分岐（絶対に訪れることのない葉ノード）が発生する.

        if node.sample_indices.sum() == 1:
            return (None,None), None
        
        sub_x_continuous_vecs = self.x_continuous_vecs[node.sample_indices]
        sub_x_categorical_vecs = self.x_categorical_vecs[node.sample_indices]
        sub_y_vec = self.y_vec[node.sample_indices]
        if self.sample_weight is None:
            sub_sample_weight = np.ones(sub_y_vec.shape[0])
        else:
            sub_sample_weight = self.sample_weight[node.sample_indices]

        h0_split_child = 0. if node.c_depth+1 == self.c_max_depth else self.h0_split_list[node.c_depth]
        split_candidates = self._get_split_candidates(node)

        best_split = (None,None)
        best_val = -np.inf
        node_criterion = self._calc_criterion_node(node, criterion, sub_y_vec, sub_sample_weight)

        # children_indices = [[] for _ in range(np.max(self.c_num_children_vec))]
        for rule in split_candidates: # rule[0] -> feature, rule[1] -> threshold
            val = self._calc_gain(
                rule,
                node,
                node_criterion,
                sub_x_continuous_vecs,
                sub_x_categorical_vecs,
                criterion,
                sub_y_vec,
                sub_sample_weight,
            )
            if val > best_val:
                best_val = val
                best_split = rule
        node_id = node.c_id
        return best_split, best_val

    def _get_split_candidates(
            self,
            node: _LearnNode,
        ):
        split_candidates = [] # TODO append遅いかも．先にNoneとか定義しておけないか？二重配列になるのは避けたいが...
        if self.max_features is not None:
            node.feature_candidates_considered =  _feature_subsample(node.c_feature_candidates,self.max_features,len(node.c_feature_candidates),self.rng)
        else:
            node.feature_candidates_considered = node.c_feature_candidates
        for feature in node.feature_candidates_considered:
            if feature < self.c_dim_continuous: # continuous
                for threshold in node.threshold_candidates[feature]: # TODO: mapとか使えるとfor文減らせてかっこいい
                    split_candidates.append((feature, threshold))
                pass
            else: # categorical
                split_candidates.append((feature, None))
        return split_candidates
    
    def _calc_gain_num_children3(
            self,
            rule: List,
            node,
            node_criterion,
            sub_x_continuous_vecs,
            sub_x_categorical_vecs,
            criterion,
            sub_y_vec,
            sub_sample_weight,
            ):
        num_children = self.c_num_children_vec[rule[0]]
        children_criterion = 0
        for i in range(num_children): 
            data_region = deepcopy(node.c_data_region)
            data_region.append((rule, i))
            child_indices, num_samples_children = self._get_data_indices_node(data_region, sub_x_continuous_vecs, sub_x_categorical_vecs)
            if num_samples_children == 0:
                continue
            else:
                children_criterion += self._calc_criterion_node(node, criterion, sub_y_vec[child_indices], sub_sample_weight[child_indices])
        return node_criterion - children_criterion
    
    def _calc_gain(
            self,
            rule: List,
            node,
            node_criterion,
            sub_x_continuous_vecs,
            sub_x_categorical_vecs,
            criterion,
            sub_y_vec,
            sub_sample_weight,
            ):
        l_indice = self._get_left_child_mask(rule,sub_x_continuous_vecs,sub_x_categorical_vecs,)
        r_indice = ~l_indice
        children_criterion = self._calc_criterion_node(node, criterion, sub_y_vec[l_indice], sub_sample_weight[l_indice])\
                                + self._calc_criterion_node(node, criterion, sub_y_vec[r_indice], sub_sample_weight[r_indice])
        if criterion=='squared_error_leaf':
            return node_criterion - children_criterion
        elif criterion == 'xgb_gain_leaf':
            return (children_criterion - node_criterion) / 2 + self.gamma_xgb
        elif criterion == 'marginal_likelihood':
            return children_criterion
        
    def _get_left_child_mask(
            self,
            rule,
            sub_x_continuous_vecs,
            sub_x_categorical_vecs,
    ):
        if rule[1] is None: # threshold is None -> categorical
            return (sub_x_categorical_vecs[:,rule[0]-self.c_dim_continuous] == 0)
        else:
            return (sub_x_continuous_vecs[:,rule[0]] <= rule[1])

    def _calc_criterion_node(
            self, 
            node: _LearnNode, 
            criterion: str,
            y_vec: ArrayLike,
            sample_weight: ArrayLike,
        ):
        if criterion == 'squared_error_leaf':
            val = _squared_error_with_weight(y_vec,np.mean(y_vec),sample_weight)
        elif criterion == 'squared_error_meta':
            val = self._calc_squared_error_meta(node,y_vec,sample_weight)
        elif criterion == 'xgb_gain_leaf':
            val = _calc_xgb_impurity_with_weight(y_vec,np.mean(y_vec),sample_weight, self.lambda_xgb)
        elif criterion == 'marginal_likelihood':
            node.sub_model._update_posterior(y_vec)
            node.sub_model.calc_pred_dist()
            val = node.sub_model.calc_log_marginal_likelihood()
            node.sub_model.reset_hn_params()
        else:
            raise(ParameterFormatError(f'Criterion[{criterion}] is not supported.'))
        return val
    
    def _calc_squared_error_meta(
            self,
            node: _LearnNode, 
            y_vec: ArrayLike,
            sample_weight: ArrayLike,
        ):
        raise(NotImplementedError)
    
    def _calc_criterion_children(
            self, 
            node: _LearnNode, 
            criterion: str,
            children_indices: List,
            y_vec: ArrayLike,
            sample_weight: ArrayLike,
        ):
        val = 0.
        if criterion == 'squared_error_leaf':
            if sample_weight is None:
                for indice in children_indices:
                    val += 1/y_vec.shape[0] * ((y_vec[indice]-np.mean(y_vec[indice]))**2).sum()
            else:
                for indice in children_indices:
                    val += 1/y_vec.shape[0] * ((y_vec[indice]-np.mean(y_vec[indice]))**2).sum()
        else:
            raise(ParameterFormatError(f'Criterion[{criterion}] is not supported.'))
        return val
    

    def _split_node(
            self,
            node: _LearnNode,
            sub_x_continuous_vecs:ArrayLike,
            sub_x_categorical_vecs:ArrayLike,
            sub_y_vec: ArrayLike,
            split_strategy: str,
            criterion: str,
        ): 
        node.threshold_candidates = self._generate_threshold(sub_x_continuous_vecs)
        if split_strategy == 'random':
            split_rule = self._node_split_rule_random(node,sub_x_continuous_vecs,sub_x_categorical_vecs,sub_y_vec)
            if split_rule[0] is None: # Node cannot be split because sample is concentrated in this node.
                return -1
            node._split_rule[0] = split_rule[0]
            node._split_rule[1] = split_rule[1]
        elif split_strategy == 'best':
            split_rule, criterion_value = self._node_split_rule_best(node, criterion)
            if split_rule[0] is None: # No gain obtained by the split
                # self._update_posterior_submodel_node(node, sub_x_continuous_vecs, sub_x_categorical_vecs, sub_y_vec) # 子ノードでインダイス取得処理も行うため，全データを入力する必要あり?
                # self._update_posterior_hn_split_node(node) # XXX: サブモデルを複数回同じデータで更新していたためバグが存在．一旦コメントアウトしたがそのうち修正する
                return -1
            node._split_rule[0] = split_rule[0]
            node._split_rule[1] = split_rule[1]
            node.criterion_value = criterion_value
        elif split_strategy == 'given_rule':
            if self.given_split_rules == None:
                raise(ParameterFormatError('When split_strategy is \'given_rule\', given_split_rules must be passed.'))
            else:
                split_rule = self.given_split_rules[node.c_id]
                node._split_rule[0] = split_rule[0]
                node._split_rule[1] = split_rule[1]
        else:
            raise(ParameterFormatError(f'split_strategy:{split_strategy} is not yet supported. Please choose from \'random\', \'best\' or \'given_rule\'.'))
        # eliminate feature from candidate list
        c_num_assignment_vec_child = deepcopy(node.c_num_assignment_vec)
        c_num_assignment_vec_child[node._split_rule[0]] = node.c_num_assignment_vec[node._split_rule[0]] - 1
        if c_num_assignment_vec_child[node._split_rule[0]] == 0: # node._split_rule[0] won't be used more.
            feature_candidate_child = [item for item in node.c_feature_candidates if item != node._split_rule[0]]
        else:
            feature_candidate_child = node.c_feature_candidates
        h0_split_child = 0. if node.c_depth+1 == self.c_max_depth else self.h0_split_list[node.c_depth]

        num_children = self.c_num_children_vec[node._split_rule[0]]
        node.children = [None for _ in range(num_children)]
        c_data_region_child = [(node._split_rule, i) for i in range(num_children)]
        # NOTE: threshold周りの記述を削除，ノードごとにthresholdを計算するように変更
        # if node._split_rule[1] is not None: # continuous
        #     thresholds_tobe_sliced = node.threshold_candidates[node._split_rule[0]]

        for i in range(num_children):
            # threshold_candidates_child = deepcopy(node.threshold_candidates) # NOTE: メモリ共有の問題が起きるのでdeepcopyしておく．遅くなるかもしれないけど...
            # if node._split_rule[1] is not None: # continuous TODO: カテゴリカルの場合にも閾値除去を行うほうが速い？
            #     threshold_candidates_child[node._split_rule[0]] = thresholds_tobe_sliced[(thresholds_tobe_sliced >= node._split_rule[1]) == i]
            node.children[i] = _LearnNode(
                self.rng,
                self.SubModel,
                c_id = node.c_id + str(i),
                c_depth = node.c_depth + 1,
                h0_split = h0_split_child,
                c_ancestor_nodes = node.c_ancestor_nodes + [node],
                num_samples = None, # TODO: to be changed
                sample_indices = None, # TODO: to be changed
                is_leaf = True,
                c_feature_candidates = feature_candidate_child,
                c_num_assignment_vec = c_num_assignment_vec_child,
                c_data_region = node.c_data_region + [c_data_region_child[i]],
                children = [],
                feature = None,
                threshold = None,
                h0_constants_SubModel = self.h0_constants_SubModel,
                sub_h0_params = self.sub_h0_params,
                threshold_candidates = None,
            )
        # update [submodel for children] and [hn_split for path to the children].
        path_children = list(reversed(node.c_ancestor_nodes + [node] + node.children)) # 葉ノードから更新するために逆順にする．
        self._update_posterior_submodel(node.children, self.x_continuous_vecs, self.x_categorical_vecs, self.y_vec, reset_param=True) # NOTE: reset_paramsを忘れていて更新回数を間違えた，必ずTrueにする
        # 子ノードでインダイス取得処理も行うため，全データを入力する必要あり．
        # self._update_posterior_hn_split(path_children)
        return 0

    ###################################################
    # marginal likelihood calculation
    ###################################################
    def calc_tree_log_marginal_likelihood(
            self,
            ) -> float:
        if self.SubModel is linearregression: # TODO: define calc_tree_log_marginal_likelihood_lr
            raise(ParameterFormatError('linearregression is not currently supported.'))
        else:
            pass
        for node in self.nodes_list_sorted_by_depth:
            if node.is_leaf:
                node._log_marginal_likelihood_hn = node.sub_log_marginal_likelihood
                node.hn_split = 0.
            else:
                log_marginal_likelihood_children = sum(map(lambda child: child._log_marginal_likelihood_hn, node.children))
                tmp1 = np.log(node.hn_split) + log_marginal_likelihood_children
                tmp2 = np.log(1-node.hn_split) + node.sub_log_marginal_likelihood
                node._log_marginal_likelihood_hn = np.logaddexp(tmp1,tmp2)
        return self.root_node._log_marginal_likelihood_hn
    
    ###################################################
    # copy tree structure from other packages (sklearn.tree, XGB, LightGBM)
    ###################################################
    def _copy_tree_from_sklearn_tree(self, original_tree:Any):
        if all(num_children == 2 for num_children in self.c_num_children_vec): # 2分木でない場合はエラー
            pass
        else:
            ind_val = next(((index, num_children) for index, num_children in enumerate(self.c_num_children_vec) if num_children != 2), None)
            raise(ValueError(f'_copy_tree_from_sklearn_tree() only supports 2-ary tree but c_num_children_vec[{ind_val[0]}] is {ind_val[1]}.'))
        self.leaf_nodes_list[0].c_id_sklearn_tree = 0
        leaf_nodes_to_split = deque(self.leaf_nodes_list)
        leaf_nodes_end = []
        while bool(leaf_nodes_to_split): # repeat while leaf_nodes are not empty
            node = leaf_nodes_to_split.pop() # pop top-right node (latest, stack)
            self.copy_node_from_sklearn_tree(node, original_tree)
            if node.is_leaf == True:
                leaf_nodes_end.append(node)
            else:
                self.inner_nodes_list.append(node)
            for child in node.children: # 左側のノード(インデックス番号小)から先に降りる（過去の実装と同様）になるように逆順にleaf_nodesに追加
                # if node.children is empty, i.e. node is a leaf_node, this loop won't be processed  
                leaf_nodes_to_split.append(child) # add children to top right in the que
                self.nodes_list.append(child)
            
        leaf_nodes_end = leaf_nodes_end + list(leaf_nodes_to_split)
        self.leaf_nodes_list = list(leaf_nodes_end)

        return

    def copy_node_from_sklearn_tree(
            self, 
            node: _LearnNode,
            original_tree: Any,
            ):
        # 分岐ルールを貼り付け
        if original_tree.tree_.feature[node.c_id_sklearn_tree] < self.c_dim_continuous:
            node._split_rule = [original_tree.tree_.feature[node.c_id_sklearn_tree], original_tree.tree_.threshold[node.c_id_sklearn_tree]]
        else:
            node._split_rule = [original_tree.tree_.feature[node.c_id_sklearn_tree], None]
        if original_tree.tree_.children_left[node.c_id_sklearn_tree] == sklearn.tree._tree.TREE_LEAF: # 葉ノード
            node.is_leaf = True
            node.children = []
        else: # 内部ノード
            # 子ノードを作成
            node.is_leaf = False
            num_children = self.c_num_children_vec[node._split_rule[0]]
            c_data_region_child = [(node._split_rule, i) for i in range(num_children)]
            c_num_assignment_vec_child = deepcopy(node.c_num_assignment_vec)
            c_num_assignment_vec_child[node._split_rule[0]] = node.c_num_assignment_vec[node._split_rule[0]] - 1
            if c_num_assignment_vec_child[node._split_rule[0]] == 0: # node._split_rule[0] won't be used more.
                feature_candidate_child = [item for item in node.c_feature_candidates if item != node._split_rule[0]]
            else:
                feature_candidate_child = node.c_feature_candidates
            h0_split_child = 0. if node.c_depth+1 == self.c_max_depth else self.h0_split_list[node.c_depth]
            node.children = [
                _LearnNode(
                    self.rng,
                    self.SubModel,
                    c_id = node.c_id + str(i),
                    c_depth = node.c_depth + 1,
                    h0_split = h0_split_child,
                    c_ancestor_nodes = node.c_ancestor_nodes + [node],
                    num_samples = None, # TODO: to be changed
                    sample_indices = None, # TODO: to be changed
                    is_leaf = True,
                    c_feature_candidates = feature_candidate_child,
                    c_num_assignment_vec = c_num_assignment_vec_child,
                    c_data_region = node.c_data_region + [c_data_region_child[i]],
                    children = [],
                    feature = None,
                    threshold = None,
                    h0_constants_SubModel = self.h0_constants_SubModel,
                    sub_h0_params = self.sub_h0_params,
                    threshold_candidates = None,
                ) for i in range(2)
            ]
            # sklearnのノードidを子ノードに対応付ける
            node.children[0].c_id_sklearn_tree = original_tree.tree_.children_left[node.c_id_sklearn_tree]
            node.children[1].c_id_sklearn_tree = original_tree.tree_.children_right[node.c_id_sklearn_tree]
        return
    
    def _copy_singletree_from_xgb(self, tree_info:dict):
        if all(num_children == 2 for num_children in self.c_num_children_vec): # 2分木でない場合はエラー
            pass
        else:
            ind_val = next(((index, num_children) for index, num_children in enumerate(self.c_num_children_vec) if num_children != 2), None)
            raise(ValueError(f'_copy_singletree_from_xgb() only supports 2-ary tree but c_num_children_vec[{ind_val[0]}] is {ind_val[1]}.'))
        leaf_nodes_to_split = deque(self.leaf_nodes_list)
        leaf_nodes_end = []
        while bool(leaf_nodes_to_split): # repeat while leaf_nodes are not empty
            node = leaf_nodes_to_split.pop() # pop top-right node (latest, stack)
            self.copy_node_from_xgb_tree(node, tree_info)
            if node.is_leaf == True:
                leaf_nodes_end.append(node)
            else:
                self.inner_nodes_list.append(node)
            for child in node.children: # 左側のノード(インデックス番号小)から先に降りる（過去の実装と同様）になるように逆順にleaf_nodesに追加
                # if node.children is empty, i.e. node is a leaf_node, this loop won't be processed  
                leaf_nodes_to_split.append(child) # add children to top right in the que
                self.nodes_list.append(child)
            
        leaf_nodes_end = leaf_nodes_end + list(leaf_nodes_to_split)
        self.leaf_nodes_list = list(leaf_nodes_end)

        return
    
    def copy_node_from_xgb_tree(
            self,
            node: _LearnNode,
            tree_info: Dict,
            ):
        # 分岐ルールを抽出
        if tree_info[node.c_id]['leaf'] is not None: # 葉ノード
            node.is_leaf = True
        else: # 内部ノード
            node.is_leaf = False
            feature = tree_info[node.c_id]['feature']
            if feature < self.c_dim_continuous:
                threshold = tree_info[node.c_id]['threshold']
            else:
                threshold = None
        # 分岐ルールを貼り付け
        if node.is_leaf:
            node.children = []
        else:
            node._split_rule = [feature, threshold]
            # 子ノードを作成
            num_children = self.c_num_children_vec[node._split_rule[0]]
            c_data_region_child = [(node._split_rule, i) for i in range(num_children)]
            c_num_assignment_vec_child = deepcopy(node.c_num_assignment_vec)
            c_num_assignment_vec_child[node._split_rule[0]] = node.c_num_assignment_vec[node._split_rule[0]] - 1
            if c_num_assignment_vec_child[node._split_rule[0]] == 0: # node._split_rule[0] won't be used more.
                feature_candidate_child = [item for item in node.c_feature_candidates if item != node._split_rule[0]]
            else:
                feature_candidate_child = node.c_feature_candidates
            h0_split_child = 0. if node.c_depth+1 == self.c_max_depth else self.h0_split_list[node.c_depth]
            node.children = [
                _LearnNode(
                    self.rng,
                    self.SubModel,
                    c_id = node.c_id + str(i),
                    c_depth = node.c_depth + 1,
                    h0_split = h0_split_child,
                    c_ancestor_nodes = node.c_ancestor_nodes + [node],
                    num_samples = None, # TODO: to be changed
                    sample_indices = None, # TODO: to be changed
                    is_leaf = True,
                    c_feature_candidates = feature_candidate_child,
                    c_num_assignment_vec = c_num_assignment_vec_child,
                    c_data_region = node.c_data_region + [c_data_region_child[i]],
                    children = [],
                    feature = None,
                    threshold = None,
                    h0_constants_SubModel = self.h0_constants_SubModel,
                    sub_h0_params = self.sub_h0_params,
                    threshold_candidates = None,
                ) for i in range(2)
            ]
        return

