# from __future__ import annotations

import numpy as np
from copy import deepcopy
from collections import deque, defaultdict
# from heapq import heapify, heappop, heappush

import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex

from typing import Dict, List, Tuple, Union, Optional, Any
import numpy as np

from .. import _check
from .. import base
from .. import bernoulli, categorical, normal, linearregression
from .._exceptions import (CriteriaError, DataFormatError, ParameterFormatError,
                         ParameterFormatWarning, ResultWarning)

from . import _constants
_CMAP = _constants._CMAP

MODELS = _constants.MODELS
DISCRETE_MODELS = _constants.DISCRETE_MODELS
CONTINUOUS_MODELS = _constants.CONTINUOUS_MODELS
CLF_MODELS = _constants.CLF_MODELS
REG_MODELS = _constants.REG_MODELS

class _GenNode:
    def __init__(
            self,
            rng: np.random.Generator,
            # SubModel: Any, # TODO: Type annotation for submodel class
            c_id: str, # ID with binary index. Root node has index with empty str ''.
            c_depth: int,
            h_split: float, # <- h_g
            c_ancestor_nodes: list, # List of ancestor nodes. Root node should have [] for this list.
            is_leaf: bool,
            c_feature_candidates: List, # <- k_candidates
            c_num_assignment_vec: List, 
            c_data_region: List,
            children: Optional[List['_GenNode']] = None,
            feature: Optional[int] = None, # <- k
            threshold: Optional[float] = None,
            h_constants_SubModel: Optional[Dict] = {},
            sub_h_params: Optional[Dict] = {},
            sub_params: Optional[Dict] = {},
            ):
        
        # Static valuables which should not be updated
        self.c_id = c_id
        self.c_depth = c_depth
        self.c_ancestor_nodes = c_ancestor_nodes
        self.h_split = h_split

        # valuables which can be updated
        self.is_leaf = is_leaf
        self.c_feature_candidates = c_feature_candidates
        self.c_num_assignment_vec = c_num_assignment_vec

        self.c_data_region = c_data_region

        if children is None:
            self.children = []
        else:
            self.children = children

        self._split_rule = [feature, threshold]
        if h_constants_SubModel is None:
            self.h_constants_SubModel = {}
        else:
            self.h_constants_SubModel = h_constants_SubModel
        if sub_h_params is None:
            self.sub_h_params = {}
        else:
            self.sub_h_params = sub_h_params
        if sub_params is None:
            self.sub_params = {}
        else:
            self.sub_params = sub_params
        self.rng = rng
    # Do not define submodel for all nodes. Define later for leaf nodes.
        self.sub_model = None
    
    def _prepare_sub_model(
            self,
            SubModel: Any, # TODO: Type annotation for submodel class
            # h_constants_SubModel: Optional[Dict] = {},
            # sub_h_params: Optional[Dict] = {},
    ):
        # if h_constants_SubModel is None:
        #     self.h_constants_SubModel = {}
        # else:
        #     self.h_constants_SubModel = h_constants_SubModel
        # if sub_h_params is None:
        #     self.sub_h_params = {}
        # else:
        #     self.sub_h_params = sub_h_params
        self.sub_model = SubModel.GenModel(
                seed=self.rng,
                **self.h_constants_SubModel,
                **self.sub_h_params
                )
        self._gen_params()
        return self

    def _set_h_params(
            self,
            h_split: Optional[int] = None,
            sub_h_params: Optional[Dict] = None,
            ):
        if h_split is not None:
            self.h_split = _check.float_in_closed01(h_split)
        if sub_h_params is not None:
            self.sub_model.set_h_params(**self.sub_h_params)
        return
    
    def _gen_params(
            self,
    ):
        self.sub_model.gen_params()
        return
    def _set_params(
            self,
            sub_params: Optional[Dict] = {},
    ):
        if sub_params is not None:
            self.sub_model.set_params(**sub_params)
        return
    def _gen_sample(
            self,
            sample_size: int,
    ):
        y = self.sub_model.gen_sample(sample_size=sample_size)
        return y
    
    def _index_child_node_given_x(
            self, 
            x_continuous: np.ndarray, 
            x_categorical: np.ndarray, 
            c_dim_continuous: int,
            c_dim_categorical: int,
            ):
        if self._split_rule[0] < c_dim_continuous: # continuous, TODO: Extend to multi-valued thresholds
            return 1 if (x_continuous[self._split_rule[0]] >= self._split_rule[1]) else 0
        else: # categorical
            return x_categorical[self._split_rule[0] - c_dim_continuous]
        
    def _child_node_given_x(
            self, 
            x_continuous: np.ndarray, 
            x_categorical: np.ndarray, 
            c_dim_continuous: int,
            c_dim_categorical: int,
            ):
        index = self._index_child_node_given_x(x_continuous, x_categorical, c_dim_continuous, c_dim_categorical)
        return self.children[index]

class MetaTreeGenModel(base.Generative):
    """ 
    The stochastice data generative model and the prior distribution of MetaTree.
    As a gen model, we use a decision tree model (not a meta-tree).
    
    Parameters
    ----------
    SubModel : class
        bernoulli, poisson, normal, or exponential, 
    c_dim_continuous : int
        A non-negative integer.
    c_dim_categorical : int
        A non-negative integer.
    c_num_children_vec : int or numpy.ndarray, optional
        number of children of a node, when the node is split by the feature.
        A positive integer or a vector of positive integers whose length is 
        ``c_dim_features``, by default 2.
        If a single integer is input it will be broadcasted for every feature.
    c_max_depth : int, optional
        A positive integer, by default 2
    c_ranges : numpy.ndarray, optional
        A numpy.ndarray whose size is (c_dim_continuous,2).
        A threshold for the ``k``-th continuous feature will be 
        generated between ``c_ranges[k,0]`` and ``c_ranges[k,1]``. 
        By default, [[-3,3],[-3,3],...,[-3,3]].
    c_feature_candidates : 'all' or List, optional
        Str or list of possible features for split, by default 'all'
    c_num_assignment_vec : numpy.ndarray, optional
        A vector of positive integers whose length is 
        ``c_dim_continuous+c_dim_categorical``. 
        The first ``c_dim_continuous`` elements represent 
        the maximum assignment numbers of continuous features 
        on a path. The other ``c_dim_categorial`` elements 
        represent those of categorical features.
        By default [c_max_depth,...,c_max_depth,1,...,1].
    h_constants_SubModel : dict, optional
        constants for self.SubModel.LearnModel, by default {}
    h_feature_weight_vec : numpy.ndarray, optional
        A vector of positive real numbers whose length is 
        ``c_dim_continuous+c_dim_categorical``, 
        by default [1/c_num_assignment_vec.sum(),...,1/c_num_assignment_vec.sum()].
    h_split : list or float, optional
        A real number or a vector in :math:`[0, 1]`, by default 0.
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
    h_k_weight_vec : numpy.ndarray
        A vector of positive real numbers whose length is 
        ``c_dim_continuous+c_dim_categorical``
    h_split : float
        A real number in :math:`[0, 1]`
    sub_h_params : dict
        h_params for self.SubModel.GenModel
    """
    def __init__(
            self,
            SubModel: Any, # TODO: Type annotation for submodel class
            c_dim_continuous: int,
            c_dim_categorical: int,
            c_max_depth: int, # TODO: c_max_depth=0 causes error during broadcasting self.h0_split = np.ones(self.c_max_depth,dtype=float), it creates empty array.
            c_ranges: Optional[np.ndarray] = None,
            c_num_children_vec: List or int = 2,
            c_feature_candidates: Optional[List[int]] = None,
            c_num_assignment_vec: Optional[np.ndarray] = None,
            h_constants_SubModel: Dict = {},
            sub_h_params: Dict = {},
            h_split: List or float = 0.,
            h_feature_weight_vec: Optional[np.ndarray] = None,
            seed: Optional[int] = None,
        ):
    
        # Static valuables which should not be updated, from the input
        self.c_dim_continuous = _check.nonneg_int(c_dim_continuous,'c_dim_continuous',ParameterFormatError)
        self.c_dim_categorical = _check.nonneg_int(c_dim_categorical,'c_dim_categorical',ParameterFormatError)
        self.c_dim_features = _check.pos_int(c_dim_continuous+c_dim_categorical,'c_dim_continuous+c_dim_categorical',ParameterFormatError)

        self.c_max_depth = _check.nonneg_int(c_max_depth,'c_max_depth',ParameterFormatError)
        self.max_depth_after_build = None

        _check.pos_ints(c_num_children_vec,'c_num_children_vec',ParameterFormatError)
        if np.any(c_num_children_vec<2):
            raise(ParameterFormatError(
                'All the elements of c_num_children_vec must be greater than or equal to 2: '
                +f'c_num_children_vec={c_num_children_vec}.'
            ))
        self.c_num_children_vec = np.ones(self.c_dim_features,dtype=int)*2
        self.c_num_children_vec[:] = c_num_children_vec

        if c_feature_candidates is None:
            self.c_feature_candidates = list(range(self.c_dim_features))
        else:
            _check.unique_list(c_feature_candidates,'c_feature_candidates',ParameterFormatError)
            _check.floats_in_closedrange(c_feature_candidates,'c_feature_candidates',0,self.c_dim_features-1,ParameterFormatError)
            self.c_feature_candidates = c_feature_candidates

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

        self.c_ranges = np.zeros([self.c_dim_continuous,2])
        if c_ranges is not None:
            _check.float_vecs(c_ranges,'c_ranges',ParameterFormatError)
            self.c_ranges[:] = c_ranges
            if np.any(self.c_ranges[:,0] > self.c_ranges[:,1]):
                raise(ParameterFormatError(
                    'self.c_ranges[:,1] must be greater than or equal to self.c_ranges[:,0]'
                ))
        else:
            self.c_ranges[:,0] -= 1
            self.c_ranges[:,1] += 1


        # valuables which can be updated, from the input
        if type(h_split) == float:
            self.h_split = _check.float_in_closed01(h_split,'h_split',ParameterFormatError)
            self.h_split_list = np.full(self.c_max_depth+1, self.h_split, dtype=float) # TODO: Decide whether to set root node depth to 0 or 1.
        else:
            tmp = _check.one_dimensional_array_with_length(h_split,'h_split',self.c_max_depth+1, ParameterFormatError)
            self.h_split_list = _check.float_in_closed01_vec(tmp,'h_split',ParameterFormatError)

        if SubModel not in MODELS:
            raise(ParameterFormatError(
                "SubModel must be "
                +"normal"
                # +"bernoulli, poisson, normal, exponential, linearregression."
            ))
        self.SubModel = SubModel
        self.h_constants_SubModel = self.SubModel.GenModel(**h_constants_SubModel).get_constants()
        self.sub_h_params = self.SubModel.GenModel(**sub_h_params).get_h_params()

        if h_feature_weight_vec is None:
            self.h_feature_weight_vec = np.full(self.c_dim_features, 1/self.c_dim_features)
        else:
            self.h_feature_weight_vec = _check.proba_vec(h_feature_weight_vec,'h_feature_weight_vec',ParameterFormatError)

        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # automatically defined
        # tree structure
        self.root_node = _GenNode(
            self.rng,
            c_id = '',
            c_depth = 0,
            h_split = self.h_split_list[0],
            c_ancestor_nodes = [],
            is_leaf=True,
            c_feature_candidates=self.c_feature_candidates,
            c_num_assignment_vec=self.c_num_assignment_vec,
            c_data_region = [],
            children = [],
            feature = None,
            threshold = None,
            h_constants_SubModel = self.h_constants_SubModel,
            sub_h_params = self.sub_h_params,
            )
        self.nodes_list = [self.root_node]
        self.leaf_nodes_list = [self.root_node]
        self.inner_nodes_list = []
        self.num_leafnodes = 1
    def get_constants(self):
        """Get constants of GenModel.

        Returns
        -------
        constants : dict of {str: int, numpy.ndarray}
            * ``"c_dim_continuous"`` : the value of ``self.c_dim_continuous``
            * ``"c_dim_categorical"`` : the value of ``self.c_dim_categorical``
            * ``"c_num_children_vec"`` : the value of ``self.c_num_children_vec``
            * ``"c_max_depth"`` : the value of ``self.c_max_depth``
            * ``"c_num_assignment_vec"`` : the value of ``self.c_num_assignment_vec``
            * ``"c_ranges"`` : the value of ``self.c_ranges``
        """
        return {"c_dim_continuous":self.c_dim_continuous,
                "c_dim_categorical":self.c_dim_categorical,
                "c_num_children_vec":self.c_num_children_vec,
                "c_max_depth":self.c_max_depth,
                "c_num_assignment_vec":self.c_num_assignment_vec,
                "c_ranges":self.c_ranges,
                "sub_constants":self.sub_constants}
    
    ##########################
    # not yet finished
    ##########################
    def get_h_params(self):
        return
    def get_params(self):
        return
    def save_sample(self):
        return
    def set_h_params(self):
        return
    
    def visualize_model(self,filename=None,format=None,view=True,show_sub_constants=True,show_sub_h_params=True,show_sub_params=True):
        """Visualize the stochastic data generative model.

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
            self._visualize_model_recursion(
                tree_graph,
                self.root_node,
                None,
                None,
                None,
                show_sub_constants=show_sub_constants,
                show_sub_h_params=show_sub_h_params,
                show_sub_params=show_sub_params,
                )
            # Can we show the image on the console without saving the file?
            if view:
                tree_graph.view()
        except ImportError as e:
            print(e)
        except graphviz.CalledProcessError as e:
            print(e)
        return tree_graph

    def _visualize_model_recursion(self,tree_graph,node:_GenNode,parent_id,parent_split_rule,sibling_num,show_sub_constants=True,show_sub_h_params=True,show_sub_params=True):
        tmp_id = node.c_id
        label_string = f"node_id='{tmp_id}'\\l"
        # label_string += f"samples={node.num_samples}\\l"
        # add node information
        # if node.is_leaf:
            # label_string += 'k=None\\l'
        # else:
            # label_string += f'k={node._split_rule[0]}\\l'
            # if node._split_rule[0] < self.c_dim_continuous:
                # label_string += f'threshold=\\l{node._split_rule[1]:.2f}\\l'
        label_string += f'h_split={node.h_split:.3f}\\l'
        if node.sub_model is not None:
            # print model name
            model_names = [
                'bernoulli',
                'categorical',
                'normal',
                'linearregression',
            ]
            model_list = [eval(modelname) for modelname in model_names]
            for model, model_name in zip(model_list, model_names):
                if self.SubModel == model:
                    label_string += f'model={model_name}\\l'
                    break
            sub_constants = node.sub_model.get_constants()
            sub_h_params = node.sub_model.get_h_params()
            sub_params = node.sub_model.get_params()
            # print sub_constants
            if sub_constants != {} and show_sub_constants:
                label_string += 'sub_constants={'
                for key,value in sub_constants.items():
                    try: #int or float
                        label_string += f'\\l    {key}:{value:.2f}'
                    except:
                        try: # array
                            label_string += f'\\l    {key}:{np.array2string(value,precision=2,max_line_width=1)}'
                        except: # string
                            label_string += f'\\l    {key}:{value}'
                label_string += '}\\l'
            # print sub_h_params
            if sub_h_params != {} and show_sub_h_params:
                label_string += 'sub_h_params={'
                for key,value in sub_h_params.items():
                    try: #int or float
                        label_string += f'\\l    {key}:{value:.2f}'
                    except:
                        try: # array
                            label_string += f'\\l    {key}:{np.array2string(value,precision=2,max_line_width=1)}'
                        except: # string
                            label_string += f'\\l    {key}:{value}'
                label_string += '}\\l'
            # print sub_params
            if sub_params != {} and show_sub_params:
                label_string += 'sub_params={'
                for key,value in sub_params.items():
                    try: #int or float
                        label_string += f'\\l    {key}:{value:.2f}'
                    except:
                        try: # array
                            label_string += f'\\l    {key}:{np.array2string(value,precision=2,max_line_width=1)}'
                        except: # string
                            label_string += f'\\l    {key}:{value}'
                label_string += '}\\l'
            
        if node.is_leaf:
            fillcolor = f'{rgb2hex(_CMAP(1.))}'
            fontcolor = 'white'
        else:
            fillcolor = f'{rgb2hex(_CMAP(0.))}'
            fontcolor = 'black'
        tree_graph.node(name=f'{tmp_id}',label=label_string,fillcolor=fillcolor,fontcolor=fontcolor,style='filled')
        
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
                self._visualize_model_recursion(
                    tree_graph,
                    node.children[i],
                    tmp_id,
                    node._split_rule,
                    i,
                    show_sub_constants=show_sub_constants,
                    show_sub_h_params=show_sub_h_params,
                    show_sub_params=show_sub_params,)
        return
    
    def get_split_rules(self):
        split_rule_dict = {}
        for node in self.inner_nodes_list:
            split_rule_dict[node.c_id] = node._split_rule
        # sort by node.c_id
        split_rule_dict = dict(sorted(split_rule_dict.items()))
        return split_rule_dict


    def print_split_rules(self, depth):
        for node in self.nodes_list:
            if node.c_depth <= depth:
                print(node.c_id, node._split_rule)


    ##########################################
    # building tree
    ##########################################
    def build(
            self,
            seed: Optional[Union[int,np.random.Generator]] = None,
            building_scheme: str = 'depth_first',
            max_depth: Optional[int] = None,
            h_split: Optional[float] = None,
            ):
        """
       Build tree on GenModel.

        Returns
        -------
        constants : dict of {str: int, numpy.ndarray}
            * ``"c_dim_continuous"`` : the value of ``self.c_dim_continuous``
            * ``"c_dim_categorical"`` : the value of ``self.c_dim_categorical``
            * ``"c_num_children_vec"`` : the value of ``self.c_num_children_vec``
            * ``"c_max_depth"`` : the value of ``self.c_max_depth``
            * ``"c_num_assignment_vec"`` : the value of ``self.c_num_assignment_vec``
            * ``"c_ranges"`` : the value of ``self.c_ranges``
        """
    # Handle parameter input values. TODO: Consider preparing a function like imput_param.
        if seed is None: # use self.rng 
            pass
        else:
            self.rng = np.random.default_rng(seed)
        if max_depth is None: # use self.max_depth
            pass
        else:
            self.c_max_depth = max_depth
        if h_split is not None:
            if type(h_split) == float:
                self.h_split = _check.float_in_closed01(h_split,'h_split',ParameterFormatError)
                self.h_split_list = np.full(self.c_max_depth, self.h_split, dtype=float) # TODO ノード深さを根ノードで0とするか1とするか決定
            else:
                tmp = _check.one_dimensional_array_with_length(h_split,'h_split',self.c_max_depth+1, ParameterFormatError)
                self.h_split_list = _check.float_in_closed01_vec(tmp,'h_split',ParameterFormatError)
        for node in self.leaf_nodes_list:
            node.h_split = self.h_split_list[node.c_depth]
        # build
        if building_scheme == 'depth_first':
            return self._build_depth_first()
        elif building_scheme == 'breadth_first':
            return self._build_breadth_first()
        else:
            raise(ParameterFormatError(f'building_scheme:{building_scheme} is not yet supported.'))

    def _build_depth_first(self):
        leaf_nodes_to_split = deque(self.leaf_nodes_list)
        leaf_nodes_end = []
        while bool(leaf_nodes_to_split): # repeat while leaf_nodes are not empty
            node = leaf_nodes_to_split.pop() # take top-right node (latest, stack)

            # randomly decide to split or not
            if node.c_depth == self.c_max_depth:
                node.is_leaf = True
            elif node.c_depth < self.c_max_depth:
                node.is_leaf = self.rng.choice(a=(True,False), p=(1-node.h_split, node.h_split))
            else:
                exit(-1)
            if node.is_leaf: # Stop splitting and go to next node in leaf_nodes_to_split
                leaf_nodes_end.append(node)
                continue
            self._split_node(node)
            self.inner_nodes_list.append(node)
            for child in reversed(node.children): # Add to leaf_nodes in reverse order so that we descend from the left node (smaller index) first (same as previous implementation)
                # if node.children is empty, i.e. node is a leaf_node, this loop won't be processed  
                leaf_nodes_to_split.append(child) # Add children to top right in the stack
                self.nodes_list.append(child)
        self.leaf_nodes_list = list(leaf_nodes_end)
        return self
    
    def _build_breadth_first(self):
        leaf_nodes_to_split = deque(self.leaf_nodes_list)
        leaf_nodes_end = []
        while bool(leaf_nodes_to_split): # repeat while leaf_nodes are not empty
            node = leaf_nodes_to_split.popleft() # take top-left node (oldest, que)

            # randomly decide to split or not
            if node.c_depth == self.c_max_depth:
                node.is_leaf = True
            elif node.c_depth < self.c_max_depth:
                node.is_leaf = self.rng.choice(a=(True,False), p=(1-node.h_split, node.h_split))
            else:
                exit(-1)
            if node.is_leaf: # stop splitting and go to next node in leaf_nodes_to_split
                leaf_nodes_end.append(node)
                continue

            self._split_node(node)
            self.inner_nodes_list.append(node)
            for child in node.children: # Add to leaf_nodes in reverse order so that we descend from the left node (smaller index) first (same as previous implementation)
                # if node.children is empty, i.e. node is a leaf_node, this loop won't be processed  
                leaf_nodes_to_split.append(child) # Add children to top right in the queue
                self.nodes_list.append(child)
        self.leaf_nodes_list = list(leaf_nodes_end)
        return self
    
    ##########################################
    # split gen node 
    # TODO: Decide whether to implement split function as a method of MetaTree or Node.
    ##########################################
    def _gen_split_rule_node(
            self,
            node: _GenNode
    ): # TODO: If multiple splits occur on the same continuous feature, meaningless splits (leaf nodes that are never visited) may be generated.

        tmp = [self.h_feature_weight_vec[i] for i in node.c_feature_candidates]
        prob_feature = tmp / np.sum(tmp)
        node._split_rule[0] = self.rng.choice(a=node.c_feature_candidates, p=prob_feature)
        if node._split_rule[0] < self.c_dim_continuous:
            node._split_rule[1] = self.rng.uniform(self.c_ranges[node._split_rule[0]][0], self.c_ranges[node._split_rule[0]][1])
        else:
            node._split_rule[1] = None
        return node
    def _split_node(
            self,
            node: _GenNode,
    ):  
        node = self._gen_split_rule_node(node)
        # eliminate feature from candidate list, if the feature is categorical
        c_num_assignment_vec_child = deepcopy(node.c_num_assignment_vec)
        c_num_assignment_vec_child[node._split_rule[0]] = node.c_num_assignment_vec[node._split_rule[0]] - 1
        if c_num_assignment_vec_child[node._split_rule[0]] == 0:
            feature_candidate_child = [item for item in node.c_feature_candidates if item != node._split_rule[0]]
        else:
            feature_candidate_child = deepcopy(node.c_feature_candidates)
        h_split_child = 0. if node.c_depth+1 == self.c_max_depth else self.h_split_list[node.c_depth+1]

        num_children = self.c_num_children_vec[node._split_rule[0]]
        node.children = [None for _ in range(num_children)]
        c_data_region_children = [(node._split_rule, i) for i in range(num_children)]
        for i in range(num_children):
            node.children[i] = _GenNode(
                self.rng,
                c_id = node.c_id + str(i),
                c_depth = node.c_depth + 1,
                h_split = h_split_child,
                c_ancestor_nodes = node.c_ancestor_nodes + [node],
                is_leaf=True,
                c_feature_candidates=feature_candidate_child,
                c_num_assignment_vec=c_num_assignment_vec_child,
                c_data_region = node.c_data_region + [c_data_region_children[i]],
                children = [],
                feature = None,
                threshold = None,
                h_constants_SubModel = self.h_constants_SubModel,
                sub_h_params = self.sub_h_params, # TODO: shrinkage prior option
            )
        return
    ##########################################
    # set submodel params on leafnodes
    ##########################################
    def _set_submodel_leaf(self):
        self.leaf_nodes_list = list(map(lambda x: x._prepare_sub_model(self.SubModel), self.leaf_nodes_list))
        return
    def _set_submodel_all(self):
        self.nodes_list = list(map(lambda x: x._prepare_sub_model(self.SubModel), self.nodes_list))
        return
    
    ##########################################
    # prepare genmodel: build tree and gen params on leafnodes
    ##########################################
    def init_tree(
            self,
            seed: Optional[Union[int,np.random.Generator]] = None,
            building_scheme: str = 'depth_first',
            max_depth: Optional[int] = None,
            h_split: Optional[float] = None,
            ):
        """
        Build trees and define SubModel on leaf nodes.

        Parameters
        ----------
        seed : int, Generator, optional
        building_scheme : 'depth_first' or 'breadth_first', optional # TODO: add 'fixed_tree'
            Order of visiting nodes to build trees. By default 'depth_first.'
        max_depth : int, optional
        h_split : float, optional # TODO: isn't it ArrayLike for each depth ?
        ↓ not yet
        feature_fix : bool, optional
            If ``True``, feature assignment indices will be fixed, by default ``False``.
        threshold_fix : bool, optional
            If ``True``, thresholds for continuous features will be fixed, by default ``False``. 
            If ``feature_fix`` is ``False``, ``threshold_fix`` must be ``False``. 
        tree_fix : bool, optional
            If ``True``, tree shape will be fixed, by default ``False``. 
            If ``feature_fix`` is ``False``, ``tree_fix`` must be ``False``.
        threshold_type : {'even', 'random'}, optional
            A type of threshold generating procedure, by default ``'even'``
            If ``'even'``, self.c_ranges will be recursively divided by equal intervals. 
            if ``'random'``, self.c_ranges will be recursively divided by at random intervals.
        """
        if seed is None:
            seed = self.seed
        self.rng = np.random.default_rng(seed)
        
        self.build(
            building_scheme=building_scheme,
            max_depth=max_depth,
            h_split=h_split
            )
        self.max_depth_after_build = np.max([node.c_depth for node in self.leaf_nodes_list])
        self.num_leafnodes = len(self.leaf_nodes_list)
        self._set_submodel_leaf()
        self.dict_node_id = defaultdict()
        for node in self.nodes_list:
            self.dict_node_id[node.c_id] = node
        return

    def gen_params(
            self,
            seed: Optional[Union[int,np.random.Generator]] = None,
            where: str = 'leaf',
            ):
        """Generate the parameter on the leaf nodes,  from the prior distribution.

        Parameters
        ----------
        seed : int, Generator, optional
        where: str
            node to set parameter, by default 'leaf'.
            'leaf' -> set params for all leaf nodes.
            'all' ->  set params for all nodes.
            f'{node_id}' -> set params for the node with node_id.
        """
        if seed is None:
            pass
        else:
            self.rng = np.random.default_rng(seed)
            for node in self.nodes_list:
                node.rng = self.rng
                node.submodel.rng = self.rng
        if where == 'leaf':
            for node in self.leaf_nodes_list:
                node.sub_model.gen_params()
        elif where == 'all':
            for node in self.nodes_list:
                node.sub_model.gen_params()
            self.nodes_list = list(map(lambda x: x.sub_model.gen_params(), self.nodes_list))
        else:
            try:
                node = self.dict_node_id[where]
                node.sub_model.gen_params()
            except KeyError as e:
                raise(ParameterFormatError(f"\'where\' must be a \'leaf\', \'all\', or a node id. {e} was passed."))
        return
        

    ##########################################
    # prepare genmodel: set params
    ##########################################
    def set_params(
        self,
        sub_params: Optional[dict] = None,
        split_rule: Optional[List] = None,
        where: str = 'leaf',
        ): # TODO: 分岐確率などのサブモデル以外のパラメタへの対応．
        """Set the parameter of the sthocastic data generative model.

        Parameters
        ----------
        sub_params: dict, optional
            A dictionary of the parameters of the node.SubModel.
        where: str
            node to set parameter, by default 'leaf'.
            'leaf' -> set params for all leaf nodes.
            'all' ->  set params for all nodes.
            f'{node_id}' -> set params for the node with node_id.
        """
        if where == 'leaf':
            self.leaf_nodes_list = list(map(lambda x: x.sub_model.set_params(**sub_params), self.leaf_nodes_list))
        elif where == 'all':
            self.nodes_list = list(map(lambda x: x.sub_model.set_params(**sub_params), self.nodes_list))
        else:
            try:
                node = self.dict_node_id[where]
                if sub_params is not None:
                    node.sub_model.set_params(**sub_params)
                elif split_rule is not None:
                    node._split_rule = split_rule
                else:
                    raise(ParameterFormatError(f"\'sub_params\' or \'split_rule\' must be passed."))
            except KeyError as e:
                raise(ParameterFormatError(f"\'where\' must be a \'leaf\', \'all\', or a node id. {e} was passed."))
        return
    ##########################################
    # prepare genmodel: set hyperparams
    ##########################################
    def set_h_params(
        self,
        sub_h_params: Optional[dict] = None,
        where: str = 'leaf',
        ): # TODO: 分岐確率などのサブモデル以外のパラメタへの対応．
        """Set the hyperparameter of the sthocastic data generative model.

        Parameters
        ----------
        sub_params: dict, optional
            A dictionary of the parameters of the node.SubModel.
        where: str
            node to set parameter, by default 'leaf'.
            'leaf' -> set params for all leaf nodes.
            'all' ->  set params for all nodes.
            f'{node_id}' -> set params for the node with node_id.
        """
        if where == 'leaf':
            for node in self.leaf_nodes_list:
                node.sub_model.set_h_params(**sub_h_params)
        elif where == 'all':
            for node in self.nodes:
                node.sub_model.set_h_params(**sub_h_params)
        else:
            try:
                node = self.dict_node_id[where]
                node.sub_model.set_h_params(**sub_h_params)
            except KeyError as e:
                raise(ParameterFormatError(f"\'where\' must be a \'leaf\', \'all\', or a node id. {e} was passed."))
        return

    ##########################################
    # gen sample
    ##########################################
    def _gen_x_uniform(
            self,
            sample_size: Optional[int] = None,
            x_continuous: Optional[np.ndarray] = None,
            x_categorical: Optional[np.ndarray] = None
            ) -> Tuple[np.ndarray, np.ndarray]:
    # If x_continuous or x_categorical is None, generate new data.
        if x_continuous is not None:
            if not isinstance(x_continuous, np.ndarray):
                x_continuous = np.array(x_continuous)
            _check.float_vecs(x_continuous,'x_continuous',DataFormatError)
            _check.shape_consistency(
                x_continuous.shape[-1],'x_continuous.shape[-1]',
                self.c_dim_continuous,'self.c_dim_continuous',
                ParameterFormatError
                )
            x_continuous = x_continuous.reshape(-1,self.c_dim_continuous)
            sample_size = x_continuous.shape[0]
            if x_categorical is not None:
                if not isinstance(x_categorical, np.ndarray):
                    x_categorical = np.array(x_categorical)
                _check.nonneg_int_vecs(x_categorical,'x_categorical',DataFormatError)
                _check.shape_consistency(
                    x_categorical.shape[-1],'x_categorical.shape[-1]',
                    self.c_dim_categorical,'self.c_dim_categorical',
                    ParameterFormatError
                    )
                x_categorical = x_categorical.reshape(-1,self.c_dim_categorical)
                _check.shape_consistency(
                    x_categorical.shape[0],'x_categorical.shape[0]',
                    x_continuous.shape[0],'x_continuous.shape[0]',
                    ParameterFormatError
                    )
                for i in range(self.c_dim_categorical):
                    if x_categorical[:,i].max() >= self.c_num_children_vec[self.c_dim_continuous+i]:
                        raise(DataFormatError(
                            f"x_categorical[:,{i}].max() must smaller than "
                            +f"self.c_num_children_vec[{self.c_dim_continuous+i}]: "
                            +f"{self.c_num_children_vec[self.c_dim_continuous+i]}"))
            else:
                x_categorical = np.empty([sample_size,self.c_dim_categorical],dtype=int)
                for i in range(self.c_dim_categorical):
                    x_categorical[:,i] = self.rng.choice(
                        self.c_num_children_vec[self.c_dim_continuous+i],
                        sample_size)
        elif x_categorical is not None:
            if not isinstance(x_categorical, np.ndarray):
                x_categorical = np.array(x_categorical)
            _check.nonneg_int_vecs(x_categorical,'x_categorical',DataFormatError)
            _check.shape_consistency(
                x_categorical.shape[-1],'x_categorical.shape[-1]',
                self.c_dim_categorical,'self.c_dim_categorical',
                ParameterFormatError
                )
            x_categorical = x_categorical.reshape(-1,self.c_dim_categorical)
            for i in range(self.c_dim_categorical):
                if x_categorical[:,i].max() >= self.c_num_children_vec[self.c_dim_continuous+i]:
                    raise(DataFormatError(
                        f"x_categorical[:,{i}].max() must smaller than "
                        +f"self.c_num_children_vec[{self.c_dim_continuous+i}]: "
                        +f"{self.c_num_children_vec[self.c_dim_continuous+i]}"))
            sample_size = x_categorical.shape[0]
            x_continuous = np.empty([sample_size,self.c_dim_continuous],dtype=float)
            for i in range(self.c_dim_continuous):
                x_continuous[:,i] = ((self.c_ranges[i,1]-self.c_ranges[i,0])
                                        * self.rng.random(sample_size)
                                        + self.c_ranges[i,0])
        elif sample_size is not None:
            sample_size = _check.pos_int(sample_size,'sample_size',DataFormatError)
            x_continuous = np.empty([sample_size,self.c_dim_continuous],dtype=float)
            for i in range(self.c_dim_continuous):
                x_continuous[:,i] = ((self.c_ranges[i,1]-self.c_ranges[i,0])
                                        * self.rng.random(sample_size)
                                        + self.c_ranges[i,0])
            x_categorical = np.empty([sample_size,self.c_dim_categorical],dtype=int)
            for i in range(self.c_dim_categorical):
                x_categorical[:,i] = self.rng.choice(
                    self.c_num_children_vec[self.c_dim_continuous+i],
                    sample_size)
        else:
            raise(DataFormatError("Either of sample_size, x_continuous, and x_categorical must be given as a input."))
        return x_continuous, x_categorical

    # def _gen_y_given_x_instance(
    #         self,
    #         x_continuous_instance: ArrayLike,
    #         x_categorical_instance: ArrayLike,
    # ): # TODO もしかしたら遅いかも，複数データまとめて生成できる？
    #     node = self.root_node
    #     while node.is_leaf == False:
    #         node = node._child_node_given_x(x_continuous_instance, x_categorical_instance, self.c_dim_continuous, self.c_dim_categorical)
    #     if self.SubModel is linearregression:
    #         _, y = node.sub_model.gen_sample(sample_size=1,x=x_continuous_instance)
    #         return y, node.c_id
    #     elif self.SubModel is categorical:
    #         return node.sub_model.gen_sample(sample_size=1,onehot=False), node.c_id
    #     else:
    #         return node.sub_model.gen_sample(sample_size=1), node.c_id

    def _gen_y_given_x(
            self,
            x_continuous: np.ndarray,
            x_categorical: np.ndarray,
    ):
        y = np.empty(x_continuous.shape[0])
        y_node_id = np.zeros(x_continuous.shape[0], dtype=object)
        for node in self.leaf_nodes_list:
            indices = self._get_data_indices_node(node, x_continuous, x_categorical)
            sample_size_node = np.count_nonzero(indices)
            if sample_size_node > 0:
                y_node_id[indices] = np.full(sample_size_node, node.c_id, dtype=object)
                if self.SubModel is linearregression:
                    _, y[indices] = node.sub_model.gen_sample(sample_size=sample_size_node,x=x_continuous)
                elif self.SubModel is categorical:
                    y[indices] = node.sub_model.gen_sample(sample_size=sample_size_node,onehot=False)
                else:
                    y[indices] =  node.sub_model.gen_sample(sample_size=sample_size_node)
        return y, y_node_id
                
    def _get_data_indices_node(
            self,
            node: _GenNode,
            x_continuous: np.ndarray,
            x_categorical: np.ndarray,
    ):
        indices = np.full(x_continuous.shape[0], True) # Faster than np.ones
        for reg in node.c_data_region:
            if reg[0][1] is None: # threshold is None -> categorical
                indices = indices * (x_categorical[:,reg[0][0]-self.c_dim_continuous] == reg[1])
            else: # continuous
                indices = indices * ((x_continuous[:,reg[0][0]] >= reg[0][1]) == reg[1])
        return indices

    def gen_sample(
            self,
            sample_size: Optional[int] = None,
            x_continuous: Optional[np.ndarray] = None,
            x_categorical: Optional[np.ndarray] = None,
            ):
    # If x_continuous or x_categorical is None, generate with _uniform.
        if (x_continuous is None) or (x_categorical is None):
            x_continuous_gen, x_categorical_gen = self._gen_x_uniform(sample_size=sample_size)
            x_continuous = x_continuous_gen
            x_categorical = x_categorical_gen
        y, y_node_id = self._gen_y_given_x(x_continuous, x_categorical)
        return x_continuous, x_categorical, y, y_node_id
        """Generate a sample from the stochastic data generative model.

        Parameters
        ----------
        sample_size : int, optional
            A positive integer, by default ``None``
        x_continuous : numpy ndarray, optional
            2 dimensional float array whose size is ``(sample_size,c_dim_continuous)``, 
            by default None.
        x_categorical : numpy ndarray, optional
            2 dimensional int array whose size is ``(sample_size,c_dim_categorical)``, 
            by default None. Each element x_categorical[i,j] must satisfy 
            0 <= x_categorical[i,j] < self.c_num_children_vec[self.c_dim_continuous+i].

        Returns
        -------
        x_continuous : numpy ndarray
            2 dimensional float array whose size is ``(sample_size,c_dim_continuous)``.
        x_categorical : numpy ndarray, optional
            2 dimensional int array whose size is ``(sample_size,c_dim_categorical)``.
            Each element x_categorical[i,j] must satisfies 
            0 <= x_categorical[i,j] < self.c_num_children_vec[self.c_dim_continuous+i].
        y : numpy ndarray
            1 dimensional array whose size is ``sample_size``.
        """ 

        if (x_continuous is None) or (x_categorical is None):
            x_continuous, x_categorical = self._gen_x_uniform(sample_size=sample_size, x_continuous=x_continuous, x_categorical=x_categorical)   
        else:
            pass # TODO if x_continuous is None and x_categorical is not None ?

        y, y_node_id = self._gen_y_given_x(x_continuous, x_categorical)
                   
        return x_continuous, x_categorical, y, y_node_id