import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gradient_boosting_meta_tree.metatree as metatree
import gradient_boosting_meta_tree.normal as normal

import numpy as np

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

SEED = 1

def test_gen_sample():
    # MetaTreeGenModelのインスタンス生成
    c_dim_continuous = 3
    c_dim_categorical = 2
    sample_size = 1000
    c_max_depth = 1
    h0_split_genmodel = 1.
    sub_h_params_genmodel = {
        'tau': 1e5,
        'h_m': 0.0,
        'h_tau': 1e0,
        'known_precision': True,
    }
    h_split_list = np.array([h0_split_genmodel]*c_max_depth + [0.])
    genmodel = metatree.MetaTreeGenModel(
        SubModel=normal,
        c_dim_continuous=c_dim_continuous,
        c_dim_categorical=c_dim_categorical,
        c_max_depth=c_max_depth,
    )
    genmodel.init_tree(
        seed=SEED,
        max_depth=c_max_depth,
        h_split=h_split_list,
    )
    genmodel.set_h_params(
        sub_h_params=sub_h_params_genmodel,
        where='leaf',
    )
    genmodel.gen_params()
    x_continuous, x_categorical, y, y_node_id = genmodel.gen_sample(
        sample_size=sample_size
    )
    print(genmodel.num_leafnodes)
    return x_continuous, x_categorical, y, y_node_id

def test_learn(x_continuous, x_categorical, y):
    input_data = {
        'x_continuous_vecs': x_continuous,
        'x_categorical_vecs': x_categorical,
        'y_vec': y,
    }
    data_info = {
            'c_dim_continuous': x_continuous.shape[1],
            'c_dim_categorical': x_categorical.shape[1],
    }

    # create dict for model init input
    sub_h_params_learnmodel = {
        'h0_tau_x': 1e10,
        'h0_m': 0,
        'h0_tau': 1e-10,
        'known_precision': True,
    }
    MT_model_dict = {
        'model': metatree.MetaTreeLearnModel,
        'model_type': 'metatree',
        'init_params': {
            'SubModel': normal,
            'sub_h0_params': sub_h_params_learnmodel, # same to true-generative model
            'c_max_depth': 5,
            'h0_split': 0.8,
            'seed': SEED,
        },
        'build_params': {
            'split_strategy': 'best',
            'criterion': 'squared_error_leaf',
            'building_scheme': 'depth_first',
        }
    }
    
    init_inputs = {
        **MT_model_dict['init_params'],
        **data_info,
    }
    build_inputs = {
        **MT_model_dict['build_params'],
        **input_data,
    }
    learnmodel = MT_model_dict['model'](**init_inputs)
    learnmodel.build(**build_inputs)
    learnmodel.calc_pred_dist()

    hat_y = learnmodel.make_prediction(
        x_continuous,
        x_categorical,
        loss='squared'
    )

    print("Predicted:", hat_y[:5])
    print("Actual:", y[:5])

    assert np.isclose(hat_y, y, atol=1e-1).all()
if __name__ == "__main__":
    x_continuous, x_categorical, y, y_node_id = test_gen_sample()
    test_learn(x_continuous, x_categorical, y)

