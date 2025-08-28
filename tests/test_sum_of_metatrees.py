import warnings
warnings.simplefilter('ignore', category=FutureWarning)
warnings.simplefilter('ignore', category=RuntimeWarning)

import unittest
from gradient_boosting_meta_tree import normal, bernoulli, categorical, metatree
import numpy as np


SEED = 42

class TestSumOfMetatrees(unittest.TestCase):
    def setUp(self):
        # 学習データ生成（各テストで使い回し）
        self.c_dim_continuous = 3
        self.c_dim_categorical = 2
        self.sample_size = 1000
        self.c_max_depth = 1
        self.h0_split_genmodel = 1.
        self.sub_h_params_genmodel = {
            'tau': 1e5,
            'h_m': 0.0,
            'h_tau': 1e0,
            'known_precision': True,
        }
        self.h_split_list = np.array([self.h0_split_genmodel]*self. c_max_depth + [0.])
        self.genmodel = metatree.MetaTreeGenModel(
            SubModel=normal,
            c_dim_continuous=self.c_dim_continuous,
            c_dim_categorical=self.c_dim_categorical,
            c_max_depth=self.c_max_depth,
        )
        self.genmodel.init_tree(
            seed=SEED,
            max_depth=self.c_max_depth,
            h_split=self.h_split_list,
        )
        self.genmodel.set_h_params(
            sub_h_params=self.sub_h_params_genmodel,
            where='leaf',
        )
        self.genmodel.gen_params()
        self.x_continuous, self.x_categorical, self.y, self.y_node_id \
            = self.genmodel.gen_sample(sample_size=self.sample_size)

    def test_import(self):
        # インポートが正常にできるか
        self.assertTrue(hasattr(metatree, '__file__'))

    def test_basic_functionality(self):
        # 主要なクラスや関数が存在するか
        self.assertTrue(hasattr(metatree, 'MetaTreeGenModel'))
        self.assertTrue(hasattr(metatree, 'MetaTreeLearnModel'))

    def test_gen_samples(self):
        # サンプル生成関数のテスト
        self.assertIsInstance(self.x_continuous, np.ndarray)
        self.assertIsInstance(self.x_categorical, np.ndarray)
        self.assertIsInstance(self.y, np.ndarray)
        self.assertIsInstance(self.y_node_id, np.ndarray)
        self.assertGreater(len(self.x_continuous), 0)
        self.assertGreater(len(self.x_categorical), 0)
        self.assertGreater(len(self.y), 0)
        self.assertGreater(len(self.y_node_id), 0)

    def test_learn(self):
        input_data = {
            'x_continuous_vecs': self.x_continuous,
            'x_categorical_vecs': self.x_categorical,
            'y_vec': self.y,
        }
        data_info = {
                'c_dim_continuous': self.x_continuous.shape[1],
                'c_dim_categorical': self.x_categorical.shape[1],
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
            self.x_continuous,
            self.x_categorical,
            loss='squared'
        )

        self.assertIsInstance(hat_y, np.ndarray)
        self.assertEqual(hat_y.shape[0], self.x_continuous.shape[0])

        assert np.isclose(hat_y[0], self.y[0], atol=1e-1)

if __name__ == '__main__':
    unittest.main()
