class Config:
    def __init__(self):
        self.ope2id = {
            'MinMaxScaler': 1,
            'MaxAbsScaler': 2,
            'RobustScaler': 3,
            'StandardScaler': 4,
            'QuantileTransformer': 5,
            'PowerTransformer': 6,
            'Normalizer': 7,
            'KBinsDiscretizer': 8,
            'PolynomialFeatures': 9,
            'InteractionFeatures': 10,
            'PCA_LAPACK': 11,
            'PCA_ARPACK': 12,
            'PCA_Randomized': 13,
            'IncrementalPCA': 14,
            'KernelPCA': 15,
            'TruncatedSVD': 16,
            'RandomTreesEmbedding': 17,
            'PCA_AUTO': 18,
            'VarianceThreshold': 19
        }
        self.dataset_feature_colnum = 100
        self.column_feature_dim = 19

        self.seq_len = 30
        self.ope_num = len(self.ope2id) + 2
        self.info_triple_path = 'HybridPipeGen/core/merged_info_triple.npy'
        self.dataset_label_path = 'HybridPipeGen/core/merged_dataset_label.json'

        self.param_candidate_offline_hybrid_val_json_save_root_path = '../hybridpipe/data/param_gamma/offline_26000/candidate_hybrid_val_json/'
        

        self.param_offline_hybrid_val_result_save_root_path = '../hybridpipe/data/param_gamma/offline_26000/result_data/xgb_hybrid_val/'
        self.origin_dataset = 'data/dataset/'
        self.train_features = 'HybridPipeGen/core/al/train_features/'
        self.dataset_path = 'data/dataset/'
        

        self.dtype_dic = {
            'interval[float64]':4,
            'uint8':1,
            'uint16': 1,
            'int64': 1,
            'int': 1,
            'int32': 1,
            'int16': 1,
            'np.int32': 1,
            'np.int64': 1,
            'np.int': 1,
            'np.int16': 1,
            'float64': 2,
            'float': 2,
            'float32': 2,
            'float16': 2,
            'np.float32': 2,
            'np.float64': 2,
            'np.float': 2,
            'np.float16': 2,
            'str':3,
            'Category':4,
            'object':4,
            'bool':4
        }

        # elf.test_features = 'origin_features/test_user/'
        # self.param_candidate_hybrid_val_json_save_root_path = '../hybridpipe/data/param_gamma/al_user/candidate_hybrid_val_json/'
        # self.param_candidate_hybrid_val_py_save_root_path = '../hybridpipe/data/param_gamma/al_user/candidate_hybrid_val_py/'
        # self.param_hybrid_test_result_save_root_path = '../hybridpipe/data/param_gamma/al_user/result_data/hybrid_test/'
        # self.param_candidate_hybrid_test_py_save_root_path = '../hybridpipe/data/param_gamma/al_user/candidate_hybrid_test_py/'
        # self.param_hybrid_val_result_save_root_path = '../hybridpipe/data/param_gamma/al_user/result_data/hybrid_val/'

        self.test_features = 'HybridPipeGen/core/al/test_features/'
        self.param_candidate_hybrid_val_json_save_root_path = 'HybridPipeGen/core/tmpdata/rl_cross_validation_code/'
        self.param_candidate_hybrid_val_py_save_root_path = 'HybridPipeGen/core/tmpdata/rl_cross_validation_code_py/'
        self.param_hybrid_test_result_save_root_path = "HybridPipeGen/core/tmpdata/merge_max_result_rl/"
        self.param_candidate_hybrid_test_py_save_root_path = 'HybridPipeGen/core/tmpdata/rl_test_merge_code_py/'
        self.param_hybrid_val_result_save_root_path = 'HybridPipeGen/core/tmpdata/rl_cross_val_res/'