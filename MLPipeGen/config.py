
from env.primitives.encoder import *
from env.primitives.fengine import *
from env.primitives.fpreprocessing import *
from env.primitives.fselection import *
from env.primitives.imputernum import *
from env.primitives.predictor import *
from env.metric import *
import json
import math
import os
class Config:
    version = 0
    

    env: str = None
    gamma: float = 0
    learning_rate: float = 1*1e-5
    frames: int = 100000
    test_frames: int = 6
    episodes: int = 1
    max_buff: int = 2000
    batch_size: int = 100
    logic_batch_size: int = 20
    column_num: int = 20
    column_num: int = 100
    column_feature_dim: int = 19
    # dataset_path : str = '/home/datamanager/dataset/fixline/'
    dataset_path : str = 'datasets/'

    epsilon: float = 1
    eps_decay: float = 500
    epsilon_min: float = 0.4

    logic_pipeline_1 = ['ImputerNum', 'ImputerCat', 'Encoder', 'FeaturePreprocessing', 'FeatureEngine', 'FeatureSelection']
    logic_pipeline_2 = ['ImputerNum', 'ImputerCat', 'Encoder', 'FeaturePreprocessing', 'FeatureSelection', 'FeatureEngine']
    logic_pipeline_3 = ['ImputerNum', 'ImputerCat', 'Encoder', 'FeatureEngine', 'FeatureSelection', 'FeaturePreprocessing']
    logic_pipeline_4 = ['ImputerNum', 'ImputerCat', 'Encoder', 'FeatureEngine', 'FeaturePreprocessing', 'FeatureSelection']
    logic_pipeline_5 = ['ImputerNum', 'ImputerCat', 'Encoder', 'FeatureSelection', 'FeatureEngine', 'FeaturePreprocessing']
    logic_pipeline_6 = ['ImputerNum', 'ImputerCat', 'Encoder', 'FeatureSelection', 'FeaturePreprocessing', 'FeatureEngine']

    # state dim
    data_dim: int = column_num*column_feature_dim
    prim_state_dim: int = data_dim + len(logic_pipeline_1) + 1 + 1
    lpip_state_dim: int = data_dim + 1

    # RNN param
    seq_embedding_dim: int = 30
    seq_hidden_size: int = 18
    seq_num_layers: int = 2
    predictor_embedding_dim: int = 10
    lpipeline_embedding_dim: int = 5


    output = 'out'

    use_cuda: bool = False

    checkpoint: bool = True
    checkpoint_interval: int = None

    record: bool = False
    # record_ep_interval: int = None

    log_interval: int = 200
    print_interval: int = 200
    test_interval: int = 500
    train_interval: int= 100
    # test_interval: int =    100

    update_tar_interval: int = 100

    win_reward: float = 198
    win_break: bool = True
 
    taskdim: int = 56

    classification_metric_id: int = 1
    regression_metric_id: int = 4

    k_fold = 3
    with open('jsons/classification_task_dic.json', 'r') as f:
        classification_task_dic = json.load(f)
    
    train_classification_task_dic = {}
    fold_length = math.ceil(len(classification_task_dic)/k_fold)
    with open('jsons/test_index.json','r') as f:
        test_index = json.load(f)
    # test_index = [str(i) for i in range(version*k_fold, min(version*k_fold + fold_length, len(classification_task_dic)))]
    train_index = list(set(classification_task_dic.keys())-set(test_index))
    # for task_index in classification_task_dic:
    #     if int(task_index) in train_index:
    #         train_classification_task_dic[task_index] = classification_task_dic[task_index].copy()
    # classification_task_dic = {
    #     1: 'Accident_Casualties', 2: 'Frogs_MFCCs_family', 3: 'Frogs_MFCCs_gen', 4: 'Frogs_MFCCs_spec', 
    #     5: 'HTRU_2', 6: 'Indian Liver Patient Dataset (ILPD)', 7: 'Iris', 8: 'LendingClub Issued Loans', 
    #     9: 'Skin_NonSkin', 10: 'The_broken_machine', 11: 'Wine_classification', 12: 'adult', 
    #     13: 'analcatdata_broadwaymult', 14: 'analcatdata_germangss', 15: 'ar4', 16: 'bank-full', 
    #     17: 'baseball', 18: 'biodeg', 19: 'blood-transfusion', 20: 'bodyfat', 
    #     21: 'braziltourism', 22: 'bureau', 23: 'car', 24: 'chatfield_4', 
    #     25: 'cmc', 26: 'crx', 27: 'data_banknote_authentication', 28: 'dermatology', 
    #     29: 'diggle_table_a2', 30: 'disclosure_z', 31: 'glass', 32: 'haberman', 
    #     33: 'home_credit', 34: 'image segmentation', 35: 'kc3', 36: 'kidney', 
    #     37: 'magic04', 38: 'mammographic_masses', 39: 'movement_libras', 40: 'no2', 
    #     41: 'plasma_retinol', 42: 'pm10', 43: 'schizo', 44: 'socmob', 
    #     45: 'solar-flare', 46: 'test_bng_cmc', 47: 'test_breast', 48: 'test_credit', 
    #     49: 'test_eucalyptus', 50: 'test_ilpd', 51: 'test_irish', 52: 'test_phoneme', 
    #     53: 'triazines', 54: 'veteran', 55: 'weatherAUS', 56: 'wilt', 57: 'titanic'
    # }
 
    # ###################################################### v0
    # if version == 0:
    #     train_classification_task_dic = {
    #         15: 'ar4', 16: 'bank-full', 
    #         17: 'baseball', 18: 'biodeg', 19: 'blood-transfusion', 20: 'bodyfat', 
    #         21: 'braziltourism', 22: 'bureau', 23: 'car', 24: 'chatfield_4', 
    #         25: 'cmc', 26: 'crx', 27: 'data_banknote_authentication', 28: 'dermatology', 
    #         29: 'diggle_table_a2', 30: 'disclosure_z', 31: 'glass', 32: 'solar-flare', 
    #         33: 'home_credit', 34: 'image segmentation', 35: 'kc3', 36: 'kidney', 
    #         37: 'magic04', 38: 'mammographic_masses', 39: 'movement_libras', 40: 'no2', 
    #         41: 'plasma_retinol', 42: 'pm10', 43: 'schizo', 44: 'socmob', 
    #         45: 'solar-flare', 46: 'test_bng_cmc', 47: 'test_breast', 48: 'test_credit', 
    #         49: 'test_eucalyptus', 50: 'test_ilpd', 51: 'test_irish', 52: 'test_phoneme', 
    #         53: 'triazines', 54: 'veteran', 55: 'weatherAUS', 56: 'wilt'
    #     }

    #     test_classification_task_dic = {
    #         1: 'Accident_Casualties', 2: 'Frogs_MFCCs_family', 3: 'Frogs_MFCCs_gen', 4: 'Frogs_MFCCs_spec', 
    #         5: 'HTRU_2', 6: 'Indian Liver Patient Dataset (ILPD)', 7: 'Iris', 8: 'LendingClub Issued Loans', 
    #         9: 'Skin_NonSkin', 10: 'The_broken_machine', 11: 'Wine_classification', 12: 'adult', 
    #         13: 'analcatdata_broadwaymult', 14: 'analcatdata_germangss'
    #     }
    # ###################################################### v1
    # elif version == 1:
    #     train_classification_task_dic = {
    #         1: 'Accident_Casualties', 2: 'Frogs_MFCCs_family', 3: 'Frogs_MFCCs_gen', 4: 'Frogs_MFCCs_spec', 
    #         5: 'HTRU_2', 6: 'Indian Liver Patient Dataset (ILPD)', 7: 'Iris', 8: 'LendingClub Issued Loans', 
    #         9: 'Skin_NonSkin', 10: 'The_broken_machine', 11: 'Wine_classification', 12: 'adult', 
    #         13: 'analcatdata_broadwaymult', 14: 'analcatdata_germangss',
    #         29: 'diggle_table_a2', 30: 'disclosure_z', 31: 'glass', 32: 'haberman', 
    #         33: 'home_credit', 34: 'image segmentation', 35: 'kc3', 36: 'kidney', 
    #         37: 'magic04', 38: 'mammographic_masses', 39: 'movement_libras', 40: 'no2', 
    #         41: 'plasma_retinol', 42: 'pm10', 43: 'schizo', 44: 'socmob', 
    #         45: 'solar-flare', 46: 'test_bng_cmc', 47: 'test_breast', 48: 'test_credit', 
    #         49: 'test_eucalyptus', 50: 'test_ilpd', 51: 'test_irish', 52: 'test_phoneme', 
    #         53: 'triazines', 54: 'veteran', 55: 'weatherAUS', 56: 'wilt'
    #     }

    #     test_classification_task_dic = {
    #         15: 'ar4', 16: 'bank-full', 
    #         17: 'baseball', 18: 'biodeg', 19: 'blood-transfusion', 20: 'bodyfat', 
    #         21: 'braziltourism', 22: 'bureau', 23: 'car', 24: 'chatfield_4', 
    #         25: 'cmc', 26: 'crx', 27: 'data_banknote_authentication', 28: 'dermatology', 
    #     }
    # ####################################################### v2
    # elif version == 2:
    #     train_classification_task_dic = {
    #         1: 'Accident_Casualties', 2: 'Frogs_MFCCs_family', 3: 'Frogs_MFCCs_gen', 4: 'Frogs_MFCCs_spec', 
    #         5: 'HTRU_2', 6: 'Indian Liver Patient Dataset (ILPD)', 7: 'Iris', 8: 'LendingClub Issued Loans', 
    #         9: 'Skin_NonSkin', 10: 'The_broken_machine', 11: 'Wine_classification', 12: 'adult', 
    #         13: 'analcatdata_broadwaymult', 14: 'analcatdata_germangss',
    #         15: 'ar4', 16: 'bank-full', 
    #         17: 'baseball', 18: 'biodeg', 19: 'blood-transfusion', 20: 'bodyfat', 
    #         21: 'braziltourism', 22: 'bureau', 23: 'car', 24: 'chatfield_4', 
    #         25: 'cmc', 26: 'crx', 27: 'data_banknote_authentication', 28: 'dermatology', 
    #         43: 'schizo', 44: 'socmob', 
    #         45: 'solar-flare', 46: 'test_bng_cmc', 47: 'test_breast', 48: 'test_credit', 
    #         49: 'test_eucalyptus', 50: 'test_ilpd', 51: 'test_irish', 52: 'test_phoneme', 
    #         53: 'triazines', 54: 'veteran', 55: 'weatherAUS', 56: 'wilt'
    #     }

    #     test_classification_task_dic = {
    #         29: 'diggle_table_a2', 30: 'disclosure_z', 31: 'glass', 32: 'haberman', 
    #         33: 'home_credit', 34: 'image segmentation', 35: 'kc3', 36: 'kidney', 
    #         37: 'magic04', 38: 'mammographic_masses', 39: 'movement_libras', 40: 'no2', 
    #         41: 'plasma_retinol', 42: 'pm10', 
    #     }
    # ####################################################### v3
    # elif version == 3:
    #     train_classification_task_dic = {
    #         1: 'Accident_Casualties', 2: 'Frogs_MFCCs_family', 3: 'Frogs_MFCCs_gen', 4: 'Frogs_MFCCs_spec', 
    #         5: 'HTRU_2', 6: 'Indian Liver Patient Dataset (ILPD)', 7: 'Iris', 8: 'LendingClub Issued Loans', 
    #         9: 'Skin_NonSkin', 10: 'The_broken_machine', 11: 'Wine_classification', 12: 'adult', 
    #         13: 'analcatdata_broadwaymult', 14: 'analcatdata_germangss',
    #         15: 'ar4', 16: 'bank-full', 
    #         17: 'baseball', 18: 'biodeg', 19: 'blood-transfusion', 20: 'bodyfat', 
    #         21: 'braziltourism', 22: 'bureau', 23: 'car', 24: 'chatfield_4', 
    #         25: 'cmc', 26: 'crx', 27: 'data_banknote_authentication', 28: 'dermatology', 
    #         29: 'diggle_table_a2', 30: 'disclosure_z', 31: 'glass', 32: 'haberman', 
    #         33: 'home_credit', 34: 'image segmentation', 35: 'kc3', 36: 'kidney', 
    #         37: 'magic04', 38: 'mammographic_masses', 39: 'movement_libras', 40: 'no2', 
    #         41: 'plasma_retinol', 42: 'pm10', 
    #     }

    #     test_classification_task_dic = {
    #         43: 'schizo', 44: 'socmob', 
    #         45: 'solar-flare', 46: 'test_bng_cmc', 47: 'test_breast', 48: 'test_credit', 
    #         49: 'test_eucalyptus', 50: 'test_ilpd', 51: 'test_irish', 52: 'test_phoneme', 
    #         53: 'triazines', 54: 'veteran', 55: 'weatherAUS', 56: 'wilt'
    #     }





    classifier_predictor_list = [
        # LogisticRegressionPrim(),
        RandomForestClassifierPrim(),
        # AdaBoostClassifierPrim(),
        # BaggingClassifierPrim(),
        # BernoulliNBClassifierPrim(),
        # ComplementNBClassifierPrim(),
        DecisionTreeClassifierPrim(),
        # ExtraTreesClassifierPrim(),
        # GaussianNBClassifierPrim(),
        # GaussianProcessClassifierPrim(),
        # GradientBoostingClassifierPrim(),
        KNeighborsClassifierPrim(),
        # LinearDiscriminantAnalysisPrim(),
        # LinearSVCPrim(),
        LogisticRegressionPrim(),
        # LogisticRegressionCVPrim(),
        # MultinomialNBPrim(),
        # NearestCentroidPrim(),
        # PassiveAggressiveClassifierPrim(),
        # QuadraticDiscriminantAnalysisPrim(),
        # RidgeClassifierPrim(),
        # RidgeClassifierCVPrim(),
        # SGDClassifierPrim(),
        SVCPrim(),
        # XGBClassifierPrim(),
        # BalancedRandomForestClassifierPrim(),
        # EasyEnsembleClassifierPrim(),
        # RUSBoostClassifierPrim(),
        # LGBMClassifierPrim()
    ]

    metric_list = [
        AccuracyMetric(),
        F1Metric(),
        AucMetric(),
        MseMetric(),
    ]

    dtype_dic = {
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
    }

    imputernums = [ImputerMean(), ImputerMedian(), ImputerNumPrim()] # 填补数值列的缺失值。
    # 填补categorical列的缺失值。众数。
    encoders = [NumericDataPrim(), LabelEncoderPrim(), OneHotEncoderPrim()] #
    fpreprocessings = [MinMaxScalerPrim(), MaxAbsScalerPrim(), RobustScalerPrim(), StandardScalerPrim(), QuantileTransformerPrim(), PowerTransformerPrim(), NormalizerPrim(), KBinsDiscretizerOrdinalPrim(), Primitive()]
    fengines = [PolynomialFeaturesPrim(), InteractionFeaturesPrim(), PCA_AUTO_Prim(), IncrementalPCA_Prim(), KernelPCA_Prim(), TruncatedSVD_Prim(), RandomTreesEmbeddingPrim(), Primitive()]
    # fselections = [
    #     VarianceThresholdPrim(),         
    #     UnivariateSelectChiKbestPrim(), 
    #     f_classifKbestPrim(), mutual_info_classifKbestPrim(), f_classifPercentilePrim(), mutual_info_classifPercentilePrim(), UnivariateSelectChiFPRPrim(), f_classifFPRPrim(), f_classifFDRPrim(), UnivariateSelectChiFWEPrim(), f_classifFWEPrim(), Primitive()]
    fselections = [VarianceThresholdPrim(), Primitive()]
    lpipelines = [logic_pipeline_1, logic_pipeline_2, logic_pipeline_3, logic_pipeline_4, logic_pipeline_5, logic_pipeline_6]
    # fselections = [VarianceThresholdPrim(), Primitive()]
   
    # pipeline_action_dim: int = len(lpipelines)
    imputernum_action_dim: int = len(imputernums)
    encoder_action_dim: int = len(encoders)
    fpreprocessing_action_dim: int = len(fpreprocessings)
    fegine_action_dim: int = len(fengines)
    fselection_action_dim: int = len(fselections)
    lpipeline_action_dim: int = len(lpipelines)
    single_action_dim: int = max([imputernum_action_dim, encoder_action_dim, fpreprocessing_action_dim, fegine_action_dim, fselection_action_dim])

    # result_log_file_name: str = 'result_1_loss_log_tanh_v'+ str(version) +'.npy'
    # loss_log_file_name: str = 'loss_1_loss_log_tanh_v'+ str(version) +'.npy'
    # lp_loss_log_file_name: str = 'lp_1_oss_loss_log_tanh_v'+ str(version) +'.npy'
    # test_loss_log_file_name: str = 'test_1_loss_loss_log_tanh_v'+ str(version) +'.npy'
    # # test_reward_dic_file_name: str = 'test_dtc1_loss_reward_dic_tanh_v'+str(version) +'.npy'
    # test_reward_dic_file_name: str = 'test_2dtc_reward_dic_tanh_v' +str(version) +'.npy'
    # test_q_value_file_name: str = 'test_1_loss_q_value_v'+str(version) +'.npy'
    # outputdir: str = './train_2_tanh_v' + str(version)
    # dpgoutputdir: str = './dpg_v' + str(version)
    # outputdir1: str = './train_tanh_v' + str(version)

    if not os.path.exists('logs/'+str(version)+'_more_model'):
        os.mkdir('logs/'+str(version)+'_more_model')
    if not os.path.exists('models/'+str(version)+'_more_model'):
        os.mkdir('models/'+str(version)+'_more_model')
    result_log_file_name: str = 'logs/'+str(version)+'_more_model'+'/result_log.npy'
    loss_log_file_name: str = 'logs/'+str(version)+'_more_model'+'/loss_log.npy'
    lp_loss_log_file_name: str = 'logs/'+str(version)+'_more_model'+'/lp_loss_log.npy'
    test_reward_dic_file_name: str = 'logs/'+str(version)+'_more_model'+'/test_reward_dict.npy'
    model_dir: str = 'models/'+str(version)

    single_result_log_file_name: str = 'single_result_log_tanh_v'+ str(version) +'.npy'
    single_loss_log_file_name: str = 'single_loss_log_tanh_v'+ str(version) +'.npy'
    single_lp_loss_log_file_name: str = 'single_lp_loss_log_tanh_v'+ str(version) +'.npy'
    single_test_loss_log_file_name: str = 'single_test_loss_log_tanh_v'+ str(version) +'.npy'
    single_test_reward_dic_file_name: str = 'single_test_lg_reward_dic_tanh_v'+str(version) +'.npy'
    single_test_q_value_file_name: str = 'single_test_q_value_v'+str(version) +'.npy'
    single_outputdir: str = './single_train_tanh_v' + str(version)
