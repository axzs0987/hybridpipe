from .pipeline import Pipeline
import random
from MLPipeGen.core.config import Config
import numpy as np
import pandas as pd

from collections import defaultdict, OrderedDict, deque
import copy
import sys
# from ..primitives.data_preprocessing import ImputerOneHotEncoderPrim
import scipy.stats
from scipy.linalg import LinAlgError
import scipy.sparse
import sklearn
# TODO use balanced accuracy!
import sklearn.metrics
import sklearn.model_selection
from sklearn.utils import check_array
from sklearn.multiclass import OneVsRestClassifier
# from gym_atml.envs.metafeatures.OneHotEncoder import OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
# from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

import time
class Environment:
    def __init__(self, config: Config, train=True):
        self.config = config
        self.pipeline = None
        self.column_num = self.config.column_num
        self.state = None
        self.reward = None
        self.action = None
        self.next_state = None
        self.done = False
        self.train = train
        self.lpip_state = None

    def set_pipeline(self, taskid, predictor, metric, train=True):
        if self.pipeline:
            self.pipeline.reset_data()
            del self.pipeline
        self.pipeline = Pipeline(taskid, predictor, metric, self.config, train=train)

    def reset(self, taskid=None, predictor=None, metric=None, default=True):
        if default or taskid is None or predictor is None or metric is None :
            # taskid = random.choice(list(self.config.train_classification_task_dic.keys())) 
            taskid = random.choice(list(self.config.train_index)) 
            # if taskid in self.config.train_classification_task_dic.keys():
            # print('train_index', self.config.train_index)
            if taskid in self.config.train_index:
                predictor_id = random.randrange(1,len(self.config.classifier_predictor_list)+1)
                predictor = [i for i in self.config.classifier_predictor_list if i.id==predictor_id][0]
                metric = [i for i in self.config.metric_list if i.id==self.config.classification_metric_id][0]
            else:
                predictor_id = random.randrange(len(self.config.classifier_predictor_list)+1,len(self.config.classifier_predictor_list)+self.config.regression_dim+1)
                predictor = [i for i in self.config.classifier_predictor_list if i.id==predictor_id][0]
                metric = [i for i in self.config.metric_list if i.id==self.config.regression_metric_id][0]

        self.set_pipeline(taskid, predictor, metric, train=self.train)
        self.column_num = self.config.column_num
        self.lpip_state = self.get_lpip_state()
        # self.prim_tate = self.get_state()
        self.reward = None
        self.action = None
        self.next_state = None
        self.done = False

    def step(self, step):
        if self.pipeline is None:
            self.reset()
        self.prim_state = self.get_state()
        # self.data_feature, self.seq_feature, self.predictor_feature, self.logic_pipeline_feature
        # self.state = self.get_state()
        step_result = self.pipeline.add_step(step)
        # self.state = self.get_rnn_state()
        # print('after_add_pip', step_result)
        if step_result == -1:
            print('pipeline cant do action')
            return -1
        elif step_result == 0:
            print('step not fit')
            return 0
        # print('next data', self.pipeline.data_x)
        # self.next_state = self.get_state()
        self.next_prim_state = self.get_state()
        self.get_reward()
        self.is_done()
        self.action = step

        return self.prim_state, self.reward, self.next_prim_state, self.done
        
    def get_more_data_feature(self):
        # get data feature
        # inp_data = self.pipeline.data_x
        # print(self.pipeline.train_x.shape)
        # print(self.pipeline.test_x.shape)
        inp_data = pd.concat([self.pipeline.train_x, self.pipeline.test_x], ignore_index=True)
        if len(self.pipeline.train_y.shape) > 1:
            train_y = self.pipeline.train_y[0]
        else:
            train_y = self.pipeline.train_y
        categorical = list(self.pipeline.train_x.dtypes == object)
        column_info = {}
        # print(inp_data)
        # try:
        #     inp_data_temp = inp_data[0]
        # except:
        #     inp_data = np.array([inp_data[0]])

        for i in range(len(inp_data.columns)):
            # print(i)
            # print(inp_data.iloc[:,i])
            col = inp_data.iloc[:,i]
            if i >= self.column_num:
                break
            s_s = col

            column_info[i] = {}
            column_info[i]['col_name'] = 'unknown_' + str(i)
            column_info[i]['dtype'] = str(s_s.dtypes) # 1
            column_info[i]['length'] = len(s_s.values) # 2
            column_info[i]['null_ratio'] = s_s.isnull().sum() / len(s_s.values) # 3
            column_info[i]['ctype'] = 1 if inp_data.columns[i] in self.pipeline.num_cols else 2 # 4
            column_info[i]['nunique'] = s_s.nunique() # 5
            column_info[i]['nunique_ratio'] = s_s.nunique() / len(s_s.values) # 6
        
            if column_info[i]['ctype'] == 1:  # 如果是数字列
                column_info[i]['mean'] = 0 if np.isnan(s_s.describe()['mean']) or abs(s_s.describe()['mean'])==np.inf else s_s.describe()['mean'] # 7
                column_info[i]['std'] = 0 if np.isnan(s_s.describe()['std']) or abs(s_s.describe()['std'])==np.inf else s_s.describe()['std'] # 8

                column_info[i]['min'] = 0 if np.isnan(s_s.describe()['min']) or abs(s_s.describe()['min'])==np.inf else s_s.describe()['min'] # 9
                column_info[i]['25%'] = 0 if np.isnan(s_s.describe()['25%']) or abs(s_s.describe()['25%'])==np.inf else s_s.describe()['25%']
                column_info[i]['50%'] = 0 if np.isnan(s_s.describe()['50%']) or abs(s_s.describe()['50%'])==np.inf else s_s.describe()['50%']
                column_info[i]['75%'] = 0 if np.isnan(s_s.describe()['75%']) or abs(s_s.describe()['75%'])==np.inf else s_s.describe()['75%']
                column_info[i]['max'] = 0 if np.isnan(s_s.describe()['max']) or abs(s_s.describe()['max'])==np.inf else s_s.describe()['max']
                column_info[i]['median'] = 0 if np.isnan(s_s.median()) or abs(s_s.median())==np.inf else s_s.median()
                if len(s_s.mode()) != 0:
                    column_info[i]['mode'] = 0 if np.isnan(s_s.mode().iloc[0]) or abs(s_s.mode().iloc[0])==np.inf else s_s.mode().iloc[0]
                else:
                    column_info[i]['mode'] = 0
                column_info[i]['mode_ratio'] = 0 if np.isnan(s_s.astype('category').describe().iloc[3] / column_info[i]['length']) or abs(s_s.astype('category').describe().iloc[3] / column_info[i]['length'])==np.inf else s_s.astype('category').describe().iloc[3] / column_info[i]['length']
                column_info[i]['sum'] = 0 if np.isnan(s_s.sum()) or abs(s_s.sum())==np.inf else s_s.sum()
                column_info[i]['skew'] = 0 if np.isnan(s_s.skew()) or abs(s_s.skew())==np.inf else s_s.skew()
                column_info[i]['kurt'] = 0 if np.isnan(s_s.kurt()) or abs(s_s.kurt())==np.inf else s_s.kurt()

            elif column_info[i]['ctype'] == 2:  # category列
                column_info[i]['mean'] = 0
                column_info[i]['std'] = 0
                column_info[i]['min'] = 0
                column_info[i]['25%'] = 0
                column_info[i]['50%'] = 0
                column_info[i]['75%'] = 0
                column_info[i]['max'] = 0
                column_info[i]['median'] = 0
                column_info[i]['mode'] = 0
                column_info[i]['mode_ratio'] = 0
                column_info[i]['sum'] = 0
                column_info[i]['skew'] = 0
                column_info[i]['kurt'] = 0
        data_feature = []
        for index in column_info.keys():
            one_column_feature = []
            column_dic = column_info[index]
            for kw in column_dic.keys():
                if kw == 'col_name' or kw == 'content':
                    continue
                elif kw == 'dtype':
                    content = self.config.dtype_dic[column_dic[kw]]
                else:
                    content = column_dic[kw]
                one_column_feature.append(content)
            data_feature.append(one_column_feature)
        if len(column_info) < self.column_num:
            for index in range(len(column_info),self.column_num):
                one_column_feature = np.zeros(self.config.column_feature_dim)
                data_feature.append(one_column_feature)
        data_feature = np.ravel(np.array(data_feature))

        # NumberOfInstances
        instances_num = np.array([float(self.pipeline.train_x.shape[0])])
        # class entropy 


        train_y_1 = train_y.reshape((-1, 1))

        entropies = []
        for i in range(1):
            occurence_dict = defaultdict(float)
            for value in train_y_1[:, i]:
                occurence_dict[value] += 1
            entropies.append(scipy.stats.entropy([occurence_dict[key] for key in
                                                  occurence_dict], base=2))

        class_entropy = np.array([np.mean(entropies)])

        # SymbolsSum
        symbols_per_column = []
        for i, column in enumerate(self.pipeline.train_x.T):
            if i < len(categorical):
                if categorical[i]:
                    unique_values = np.unique(column)
                    num_unique = np.sum(np.isfinite(unique_values))
                    symbols_per_column.append(num_unique)
        
        symbols_num = np.array(symbols_per_column)
        symbols_sum = np.array(np.nansum(symbols_num))

        # symbols STD
        print("symbols std")
        values = [val for val in symbols_num if val > 0]
        if len(values) == 0:
            std = 0
        else:
            std = np.nanstd(values)
        symbols_std = np.array([std if np.isfinite(std) else 0])

        # symbols mean
        print("symbols mean")
        values = [val for val in symbols_num if val > 0]
        if len(values) == 0:
            mean = 0
        else:
            mean = np.nanmean(values)
        symbols_mean = np.array([mean])

        # SymbolsMax
        print("symbols max")
        values = symbols_num
        if len(values) == 0:
            symbols_max = np.array([0])
        else:
            symbols_max = np.array([max(max(values), 0)])

        # SymbolsMin
        print("symbols min")
        minimum = None
        for unique in symbols_num:
            if unique > 0 and (minimum is None or unique < minimum):
                minimum = unique
        symbols_min = np.array([minimum if minimum is not None else 0])

        # ClassProbabilitySTD"
        print("class_prob std")
        occurence_dict = defaultdict(float)

        for value in train_y:
            occurence_dict[float(value)] += 1
        class_occurences =  occurence_dict
        
        occurence_dict = class_occurences

        occurences = np.array([occurrence for occurrence in occurence_dict.values()],
                                dtype=np.float64)
        class_prob_std = (occurences / train_y.shape[0]).std()
        # class prob mean
        print("class_prob mean")
        occurence_dict = class_occurences

        occurences = np.array([occurrence for occurrence in occurence_dict.values()],
                                dtype=np.float64)
        class_prob_mean = np.array([(occurences / train_y.shape[0]).mean()])

        # class prob max
        print("class_prob max")
        occurences = class_occurences
        max_value = -1
        for num_occurences in occurences.values():
            if num_occurences > max_value:
                max_value = num_occurences
        class_prob_max = np.array([float(max_value) / float(train_y.shape[0])])

        # class prob min
        print("class_prob min")
        occurences = class_occurences

        min_value = np.iinfo(np.int64).max

        for num_occurences in occurences.values():
            if num_occurences < min_value:
                min_value = num_occurences
        class_prob_min = np.array([float(min_value) / float(train_y.shape[0])])

        # InverseDatasetRatio
        print("InverseDatasetRatio")
        features_num = float(self.pipeline.train_x.shape[1])
        inverse_dataset_ratio =  np.array([instances_num / features_num])

        # datasetRatio
        print("datasetRatio")
        dataset_ratio =  np.array([features_num / instances_num])

        # RatioNominalToNumerical
        print("RatioNominalToNumerical")
        category_feature_num = np.sum(categorical)
        numeric_feature_num = len(categorical) - np.sum(categorical)

        if numeric_feature_num == 0.0:
            ratio_nominal_to_numerical = np.array([0.])
        else:
            ratio_nominal_to_numerical = np.array([category_feature_num / numeric_feature_num])

        # RatioNumericalToNominal
        if numeric_feature_num == 0.0:
            ratio_numerical_to_nominal = np.array([0.])
        else:
            ratio_numerical_to_nominal = np.array([numeric_feature_num / category_feature_num])

        # NumberOfCategoricalFeatures
        categorical_features_num = np.array([category_feature_num])

        # NumberOfNumericFeatures
        numeric_features_num = np.array([numeric_feature_num])

        # NumberOfMissingValues
        try:
            missing = ~np.isfinite(self.pipeline.train_x)
        except:
            missing = np.array([0])
        # print(missing.sum())
        missing_values_number = np.array(missing.sum())

        # NumberOfFeaturesWithMissingValues
        num_missing = missing.sum(axis=0)
        try:
            features_with_missing_values = np.array([float(np.sum([1 if num > 0 else 0 for num in num_missing]))])
        except:
            features_with_missing_values = np.array([0])

        # NumberOfInstancesWithMissingValues
        num_missing = missing.sum(axis=1)
        instances_with_missing_values = np.array([float(np.sum([1 if num > 0 else 0 for num in num_missing]))])

        # NumberOfFeatures
        features_num_array = np.array([features_num])

        # NumberOfClasses
        class_num = float(len(np.unique(train_y)))
        class_num_array = np.array([class_num])

        # NumberOfInstances
        instances_num_array = np.array([instances_num])

        # LogInverseDatasetRatio
        log_inverse_dataset_ratio = np.array([np.log(inverse_dataset_ratio[0])])

        # LogDatasetRatio
        log_dataset_ratio = np.array([np.log(dataset_ratio[0])])

        # PercentageOfMissingValues
        percentage_of_missing_values = np.array([float(missing_values_number[0]) / \
               float(self.pipeline.train_x.shape[0] * self.pipeline.train_x.shape[1])])

        # PercentageOfFeaturesWithMissingValues
        percentage_of_features_with_missing_values = np.array([features_with_missing_values[0] \
                     / float(features_num)])

        # PercentageOfInstancesWithMissingValues
        percentage_of_instances_with_missing_values = np.array([instances_with_missing_values[0] \
                     / float(instances_num)])

        # LogNumberOfFeatures
        log_features_num = np.array([np.log(features_num)])

        # LogNumberOfInstances
        log_instances_num = np.array([np.log(instances_num)])

        # # LandmarkRandomNodeLearner
        import sklearn.tree


        kf = sklearn.model_selection.StratifiedKFold(n_splits=2)
        accuracy = 0.

        for train, test in kf.split(self.pipeline.train_x, train_y):
            random_state = sklearn.utils.check_random_state(42)
            node = sklearn.tree.DecisionTreeClassifier(
                criterion="entropy", max_depth=1, random_state=random_state,
                min_samples_split=2, min_samples_leaf=1, max_features=1)
            node.fit(self.pipeline.train_x[train], train_y[train])
            predictions = node.predict(X[test])
            accuracy += sklearn.metrics.accuracy_score(predictions, train_y[test])
        landmark_random_node_learner = accuracy / 2

        # landmark dtc
        import sklearn.tree

        kf = sklearn.model_selection.StratifiedKFold(n_splits=2)
        accuracy = 0.
        for train, test in kf.split(self.pipeline.train_x, y):
            tree = sklearn.tree.DecisionTreeClassifier(random_state=0)
            tree.fit(self.pipeline.train_x[train], train_y[train])


            predictions = tree.predict(self.pipeline.train_x[test])
            accuracy += sklearn.metrics.accuracy_score(predictions, train_y[test])
        landmark_dtc = accuracy / 2

        # LandmarkNaiveBayes
        import sklearn.naive_bayes
        kf = sklearn.model_selection.StratifiedKFold(n_splits=2)

        accuracy = 0.
        for train, test in kf.split(self.pipeline.train_x, train_y):
            nb = sklearn.naive_bayes.GaussianNB()
            nb.fit(self.pipeline.train_x[train], train_y[train])

            predictions = nb.predict(self.pipeline.train_x[test])
            accuracy += sklearn.metrics.accuracy_score(predictions, train_y[test])
        landmark_knaive_bayes = accuracy / 2

        # SkewnessSTD
        for i in range(self.pipeline.train_x.shape[1]):
            if not categorical[i]:
                skews.append(scipy.stats.skew(self.pipeline.train_x[:, i]))
        skew = skews
        std = np.nanstd(skews) if len(skews) > 0 else 0
        skews_std = np.array([std if np.isfinite(std) else 0])
        
        #SkewnessMean
        mean = np.nanmean(skews) if len(skews) > 0 else 0
        skews_mean = np.array([mean if np.isfinite(mean) else 0])

        #SkewnessMax
        maximum = np.nanmax(skews) if len(skews) > 0 else 0
        skews_max = np.array([maximum if np.isfinite(maximum) else 0])

        #SkewnessMin
        minimum = np.nanmin(skews) if len(skews) > 0 else 0
        skews_min = np.array([minimum if np.isfinite(minimum) else 0])

        # KutStd
        kurts = []
        for i in range(self.pipeline.train_x.shape[1]):
            if not categorical[i]:
                kurts.append(scipy.stats.kurtosis(self.pipeline.train_x[:, i]))

        std = np.nanstd(kurts) if len(kurts) > 0 else 0
        kurts_std = std if np.isfinite(std) else 0
        kurts_std = np.array([kurts_std])

        # kurt mean
        mean = np.nanmean(kurts) if len(kurts) > 0 else 0
        kurts_mean =np.array([mean if np.isfinite(mean) else 0])

        # KurtosisMax
        maximum = np.nanmax(kurts) if len(kurts) > 0 else 0
        kurts_max = np.array([maximum if np.isfinite(maximum) else 0])

        # kurt min
        minimum = np.nanmin(kurts) if len(kurts) > 0 else 0
        kurts_min = np.array([minimum if np.isfinite(minimum) else 0])

        data_feature = np.concatenate((data_feature, class_entropy, symbols_sum, symbols_std, symbols_mean, symbols_max, symbols_min, class_prob_std, class_prob_mean, class_prob_max, class_prob_min, inverse_dataset_ratio, dataset_ratio, ratio_nominal_to_numerical, ratio_numerical_to_nominal, categorical_features_num, numeric_features_num, missing_values_number, features_with_missing_values, instances_with_missing_values, instances_with_missing_values, features_num_array, class_num_array, instances_num_array, log_inverse_dataset_ratio, log_dataset_ratio, percentage_of_missing_values, percentage_of_features_with_missing_values, percentage_of_instances_with_missing_values, log_features_num, log_instances_num, landmark_random_node_learner, landmark_dtc, landmark_knaive_bayes, skews_std, skews_mean, skews_max, skews_min, kurts_std, kurts_mean, kurts_max, kurts_min))
        print("end data featurex")
        del inp_data
        del column_info
        return data_feature




    def get_data_feature(self):
        # get data feature
        # inp_data = self.pipeline.data_x
        # print(self.pipeline.train_x.shape)
        # print(self.pipeline.test_x.shape)
        inp_data = pd.DataFrame(self.pipeline.train_x)
        # inp_data = pd.concat([self.pipeline.train_x, self.pipeline.test_x], ignore_index=True, axis=0)
        if len(self.pipeline.train_y.shape) > 1:
            train_y = self.pipeline.train_y[0]
        else:
            train_y = self.pipeline.train_y
        categorical = list(self.pipeline.train_x.dtypes == object)
        column_info = {}
        # print(inp_data)
        # try:
        #     inp_data_temp = inp_data[0]
        # except:
        #     inp_data = np.array([inp_data[0]])

        for i in range(len(inp_data.columns)):
            # print(i)
            # print(inp_data.iloc[:,i])
            col = inp_data.iloc[:,i]
            if i >= self.column_num:
                break
            s_s = col

            column_info[i] = {}
            column_info[i]['col_name'] = 'unknown_' + str(i)
            column_info[i]['dtype'] = str(s_s.dtypes) # 1
            column_info[i]['length'] = len(s_s.values) # 2
            column_info[i]['null_ratio'] = s_s.isnull().sum() / len(s_s.values) # 3
            column_info[i]['ctype'] = 1 if inp_data.columns[i] in self.pipeline.num_cols else 2 # 4
            column_info[i]['nunique'] = s_s.nunique() # 5
            column_info[i]['nunique_ratio'] = s_s.nunique() / len(s_s.values) # 6
        
            if 'mean' not in s_s.describe():
                column_info[i]['ctype'] = 2
            if column_info[i]['ctype'] == 1:  # 如果是数字列
                
                column_info[i]['mean'] = 0 if np.isnan(s_s.describe()['mean']) or abs(s_s.describe()['mean'])==np.inf else s_s.describe()['mean'] # 7
                column_info[i]['std'] = 0 if np.isnan(s_s.describe()['std']) or abs(s_s.describe()['std'])==np.inf else s_s.describe()['std'] # 8

                column_info[i]['min'] = 0 if np.isnan(s_s.describe()['min']) or abs(s_s.describe()['min'])==np.inf else s_s.describe()['min'] # 9
                column_info[i]['25%'] = 0 if np.isnan(s_s.describe()['25%']) or abs(s_s.describe()['25%'])==np.inf else s_s.describe()['25%']
                column_info[i]['50%'] = 0 if np.isnan(s_s.describe()['50%']) or abs(s_s.describe()['50%'])==np.inf else s_s.describe()['50%']
                column_info[i]['75%'] = 0 if np.isnan(s_s.describe()['75%']) or abs(s_s.describe()['75%'])==np.inf else s_s.describe()['75%']
                column_info[i]['max'] = 0 if np.isnan(s_s.describe()['max']) or abs(s_s.describe()['max'])==np.inf else s_s.describe()['max']
                column_info[i]['median'] = 0 if np.isnan(s_s.median()) or abs(s_s.median())==np.inf else s_s.median()
                if len(s_s.mode()) != 0:
                    column_info[i]['mode'] = 0 if np.isnan(s_s.mode().iloc[0]) or abs(s_s.mode().iloc[0])==np.inf else s_s.mode().iloc[0]
                else:
                    column_info[i]['mode'] = 0
                column_info[i]['mode_ratio'] = 0 if np.isnan(s_s.astype('category').describe().iloc[3] / column_info[i]['length']) or abs(s_s.astype('category').describe().iloc[3] / column_info[i]['length'])==np.inf else s_s.astype('category').describe().iloc[3] / column_info[i]['length']
                column_info[i]['sum'] = 0 if np.isnan(s_s.sum()) or abs(s_s.sum())==np.inf else s_s.sum()
                column_info[i]['skew'] = 0 if np.isnan(s_s.skew()) or abs(s_s.skew())==np.inf else s_s.skew()
                column_info[i]['kurt'] = 0 if np.isnan(s_s.kurt()) or abs(s_s.kurt())==np.inf else s_s.kurt()

            elif column_info[i]['ctype'] == 2:  # category列
                column_info[i]['mean'] = 0
                column_info[i]['std'] = 0
                column_info[i]['min'] = 0
                column_info[i]['25%'] = 0
                column_info[i]['50%'] = 0
                column_info[i]['75%'] = 0
                column_info[i]['max'] = 0
                column_info[i]['median'] = 0
                column_info[i]['mode'] = 0
                column_info[i]['mode_ratio'] = 0
                column_info[i]['sum'] = 0
                column_info[i]['skew'] = 0
                column_info[i]['kurt'] = 0
        data_feature = []
        for index in column_info.keys():
            one_column_feature = []
            column_dic = column_info[index]
            for kw in column_dic.keys():
                if kw == 'col_name' or kw == 'content':
                    continue
                elif kw == 'dtype':
                    content = self.config.dtype_dic[column_dic[kw]]
                else:
                    content = column_dic[kw]
                one_column_feature.append(content)
            data_feature.append(one_column_feature)
        if len(column_info) < self.column_num:
            for index in range(len(column_info),self.column_num):
                one_column_feature = np.zeros(self.config.column_feature_dim)
                data_feature.append(one_column_feature)
        data_feature = np.ravel(np.array(data_feature))

        del inp_data
        del column_info
        return data_feature

    def get_lpip_state(self):
        data_feature = self.get_data_feature()
        predictor = np.array([self.pipeline.predictor.id])
        state = np.concatenate((data_feature, predictor))
        return state

    def get_state(self):
        data_feature = self.get_data_feature()
        # get seq feature
        # print('pip.seq', [i.name for i in self.pipeline.sequence])
        # sequence = [i.id for i in self.pipeline.sequence]
        # for i in range(len(self.config.lpipelines[0])):
        #     if i >= len(self.pipeline.sequence):
        #         sequence += [-1]
        sequence = np.array(self.pipeline.gsequence)
        # get predictor feature
        predictor = np.array([self.pipeline.predictor.id-1])
        # get pipeline feature
        # if self.pipeline.logic_pipeline_id:
        logic_pipeline_id = np.array([self.pipeline.logic_pipeline_id])
        # else:
        #     logic_pipeline_id = np.array([-1])
        state = np.concatenate((data_feature, sequence, predictor, logic_pipeline_id))
        return state

    def get_reward(self):
        if len(self.pipeline.sequence) < 6:
            self.reward = 0
        else:
            self.end_time = time.time()
            self.reward = self.pipeline.evaluate()
            

    def is_done(self):
        if len(self.pipeline.sequence) < 6:
            self.done = False
        elif len(self.pipeline.sequence) == 6:
            self.done = True
        else:
            print('bad sequence')
            return

    def has_nan(self):
        has_num_nan = False
        has_cat_nan = False
        def catch_num(data):
            
            num_cols = [col for col in data.columns if str(data[col].dtypes) != 'object']
            # num_cols.sort()
            cat_cols = [col for col in data.columns if col not in num_cols]
            # print(data.values)
            cat_train_x = data[cat_cols]
            num_train_x = data[num_cols]
            return cat_train_x, num_train_x

        
        with pd.option_context('mode.use_inf_as_null', True):
            cat_train_x, num_train_x = catch_num(self.pipeline.train_x)
            cat_test_x, num_test_x = catch_num(self.pipeline.test_x)
            if len(self.pipeline.cat_cols) != 0:
                
                # print('self.pipline.catcols', self.pipeline.cat_cols)
                # print('self.pipline.data_x', self.pipeline.data_x)
                # cat_data = pd.DataFrame(self.pipeline.train_x, columns = self.pipeline.cat_cols)
                
                if cat_train_x.isnull().any().any():
                    has_cat_nan = True
                # cat_data = pd.DataFrame(self.pipeline.test_x, columns = self.pipeline.cat_cols)
                
                if cat_test_x.isnull().any().any():
                    has_cat_nan = True
            if len(self.pipeline.num_cols) != 0:
                # num_data = pd.DataFrame(self.pipeline.train_x, columns = self.pipeline.num_cols)
                if num_train_x.isnull().any().any():
                    has_num_nan = True
                # num_data = pd.DataFrame(self.pipeline.test_x, columns = self.pipeline.num_cols)
                if num_test_x.isnull().any().any():
                    has_num_nan = True
        return has_num_nan, has_cat_nan

    def has_cat_cols(self):
        if not len(self.pipeline.cat_cols) == 0:
            return True
        else:
            return False