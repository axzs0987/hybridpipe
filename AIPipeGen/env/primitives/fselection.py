from .primitive import Primitive
from sklearn.feature_selection import VarianceThreshold, GenericUnivariateSelect, chi2, SelectKBest, f_classif,\
    mutual_info_classif, f_regression, mutual_info_regression, SelectPercentile, SelectFpr, SelectFdr, SelectFwe, RFE
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
import pandas as pd
from copy import deepcopy
from itertools import compress
import numpy as np
np.random.seed(1)

class VarianceThresholdPrim(Primitive):
    def __init__(self, random_state=0):
        super(VarianceThresholdPrim, self).__init__(name='VarianceThreshold')
        self.id = 1
        self.gid = 24
        self.PCA_LAPACK_Prim = []
        self.type = 'feature selection'
        self.description = "Feature selector that removes all low-variance features."
        self.selector = VarianceThreshold()
        self.accept_type = 'c_t'
        self.need_y = True
    def can_accept(self, data):
        return self.can_accept_c(data)

    def is_needed(self, data):
        # if data.shape[1] < 3:
        #     return False
        return True

    def transform(self, train_x, test_x, train_y):
        self.selector.fit(train_x)

        cols = list(train_x.columns)
        mask = self.selector.get_support(indices=False)
        final_cols = list(compress(cols, mask))
        train_data_x = pd.DataFrame(self.selector.transform(train_x), columns=final_cols)

        cols = list(test_x.columns)
        mask = self.selector.get_support(indices=False)
        final_cols = list(compress(cols, mask))
        test_data_x = pd.DataFrame(self.selector.transform(test_x), columns=final_cols)
        return train_data_x, test_data_x


class UnivariateSelectChiKbestPrim(Primitive):
    def __init__(self, random_state=0):
        super(UnivariateSelectChiKbestPrim, self).__init__(name='UnivariateSelectChiKbest')
        self.id = 2
        self.gid = 26
        self.PCA_LAPACK_Prim = []
        self.type = 'feature selection'
        self.description = "Select features according to the k highest scores with Chi-square"
        self.hyperparams_run = {'default': True}
        self.selector = None
        self.accept_type = 'd'
        self.need_y = True

    def can_accept(self, data):
        return self.can_accept_d(data, 'Classification')

    def is_needed(self, data):
        if data.shape[1] < 3:
            return False
        return True

    def transform(self, train_x, test_x, train_y):
        k = 10
        if train_x.shape[1] < k: k = 'all'
        self.selector = SelectKBest(chi2, k=k)
        self.selector.fit(train_x, train_y)

        cols = list(train_x.columns)
        mask = self.selector.get_support(indices=False)
        final_cols = list(compress(cols, mask))
        train_data_x = pd.DataFrame(self.selector.transform(train_x), columns=final_cols)

        cols = list(test_x.columns)
        mask = self.selector.get_support(indices=False)
        final_cols = list(compress(cols, mask))
        test_data_x = pd.DataFrame(self.selector.transform(test_x), columns=final_cols)
        return train_data_x, test_data_x


class f_classifKbestPrim(Primitive):
    def __init__(self, random_state=0):
        super(f_classifKbestPrim, self).__init__(name='f_classifKbest')
        self.id = 3
        self.gid = 27
        self.PCA_LAPACK_Prim = []
        self.type = 'feature selection'
        self.description = "Select features according to the k highest scores with ANOVA F-value between label/feature for classification tasks."
        self.hyperparams_run = {'default': True}
        self.selector = None
        self.accept_type = 'c'
        self.need_y = True

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        if data.shape[1] < 3:
            return False
        return True

    def transform(self, train_x, test_x, train_y):
        k = 10
        if train_x.shape[1] < k: k = 'all'
        self.selector = SelectKBest(f_classif, k=k)
        self.selector.fit(train_x, train_y)

        cols = list(train_x.columns)
        mask = self.selector.get_support(indices=False)
        final_cols = list(compress(cols, mask))
        train_data_x = pd.DataFrame(self.selector.transform(train_x), columns=final_cols)

        cols = list(test_x.columns)
        mask = self.selector.get_support(indices=False)
        final_cols = list(compress(cols, mask))
        test_data_x = pd.DataFrame(self.selector.transform(test_x), columns=final_cols)
        return train_data_x, test_data_x

class mutual_info_classifKbestPrim(Primitive):
    def __init__(self, random_state=0):
        super(mutual_info_classifKbestPrim, self).__init__(name='mutual_info_classifKbest')
        self.id = 4
        self.gid = 28
        self.PCA_LAPACK_Prim = []
        self.type = 'feature selection'
        self.description = "Select features according to the k highest scores with Mutual information for a discrete target."
        self.hyperparams_run = {'default': True}
        self.selector = None
        self.accept_type = 'c'
        self.need_y = True

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        if data.shape[1] < 3:
            return False
        return True

    def transform(self, train_x, test_x, train_y):
        k = 10
        if train_x.shape[1] < k: k = 'all'
        self.selector = SelectKBest(mutual_info_classif, k=k)
        self.selector.fit(train_x, train_y)

        cols = list(train_x.columns)
        mask = self.selector.get_support(indices=False)
        final_cols = list(compress(cols, mask))
        train_data_x = pd.DataFrame(self.selector.transform(train_x), columns=final_cols)

        cols = list(test_x.columns)
        mask = self.selector.get_support(indices=False)
        final_cols = list(compress(cols, mask))
        test_data_x = pd.DataFrame(self.selector.transform(test_x), columns=final_cols)
        return train_data_x, test_data_x

# class f_regressionKbestPrim(Primitive):
#     def __init__(self, random_state=0):
#         super(f_regressionKbestPrim, self).__init__(name='f_regressionKbest')
#         self.id = 5
#         self.PCA_LAPACK_Prim = []
#         self.type = 'feature selection'
#         self.description = "Select features according to the k highest scores with F-value between label/feature for regression tasks."
#         self.hyperparams_run = {'default': True}
#         self.selector = None
#         self.accept_type = 'c_r'
#         self.need_y = True

#     def can_accept(self, data):
#         return self.can_accept_c(data, 'Regression')

#     def is_needed(self, data):
#         if data.shape[1] < 3:
#             return False
#         return True

#     def transform(self, data, data_y):
#         k = 10
#         if self.hyperparams_run['default']:
#             if train_x.shape[1] < k: k = 'all'
#         self.selector = SelectKBest(f_regression, k=k)
#         self.selector.fit(data, data_y)

#         cols = list(data.columns)
#         mask = self.selector.get_support(indices=False)
#         final_cols = list(compress(cols, mask))
#         data = pd.DataFrame(self.selector.transform(data), columns=final_cols)
#         return data


# class mutual_info_regressionKbestPrim(Primitive):
#     def __init__(self, random_state=0):
#         super(mutual_info_regressionKbestPrim, self).__init__(name='mutual_info_regressionKbest')
#         self.id = 1
#         self.PCA_LAPACK_Prim = []
#         self.type = 'feature selection'
#         self.description = "Select features according to the k highest scores with mutual information for a continuous target."
#         self.hyperparams_run = {'default': True}
#         self.selector = None
#         self.accept_type = 'c_r'
#         self.need_y = False

#     def can_accept(self, data):
#         return self.can_accept_c(data, 'Regression')

#     def is_needed(self, data):
#         if data.shape[1] < 3:
#             return False
#         return True

#     def transform(self, data):
#         k = 10
#         if self.hyperparams_run['default']:
#             if train_x.shape[1] < k: k = 'all'
#         self.selector = SelectKBest(mutual_info_regression, k=k)
#         self.selector.fit(data, data_y)
#         cols = list(data.columns)
#         mask = self.selector.get_support(indices=False)
#         final_cols = list(compress(cols, mask))
#         data = pd.DataFrame(self.selector.transform(data), columns=final_cols)
#         return data


# class UnivariateSelectChiPercentilePrim(Primitive):
#     def __init__(self, random_state=0):
#         super(UnivariateSelectChiPercentilePrim, self).__init__(name='UnivariateSelectChiPercentile')
#         self.id = 2
#         self.PCA_LAPACK_Prim = []
#         self.type = 'feature selection'
#         self.description = "Select features according to a percentile of the highest scores with Chi-square"
#         self.hyperparams_run = {'default': True}
#         self.selector = None
#         self.accept_type = 'd'
#         self.need_y = False

#     def can_accept(self, data):
#         return self.can_accept_d(data, 'Classification')

#     def is_needed(self, data):
#         if data.shape[1] < 3:
#             return False
#         return True

#     def transform(self, data):
#         self.selector = SelectPercentile(chi2)
#         self.selector.fit(data, data_y)
#         cols = list(data.columns)
#         mask = self.selector.get_support(indices=False)
#         final_cols = list(compress(cols, mask))
#         data = pd.DataFrame(self.selector.transform(data), columns=final_cols)
#         return data


class f_classifPercentilePrim(Primitive):
    def __init__(self, random_state=0):
        super(f_classifPercentilePrim, self).__init__(name='f_classifPercentile')
        self.id = 5
        self.gid = 29
        self.PCA_LAPACK_Prim = []
        self.type = 'feature selection'
        self.description = "Select features according to a percentile of the highest scores with ANOVA F-value between label/feature for classification tasks."
        self.hyperparams_run = {'default': True}
        self.selector = None
        self.accept_type = 'c'
        self.need_y = False

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        if data.shape[1] < 3:
            return False
        return True

    def transform(self, train_x, test_x, train_y):
        self.selector = SelectPercentile(f_classif)
        self.selector.fit(train_x, train_y)

        cols = list(train_x.columns)
        mask = self.selector.get_support(indices=False)
        final_cols = list(compress(cols, mask))
        train_data_x = pd.DataFrame(self.selector.transform(train_x), columns=final_cols)

        cols = list(test_x.columns)
        mask = self.selector.get_support(indices=False)
        final_cols = list(compress(cols, mask))
        test_data_x = pd.DataFrame(self.selector.transform(test_x), columns=final_cols)
        return train_data_x, test_data_x
    
class mutual_info_classifPercentilePrim(Primitive):
    def __init__(self, random_state=0):
        super(mutual_info_classifPercentilePrim, self).__init__(name='mutual_info_classifPercentile')
        self.id = 6
        self.gid = 30
        self.PCA_LAPACK_Prim = []
        self.type = 'feature selection'
        self.description = "Select features according to a percentile of the highest scores with Mutual information for a discrete target."
        self.hyperparams_run = {'default': True}
        self.selector = None
        self.accept_type = 'c'
        self.need_y = False

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        if data.shape[1] < 3:
            return False
        return True

    def transform(self, train_x, test_x, train_y):
        self.selector = SelectPercentile(mutual_info_classif)
        self.selector.fit(train_x, train_y)

        cols = list(train_x.columns)
        mask = self.selector.get_support(indices=False)
        final_cols = list(compress(cols, mask))
        train_data_x = pd.DataFrame(self.selector.transform(train_x), columns=final_cols)

        cols = list(test_x.columns)
        mask = self.selector.get_support(indices=False)
        final_cols = list(compress(cols, mask))
        test_data_x = pd.DataFrame(self.selector.transform(test_x), columns=final_cols)
        return train_data_x, test_data_x

# class f_regressionPercentilePrim(Primitive):
#     def __init__(self, random_state=0):
#         super(f_regressionPercentilePrim, self).__init__(name='f_regressionPercentile')
#         self.id = 8
#         self.PCA_LAPACK_Prim = []
#         self.type = 'feature selection'
#         self.description = "Select features according to a percentile of the highest scores with F-value between label/feature for regression tasks."
#         self.hyperparams_run = {'default': True}
#         self.selector = None
#         self.accept_type = 'c_r'
#         self.need_y = False

#     def can_accept(self, data):
#         return self.can_accept_c(data, 'Regression')

#     def is_needed(self, data):
#         if data.shape[1] < 3:
#             return False
#         return True

#     def transform(self, data):
#         self.selector = SelectPercentile(f_regression)
#         self.selector.fit(data, data_y)
#         cols = list(data.columns)
#         mask = self.selector.get_support(indices=False)
#         final_cols = list(compress(cols, mask))
#         data = pd.DataFrame(self.selector.transform(data), columns=final_cols)

#         return data


# class mutual_info_regressionPercentilePrim(Primitive):
#     def __init__(self, random_state=0):
#         super(mutual_info_regressionPercentilePrim, self).__init__(name='mutual_info_regressionPercentile')
#         self.id = 9
#         self.PCA_LAPACK_Prim = []
#         self.type = 'feature selection'
#         self.description = "Select features according to a percentile of the highest scores with mutual information for a continuous target."
#         self.hyperparams_run = {'default': True}
#         self.selector = None
#         self.accept_type = 'c_r'
#         self.need_y = False

#     def can_accept(self, data):
#         return self.can_accept_c(data, 'Regression')

#     def is_needed(self, data):
#         if data.shape[1] < 3:
#             return False
#         return True

#     def transform(self, data):
#         self.selector = SelectPercentile(mutual_info_regression)
#         self.selector.fit(data, data_y)
#         cols = list(data.columns)
#         mask = self.selector.get_support(indices=False)
#         final_cols = list(compress(cols, mask))
#         data = pd.DataFrame(self.selector.transform(data), columns=final_cols)

#         return data


class UnivariateSelectChiFPRPrim(Primitive):
    def __init__(self, random_state=0):
        super(UnivariateSelectChiFPRPrim, self).__init__(name='UnivariateSelectChiFPR')
        self.id = 7
        self.gid = 31
        self.PCA_LAPACK_Prim = []
        self.type = 'feature selection'
        self.description = "Filter: Select the pvalues below alpha based on a FPR test with Chi-square. FPR test stands for False Positive Rate test. It controls the total amount of false detections."
        self.hyperparams_run = {'default': True}
        self.selector = None
        self.accept_type = 'd'
        self.need_y = False

    def can_accept(self, data):
        return self.can_accept_d(data, 'Classification')

    def is_needed(self, data):
        if data.shape[1] < 3:
            return False
        return True

    def transform(self, train_x, test_x, train_y):
        self.selector = SelectFpr(chi2, alpha=0.05)
        self.selector.fit(train_x, train_y)

        try:
            cols = list(train_x.columns)
            mask = self.selector.get_support(indices=False)
            final_cols = list(compress(cols, mask))
            train_data_x = pd.DataFrame(self.selector.transform(train_x), columns=final_cols)

            cols = list(test_x.columns)
            mask = self.selector.get_support(indices=False)
            final_cols = list(compress(cols, mask))
            test_data_x = pd.DataFrame(self.selector.transform(test_x), columns=final_cols)
        except Exception as e:
            print(e)
        return train_data_x, test_data_x

class f_classifFPRPrim(Primitive):
    def __init__(self, random_state=0):
        super(f_classifFPRPrim, self).__init__(name='f_classifFPR')
        self.id = 8
        self.gid = 32
        self.PCA_LAPACK_Prim = []
        self.type = 'feature selection'
        self.description = "Filter: Select the pvalues below alpha based on a FPR test with ANOVA F-value between label/feature for classification tasks. FPR test stands for False Positive Rate test. It controls the total amount of false detections."
        self.hyperparams_run = {'default': True}
        self.selector = None
        self.accept_type = 'c'
        self.need_y = False

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        if data.shape[1] < 3:
            return False
        return True

    def transform(self, train_x, test_x, train_y):
        self.selector = SelectFpr(f_classif, alpha=0.05)
        self.selector.fit(train_x, train_y)

        try:
            cols = list(train_x.columns)
            mask = self.selector.get_support(indices=False)
            final_cols = list(compress(cols, mask))
            train_data_x = pd.DataFrame(self.selector.transform(train_x), columns=final_cols)

            cols = list(test_x.columns)
            mask = self.selector.get_support(indices=False)
            final_cols = list(compress(cols, mask))
            test_data_x = pd.DataFrame(self.selector.transform(test_x), columns=final_cols)
        except Exception as e:
            print(e)
        return train_data_x, test_data_x

# class f_regressionFPRPrim(Primitive):
#     def __init__(self, random_state=0):
#         super(f_regressionFPRPrim, self).__init__(name='f_regressionFPR')
#         self.id = 9
#         self.PCA_LAPACK_Prim = []
#         self.type = 'feature selection'
#         self.description = "Filter: Select the pvalues below alpha based on a FPR test with F-value between label/feature for regression tasks. FPR test stands for False Positive Rate test. It controls the total amount of false detections."
#         self.hyperparams_run = {'default': True}
#         self.selector = None
#         self.accept_type = 'c_r'
#         self.need_y = False

#     def can_accept(self, data):
#         return self.can_accept_c(data, 'Regression')

#     def is_needed(self, data):
#         if data.shape[1] < 3:
#             return False
#         return True

#     def transform(self, data):
#         self.selector = SelectFpr(f_regression)
#         self.selector.fit(data, data_y)
#         cols = list(data.columns)
#         mask = self.selector.get_support(indices=False)
#         final_cols = list(compress(cols, mask))
#         data = pd.DataFrame(self.selector.transform(data), columns=final_cols)

#         return data


# class UnivariateSelectChiFDRPrim(Primitive):
#     def __init__(self, random_state=0):
#         super(UnivariateSelectChiFDRPrim, self).__init__(name='UnivariateSelectChiFDR')
#         self.id = 9
#         self.PCA_LAPACK_Prim = []
#         self.type = 'feature selection'
#         self.description = "Filter: Select the p-values for an estimated false discovery rate with Chi-square. This uses the Benjamini-Hochberg procedure. alpha is an upper bound on the expected false discovery rate."
#         self.hyperparams_run = {'default': True}
#         self.selector = None
#         self.accept_type = 'd'
#         self.need_y = False

#     def can_accept(self, data):
#         return self.can_accept_d(data, 'Classification')

#     def is_needed(self, data):
#         if data.shape[1] < 3:
#             return False
#         return True

#     def transform(self, train_x, test_x, train_y):
#         self.selector = SelectFdr(chi2, alpha=0.05)
#         self.selector.fit(train_x, train_y)

#         try:
#             cols = list(train_x.columns)
#             mask = self.selector.get_support(indices=False)
#             final_cols = list(compress(cols, mask))
#             train_data_x = pd.DataFrame(self.selector.transform(train_x), columns=final_cols)

#             cols = list(test_x.columns)
#             mask = self.selector.get_support(indices=False)
#             final_cols = list(compress(cols, mask))
#             test_data_x = pd.DataFrame(self.selector.transform(test_x), columns=final_cols)
#         except Exception as e:
#             print(e)
#         return train_data_x, test_data_x

class f_classifFDRPrim(Primitive):
    def __init__(self, random_state=0):
        super(f_classifFDRPrim, self).__init__(name='f_classifFDR')
        self.id = 9
        self.gid = 33
        self.PCA_LAPACK_Prim = []
        self.type = 'feature selection'
        self.description = "Filter: Select the p-values for an estimated false discovery rate with ANOVA F-value between label/feature for classification tasks. This uses the Benjamini-Hochberg procedure. alpha is an upper bound on the expected false discovery rate."
        self.hyperparams_run = {'default': True}
        self.selector = None
        self.accept_type = 'c'
        self.need_y = False

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        if data.shape[1] < 3:
            return False
        return True

    def transform(self, train_x, test_x, train_y):
        self.selector = SelectFdr(f_classif)
        self.selector.fit(train_x, train_y)

        try:
            cols = list(train_x.columns)
            mask = self.selector.get_support(indices=False)
            final_cols = list(compress(cols, mask))
            train_data_x = pd.DataFrame(self.selector.transform(train_x), columns=final_cols)

            cols = list(test_x.columns)
            mask = self.selector.get_support(indices=False)
            final_cols = list(compress(cols, mask))
            test_data_x = pd.DataFrame(self.selector.transform(test_x), columns=final_cols)
        except Exception as e:
            print(e)
        return train_data_x, test_data_x

# class f_regressionFDRPrim(Primitive):
#     def __init__(self, random_state=0):
#         super(f_regressionFDRPrim, self).__init__(name='f_regressionFDR')
#         self.id = 12
#         self.PCA_LAPACK_Prim = []
#         self.type = 'feature selection'
#         self.description = "Filter: Select the p-values for an estimated false discovery rate with F-value between label/feature for regression tasks. This uses the Benjamini-Hochberg procedure. alpha is an upper bound on the expected false discovery rate."
#         self.hyperparams_run = {'default': True}
#         self.selector = None
#         self.accept_type = 'c_r'
#         self.need_y = False

#     def can_accept(self, data):
#         return self.can_accept_c(data, 'Regression')

#     def is_needed(self, data):
#         if data.shape[1] < 3:
#             return False
#         return True

#     def transform(self, data):
#         self.selector = SelectFdr(f_regression)
#         self.selector.fit(data, data_y)

#         cols = list(data.columns)
#         mask = self.selector.get_support(indices=False)
#         final_cols = list(compress(cols, mask))
#         data = pd.DataFrame(self.selector.transform(data), columns=final_cols)

#         return data


class UnivariateSelectChiFWEPrim(Primitive):
    def __init__(self, random_state=0):
        super(UnivariateSelectChiFWEPrim, self).__init__(name='UnivariateSelectChiFWE')
        self.id = 10
        self.gid = 34
        self.PCA_LAPACK_Prim = []
        self.type = 'feature selection'
        self.description = "Select the p-values corresponding to Family-wise error rate with Chi-square."
        self.hyperparams_run = {'default': True}
        self.selector = None
        self.accept_type = 'd'
        self.need_y = False

    def can_accept(self, data):
        return self.can_accept_d(data, 'Classification')

    def is_needed(self, data):
        if data.shape[1] < 3:
            return False
        return True

    def transform(self, train_x, test_x, train_y):
        self.selector = SelectFwe(chi2, alpha=0.05)
        self.selector.fit(train_x, train_y)

        try:
            cols = list(train_x.columns)
            mask = self.selector.get_support(indices=False)
            final_cols = list(compress(cols, mask))
            train_data_x = pd.DataFrame(self.selector.transform(train_x), columns=final_cols)

            cols = list(test_x.columns)
            mask = self.selector.get_support(indices=False)
            final_cols = list(compress(cols, mask))
            test_data_x = pd.DataFrame(self.selector.transform(test_x), columns=final_cols)
        except Exception as e:
            print(e)
        return train_data_x, test_data_x


class f_classifFWEPrim(Primitive):
    def __init__(self, random_state=0):
        super(f_classifFWEPrim, self).__init__(name='f_classifFWE')
        self.id = 11
        self.gid = 35
        self.PCA_LAPACK_Prim = []
        self.type = 'feature selection'
        self.description = "Select the p-values corresponding to Family-wise error rate with ANOVA F-value between label/feature for classification tasks."
        self.hyperparams_run = {'default': True}
        self.selector = None
        self.accept_type = 'c'
        self.need_y = False

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        if data.shape[1] < 3:
            return False
        return True

    def transform(self, train_x, test_x, train_y):
        self.selector = SelectFwe(f_classif, alpha=0.05)
        self.selector.fit(train_x, train_y)

        try:
            cols = list(train_x.columns)
            mask = self.selector.get_support(indices=False)
            final_cols = list(compress(cols, mask))
            train_data_x = pd.DataFrame(self.selector.transform(train_x), columns=final_cols)

            cols = list(test_x.columns)
            mask = self.selector.get_support(indices=False)
            final_cols = list(compress(cols, mask))
            test_data_x = pd.DataFrame(self.selector.transform(test_x), columns=final_cols)
        except Exception as e:
            print(e)
        return train_data_x, test_data_x


# class f_regressionFWEPrim(Primitive):
#     def __init__(self, random_state=0):
#         super(f_regressionFWEPrim, self).__init__(name='f_regressionFWE')
#         self.id = 15
#         self.PCA_LAPACK_Prim = []
#         self.type = 'feature selection'
#         self.description = "Select the p-values corresponding to Family-wise error rate with F-value between label/feature for regression tasks."
#         self.hyperparams_run = {'default': True}
#         self.selector = None
#         self.accept_type = 'c_r'
#         self.need_y = False

#     def can_accept(self, data):
#         return self.can_accept_c(data, 'Regression')

#     def is_needed(self, data):
#         if data.shape[1] < 3:
#             return False
#         return True

#     def transform(self, data):
#         self.selector = SelectFwe(f_regression, alpha=0.05)
#         self.selector.fit(data, data_y)

#         cols = list(data.columns)
#         try:
#             mask = self.selector.get_support(indices=False)
#             final_cols = list(compress(cols, mask))
#             data = pd.DataFrame(self.selector.transform(data), columns=final_cols)
#         except Exception as e:
#             print(e)

#         return data


# class RFE_RandomForestPrim(Primitive):
#     def __init__(self, random_state=0):
#         super(RFE_RandomForestPrim, self).__init__(name='RFE_RandomForest')
#         self.id = 12
#         self.gid = 36
#         self.PCA_LAPACK_Prim = []
#         self.type = 'feature selection'
#         self.description = "Feature ranking with recursive feature elimination with Random-Forest classifier. Given an external estimator that assigns weights to features (e.g., the coefficients of a linear model), the goal of recursive feature elimination (RFE) is to select features by recursively considering smaller and smaller sets of features. First, the estimator is trained on the initial set of features and the importance of each feature is obtained either through a coef_ attribute or through a feature_importances_ attribute. Then, the least important features are pruned from current set of features. That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached."
#         self.hyperparams_run = {'default': True}
#         self.random_state = random_state
#         self.selector = RFE(RandomForestClassifier(random_state=self.random_state))
#         self.accept_type = 'c'
#         self.need_y = False

#     def can_accept(self, data):
#         return self.can_accept_c(data, 'Classification')

#     def is_needed(self, data):
#         if data.shape[1] < 3:
#             return False
#         return True

#     def transform(self, train_x, test_x, train_y):
#         self.selector.fit(train_x, train_y)

#         cols = list(train_x.columns)
#         mask = self.selector.get_support(indices=False)
#         final_cols = list(compress(cols, mask))
#         train_data_x = pd.DataFrame(self.selector.transform(train_x), columns=final_cols)

#         cols = list(test_x.columns)
#         mask = self.selector.get_support(indices=False)
#         final_cols = list(compress(cols, mask))
#         test_data_x = pd.DataFrame(self.selector.transform(test_x), columns=final_cols)
#         return train_data_x, test_data_x


# class RFE_GradientBoostingPrim(Primitive):
#     def __init__(self, random_state=0):
#         super(RFE_GradientBoostingPrim, self).__init__(name='RFE_GradientBoosting')
#         self.id = 13
#         self.gid = 37
#         self.PCA_LAPACK_Prim = []
#         self.type = 'feature selection'
#         self.description = "Feature ranking with recursive feature elimination with Gradient-Boosting classifier. Given an external estimator that assigns weights to features (e.g., the coefficients of a linear model), the goal of recursive feature elimination (RFE) is to select features by recursively considering smaller and smaller sets of features. First, the estimator is trained on the initial set of features and the importance of each feature is obtained either through a coef_ attribute or through a feature_importances_ attribute. Then, the least important features are pruned from current set of features. That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached."
#         self.hyperparams_run = {'default': True}
#         self.selector = RFE(GradientBoostingClassifier(n_estimators=20))
#         self.accept_type = 'c'
#         self.need_y = False

#     def can_accept(self, data):
#         return self.can_accept_c(data, 'Classification')

#     def is_needed(self, data):
#         if data.shape[1] < 3:
#             return False
#         return True

#     def transform(self, train_x, test_x, train_y):
#         self.selector.fit(train_x, train_y)

#         cols = list(train_x.columns)
#         mask = self.selector.get_support(indices=False)
#         final_cols = list(compress(cols, mask))
#         train_data_x = pd.DataFrame(self.selector.transform(train_x), columns=final_cols)

#         cols = list(test_x.columns)
#         mask = self.selector.get_support(indices=False)
#         final_cols = list(compress(cols, mask))
#         test_data_x = pd.DataFrame(self.selector.transform(test_x), columns=final_cols)
#         return train_data_x, test_data_x

# class RFE_SVRPrim(Primitive):
#     def __init__(self, random_state=0):
#         super(RFE_SVRPrim, self).__init__(name='RFE_SVR')
#         self.id = 18
#         self.PCA_LAPACK_Prim = []
#         self.type = 'feature selection'
#         self.description = "Feature ranking with recursive feature elimination with SVR regressor. Given an external estimator that assigns weights to features (e.g., the coefficients of a linear model), the goal of recursive feature elimination (RFE) is to select features by recursively considering smaller and smaller sets of features. First, the estimator is trained on the initial set of features and the importance of each feature is obtained either through a coef_ attribute or through a feature_importances_ attribute. Then, the least important features are pruned from current set of features. That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached."
#         self.hyperparams_run = {'default': True}
#         self.selector = RFE(SVR(kernel="linear"))
#         self.accept_type = 'c_r'
#         self.need_y = False

#     def can_accept(self, data):
#         return self.can_accept_c(data, 'Regression')

#     def is_needed(self, data):
#         if data.shape[1] < 3:
#             return False
#         return True

#     def transform(self, data):
        
#         self.selector.fit(data, data_y)

#         cols = list(data.columns)
#         mask = self.selector.get_support(indices=False)
#         final_cols = list(compress(cols, mask))
#         data = pd.DataFrame(self.selector.transform(data), columns=final_cols)

#         return data


# class RFE_RandomForestRegPrim(Primitive):
#     def __init__(self, random_state=0):
#         super(RFE_RandomForestRegPrim, self).__init__(name='RFE_RandomForestReg')
#         self.id = 19
#         self.PCA_LAPACK_Prim = []
#         self.type = 'feature selection'
#         self.description = "Feature ranking with recursive feature elimination with Random-Forest regressor. Given an external estimator that assigns weights to features (e.g., the coefficients of a linear model), the goal of recursive feature elimination (RFE) is to select features by recursively considering smaller and smaller sets of features. First, the estimator is trained on the initial set of features and the importance of each feature is obtained either through a coef_ attribute or through a feature_importances_ attribute. Then, the least important features are pruned from current set of features. That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached."
#         self.hyperparams_run = {'default': True}
#         self.selector = RFE(RandomForestRegressor())
#         self.accept_type = 'c_r'
#         self.need_y = False

#     def can_accept(self, data):
#         return self.can_accept_c(data, 'Regression')

#     def is_needed(self, data):
#         if data.shape[1] < 3:
#             return False
#         return True

#     def transform(self, data):  
#         self.selector.fit(data, data_y)
#         cols = list(data.columns)
#         mask = self.selector.get_support(indices=False)
#         final_cols = list(compress(cols, mask))
#         data = pd.DataFrame(self.selector.transform(data), columns=final_cols)

#         return data

# Add more RFE Primitives!

# Add  SelectFromModel Primitives!
