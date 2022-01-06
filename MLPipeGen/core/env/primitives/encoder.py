from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy.sparse import csr_matrix
from sklearn.compose import ColumnTransformer
from .primitive import Primitive
import pandas as pd
from copy import deepcopy
import numpy as np

def catch_num(data):
    num_cols = [col for col in data.columns if str(data[col].dtypes) != 'object']
    num_cols.sort()
    cat_cols = [col for col in data.columns if col not in num_cols]
    # print(data.values)
    cat_train_x = data[cat_cols]
    num_train_x = data[num_cols]
    return cat_train_x, num_train_x

class NumericDataPrim(Primitive):
    def __init__(self, random_state=0):
        super(NumericDataPrim, self).__init__(name='NumericData')
        self.id = 1
        self.gid = 6
        self.hyperparams = []
        self.type = 'Encoder'
        self.description = "Extracts only numeric data columns from input."
        self.accept_type = 'a'
        self.need_y = False

    def can_accept(self, data):
        return self.can_accept_a(data)

    def is_needed(self, data):
        cols = data.columns
        num_cols = data._get_numeric_data().columns
        if not len(cols) == len(num_cols):
            return True
        return False

    def transform(self, train_x, test_x, train_y):
        num_cols = train_x._get_numeric_data().columns
        train_x = train_x[num_cols]
        num_cols = test_x._get_numeric_data().columns
        test_x = test_x[num_cols]
        return train_x, test_x

class OneHotEncoderPrim(Primitive):
    # can handle missing values. turns nans to extra category
    def __init__(self, random_state=0):
        super(OneHotEncoderPrim, self).__init__(name='OneHotEncoder')
        self.id = 2
        self.gid = 7
        self.hyperparams = []
        self.type = 'data preprocess'
        self.description = "Encode categorical integer features as a one-hot numeric array. The input to this transformer should be an array-like of integers or strings, denoting the values taken on by categorical (discrete) features. The features are encoded using a one-hot (aka ‘one-of-K’ or ‘dummy’) encoding scheme. This creates a binary column for each category and returns a sparse matrix or dense array. By default, the encoder derives the categories based on the unique values in each feature. Alternatively, you can also specify the categories manually. The OneHotEncoder previously assumed that the input features take on values in the range [0, max(values)). This behaviour is deprecated. This encoding is needed for feeding categorical data to many scikit-learn estimators, notably linear models and SVMs with the standard kernels."
        self.preprocess = OneHotEncoder()
        self.accept_type = 'c2'
        self.need_y = False

    def can_accept(self, data):
        cols = data
        num_cols = data._get_numeric_data().columns
        cat_cols = list(set(cols) - set(num_cols))
        if len(cat_cols) > 15:
            return False
        return True

    def is_needed(self, data):
        # data = handle_data(data)
        cols = data
        num_cols = data._get_numeric_data().columns
        cat_cols = list(set(cols) - set(num_cols))
        if len(cat_cols) == 0:
            return False
        return True

    def transform(self, train_x, test_x, train_y):
        cat_trainX, num_trainX = catch_num(train_x)
        cat_testX, num_testX = catch_num(test_x)

        # cat_trainX = deepcopy(cat_trainX)
        cat_cols = cat_trainX.columns
        # self.preprocess = ColumnTransformer([("one_hot", OneHotEncoder(handle_unknown='ignore'), list(cat_trainX.columns))])
        self.preprocess = ColumnTransformer([("one_hot", OneHotEncoder(handle_unknown='ignore'), cat_cols)])
        # self.preprocess.fit(cat_trainX)  # .astype(str)

        dummies = []
        len_trainx = num_trainX.shape[0]
        len_testx = num_testX.shape[0]
        # print('onehotencoder train_x', train_x)
        # print('onehotencoder test_x', test_x)
        # print('len_trainx', len_trainx)
        # print('len_testx', len_testx)
        for col in cat_cols:
            # print('col',col)
            temp = pd.get_dummies(pd.concat([cat_trainX[col], cat_testX[col]], axis=0).reset_index(drop=True), prefix=col)
            # print('temp',temp)
            train_d = temp.iloc[0:len_trainx,:].reset_index(drop=True)
            # print('len_train_d', train_d.shape[0])
            # print('train_d', train_d)
            test_d = temp.iloc[len_trainx:,:].reset_index(drop=True)
            # print('len_test_d', test_d.shape[0])
            # print('test_d', test_d)

            # self.preprocess.fit(temp)
            # train_d = pd.DataFrame(self.preprocess.transform(cat_trainX[col]))
            train_x = pd.concat([train_x.reset_index(drop=True), train_d], axis=1).reset_index(drop=True)
            # train_x.drop(col)
            # test_d = pd.DataFrame(self.preprocess.transform(cat_testX[col]))
            test_x = pd.concat([test_x.reset_index(drop=True), test_d], axis=1).reset_index(drop=True)
            # test_x.drop(col)
        # print('train_x.columns', train_x.columns)
        # print('get dummies before drop', train_x)
        train_x= train_x.drop(columns = cat_cols).infer_objects()
        # print('test_x.columns', test_x.columns)
        # print('get dummies after drop', train_x)
        test_x = test_x.drop(columns = cat_cols).infer_objects()
        # cat_trainX = self.preprocess.transform(cat_trainX)
        # cat_testX = self.preprocess.transform(cat_testX)
        # if isinstance(cat_trainX, csr_matrix):
        #     cat_trainX = cat_trainX.toarray()
        # if isinstance(cat_testX, csr_matrix):
        #     cat_testX = cat_testX.toarray()
        # cat_trainX = pd.DataFrame(cat_trainX, columns=self.preprocess.get_feature_names()).infer_objects()
        # cat_testX = pd.DataFrame(cat_testX, columns=self.preprocess.get_feature_names()).infer_objects()
        # # num_trainX1 = pd.DataFrame(self.imp.transform(cat_trainX), columns=cols).reset_index(drop=True).infer_objects()
        # cat_trainX = pd.DataFrame(cat_trainX).infer_objects()
        # cat_testX = pd.DataFrame(cat_testX).infer_objects()
        # cat_trainX = cat_trainX.iloc[:,~cat_trainX.columns.duplicated()]
        # cat_testX = cat_testX.iloc[:,~cat_testX.columns.duplicated()]
        # # cat_trainX = pd.DataFrame(dict(X=cat_trainX))
        # train_data_x = pd.concat([cat_trainX.reset_index(drop=True), num_trainX.reset_index(drop=True)],axis=1)
        # test_data_x = pd.concat([cat_testX.reset_index(drop=True), num_testX.reset_index(drop=True)],axis=1)
        return train_x, test_x

class LabelEncoderPrim(Primitive):
    # can handle missing values. Operates on all categorical features.
    def __init__(self, random_state=0):
        super(LabelEncoderPrim, self).__init__(name='LabelEncoder')
        self.id = 3
        self.gid = 8
        self.hyperparams = []
        self.type = 'data preprocess'
        self.description = "Encode labels with value between 0 and n_classes-1."
        self.preprocess = {}
        self.accept_type = 'b'
        self.need_y = False

    def can_accept(self, data):
        return True

    def is_needed(self, data):
        # data = handle_data(data)
        cols = data
        num_cols = data._get_numeric_data().columns
        cat_cols = list(set(cols) - set(num_cols))
        if len(cat_cols) == 0:
            return False
        return True
    
    def transform(self, train_x, test_x, train_y):
        cat_trainX, num_trainX = catch_num(train_x)
        # print('num_trainx', num_trainX)
        cat_testX, num_testX = catch_num(test_x)
        cols = cat_trainX.columns

        for col in cols:
            self.preprocess[col] = LabelEncoder()
            train_arr = self.preprocess[col].fit_transform(cat_trainX[col].astype(str))
            test_arr = self.preprocess[col].fit_transform(cat_testX[col].astype(str))
            cat_trainX[col] = train_arr
            cat_testX[col] = test_arr

        cat_trainX = cat_trainX.infer_objects()
        cat_trainX = cat_trainX.iloc[:,~cat_trainX.columns.duplicated()]
        # print('cat_trainX', cat_trainX)
        train_data_x = pd.concat([cat_trainX.reset_index(drop=True), num_trainX.reset_index(drop=True)],axis=1).infer_objects()

        cat_testX = cat_testX.infer_objects()
        cat_testX = cat_testX.iloc[:,~cat_testX.columns.duplicated()]
        test_data_x = pd.concat([cat_testX.reset_index(drop=True), num_testX.reset_index(drop=True)],axis=1).infer_objects()
        # print('labelencoder', train_data_x)
        return train_data_x, test_data_x