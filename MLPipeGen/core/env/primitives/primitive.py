import pandas as pd

class Primitive:
    def __init__(self, name='blank'):
        self.id = 0
        self.gid = 25 #36
        self.name = name
        self.description = str(name)
        self.hyperparams = []
        self.type = "blank"

    def fit(self, data):
        pass

    def transform(self,train_x, test_x, train_y):
        return train_x, test_x

    def can_accept(self, data):
        return True

    def can_accept_a(self, data): #可以接受不为空的，存在数字列的df
        if data.empty:
            return False
        elif data.shape[1] == 0:
            return False
        num_cols = data._get_numeric_data().columns
        if not len(num_cols) == 0:
            return True
        return False

    def can_accept_b(self, data): #可以接受不为空的df
        if data.empty:
            return False
        elif data.shape[1] == 0:
            return False
        return True

    def can_accept_c(self, data, task=None, larpack=False): # 可以接受不为空的，不存在nan，不存在cat列的df
        if data.empty:
            # print('empty')
            return False
        elif data.shape[1] == 0:
            # print('shape1 = 0')
            return False
        cols = data
        num_cols = data._get_numeric_data().columns
        cat_cols = list(set(cols) - set(num_cols))

        # if larpack:
        #     if min(data.shape[0], data.shape[1]) - 1 == 0:
        #         print('larpack')
        #         return False
        # if data.isnull().any().any():
        #     return False
        with pd.option_context('mode.use_inf_as_null', True):
            if data.isnull().any().any():
                # print('has nan')
                return False
        if not len(cat_cols) == 0:
            # print('has cat')
            return False
        return True

    def can_accept_c1(self, data, task=None, larpack=False): # 可以接受不为空的，不存在nan，不存在cat列的df
        if data.empty:
            return False
        elif data.shape[1] == 0:
            return False
        cols = data
        num_cols = data._get_numeric_data().columns
        cat_cols = list(set(cols) - set(num_cols))

        if not len(cat_cols) == 0:
            return False
        return True

    def can_accept_c2(self, data, task=None, larpack=False): # 可以接受不为空的，不存在nan，不存在cat列的df
        if data.empty:
            return False
        elif data.shape[1] == 0:
            return False
        cols = data
        num_cols = data._get_numeric_data().columns
        cat_cols = list(set(cols) - set(num_cols))

        if not len(num_cols) == 0:
            return False
        return True

    def can_accept_d(self, data, task): # 不为空，不存在cat列，不存在nan
        if data.empty:
            return False
        elif data.shape[1] == 0:
            return False
        cols = data
        num_cols = data._get_numeric_data().columns
        cat_cols = list(set(cols) - set(num_cols))
        if not len(cat_cols) == 0:
            return False

        with pd.option_context('mode.use_inf_as_null', True):
            # print('data.lt(0)', data.lt(0))
            if data.isnull().any().any():
                return False
            # elif data.lt(0).sum().sum() > 0:
                # return False
            return True

    def is_needed(self, data):
        return True
