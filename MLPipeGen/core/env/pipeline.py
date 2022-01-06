import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .primitives.predictor import *
from MLPipeGen.core.config import Config

class Pipeline:
    def __init__(self, taskid, predictor, metric, config: Config, train=True):
        self.config = config
        self.taskid = taskid
        self.metric = metric
        self.predictor = predictor
        self.train = train

        self.code = ''
        self.result = 0
        self.sequence = []
        self.index = 0

        self.data_x = None
        self.data_y = None
        self.train_x = None
        self.test_x = None
        self.train_y = None
        self.test_y = None
        self.pred_y = None

        self.num_cols = None
        self.cat_cols = None

        self.load_data(taskid)
        self.logic_pipeline_id = None
        self.gsequence = [26,26,26,26,26,26]
        # self.init_code(taskid)
    
    def reset_data(self):
        self.data_x = None
        self.data_y = None
        self.train_x = None
        self.test_x = None
        self.train_y = None
        self.test_y = None
        self.pred_y = None

        del self.data_x
        del self.data_y
        del self.train_x
        del self.test_x
        del self.train_y
        del self.test_y
        del self.pred_y
        del self.config
        del self.taskid
        del self.metric
        del self.predictor
        del self.train
        del self.num_cols
        del self.cat_cols

        del self.code
        del self.result
        del self.sequence
        del self.index

    def get_index(self):
        return self.index
        
    def add_step(self, step): # step is a Primitive
        # print(self.index)
        if self.index >= len(self.config.lpipelines[self.logic_pipeline_id]):
            # print('pipeline compeleted')
            return -1

        pre_pipeline = []
        if self.index > 0:
            for ind in range(self.index):
                pre_pipeline.append(ind)
        if step.type in pre_pipeline or not step.can_accept(self.train_x) or not step.can_accept(self.test_x) or (not step.is_needed(self.train_x) and not step.is_needed(self.test_x)):
            return 0
        
        try:
            self.train_x, self.test_x = step.transform(self.train_x, self.test_x, self.train_y)


            self.num_cols = list(self.train_x._get_numeric_data().columns)
            self.cat_cols = list(set(self.train_x.columns) - set(self.num_cols))
        
        except Exception as e:
            # print('transform fail:', e)
            return 0
        # print('444')
        self.sequence.append(step)
        self.gsequence[self.index] = step.gid

        self.index += 1
        # print('555')
        return 1

    # def load_data(self, taskid, ratio=0.8, split_random_state=0):
    #     if self.train:
    #         data = pd.read_csv(self.config.dataset_path+self.config.train_classification_task_dic[taskid]+'.csv').infer_objects()
    #     else:
    #         data = pd.read_csv(self.config.dataset_path+self.config.classification_task_dic[taskid]+'.csv').infer_objects()
    #     # data = shuffle(data, random_state=0).reset_index(drop=True)
    #     data = data.replace([np.inf, -np.inf], np.nan)
    #     data.dropna(subset=[data.iloc[:, -1].name])
    #     if data.shape[0] > 1500:
    #         data = data.iloc[: 1500, :]
    #     self.data_x = data.iloc[:, :-1]
    #     self.data_y = data.iloc[:, -1].values
    #     # print(self.data_y.dtype)

    #     # print(self.data_y)
    #     self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.data_x, self.data_y, train_size=ratio, test_size=1-ratio, random_state=split_random_state)
    #     # print(self.train_y)
    #     if str(self.data_y.dtype) == 'Object':
    #         self.data_y = LabelEncoder().fit_transform(self.data_y)
    #     del data
    #     self.num_cols = list(self.train_x._get_numeric_data().columns)
    #     self.cat_cols = list(set(self.train_x) - set(self.num_cols))

    def load_data(self, taskid, ratio=0.8, split_random_state=0):
        # if self.train:
        #     # data = pd.read_csv(self.config.dataset_path+self.config.train_classification_task_dic[taskid]+'.csv').infer_objects()
        #     # data = pd.read_csv(self.config.dataset_path+self.config.train_classification_task_dic[taskid]).infer_objects()
        #     data = pd.read_csv(self.config.dataset_path+self.config.classification_task_dic[taskid]['dataset'] + '/' + self.config.classification_task_dic[taskid]['csv_file']).infer_objects()
        # else:
        #     # data = pd.read_csv(self.config.dataset_path+self.config.classification_task_dic[taskid]+'.csv').infer_objects()
        data = pd.read_csv(self.config.dataset_path+self.config.classification_task_dic[taskid]['dataset'] + '/' + self.config.classification_task_dic[taskid]['csv_file']).infer_objects()
        # data = shuffle(data, random_state=0).reset_index(drop=True)
        label_index = int(self.config.classification_task_dic[taskid]['label'])

        data = data.replace([np.inf, -np.inf], np.nan)
        data.dropna(subset=[data.iloc[:, label_index].name])
        # if data.shape[0] > 1500 and self.train:
        if data.shape[0] > 1500 and self.train:
            data = data.iloc[: 1500, :]
        
        self.data_y = data.iloc[:, label_index].values
        self.data_x = data.drop(columns = [data.columns[label_index]], axis = 1)
        # self.data_x = data.iloc[:, :-1]
        # self.data_y = data.iloc[:, -1].values
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.data_x, self.data_y, train_size=ratio, test_size=1-ratio, random_state=split_random_state)
        
        del data
        self.num_cols = list(self.train_x._get_numeric_data().columns)
        self.cat_cols = list(set(self.train_x) - set(self.num_cols))

        if str(self.data_y.dtype) == 'Object':
            self.data_y = LabelEncoder(self.data_y)


    def evaluate(self):
        if len(self.sequence) < 6:
            # print('pipeline not compeleted')
            return
        # print('test_y', self.data_y)
        try:
            self.pred_y = self.predictor.transform(self.train_x, self.train_y, self.test_x)
            self.result = self.metric.evaluate(self.pred_y, self.test_y)
        except Exception as e:
            # print("\033[1;31m:" + str(e)+"\033[0m")
            self.result = -1
        # print("\033[1;33mtaskid:" + str(self.taskid) + ' '+str(self.result) +"\033[0m")
        return self.result

    # def init_code(self, abs_path):
    #     code = 'import pandas as pd\n'
    #     code += 'from sklearn.preprocessing import LabelEncoder\n'
    #     code += 'data = pd.read_csv(\''+ abs_path +'/' +self.config.train_classification_task_dic[taskid]+ '.csv\').infer_objects()\n'
    #     code += 'data = shuffle(data, random_state=0).reset_index(drop=True)\n'
    #     code += 'if data.shape[0] > 1500:\n'
    #     code += '    data = data.iloc[: 1500, :]\n'
    #     code += 'data_x = data.iloc[:, :-1]\n'
    #     code += 'data_y = data.iloc[:, -1].values)\n'
    #     code += 'if str(data_y.dtype) == \'Object\':\n'
    #     code += '    data_y = LabelEncoder(data_y)\n'