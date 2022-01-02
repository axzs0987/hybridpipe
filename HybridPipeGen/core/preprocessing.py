import os
from HybridPipeGen.core.read_ipynb import ipynb2py
import ast
import astunparse
import numpy as np
import traceback
import sys
import json
import random
import time
from func_timeout import func_set_timeout
import func_timeout
from HybridPipeGen.core.remove_model import remove_model
from HybridPipeGen.core.remove_model import remove_model_2
import gc
import shutil
if os.path.exists('runed.npy'):
    runed = list(np.load('runed.npy', allow_pickle=True))
else:
    runed = []
specific_split = 0
class Logger(object):
    def __init__(self, filename='HybridPipeGen/core/human_run_continue_'+str(specific_split)+'.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger('HybridPipeGen/core/human_run_continue.log', sys.stdout)
sys.stderr = Logger('HybridPipeGen/core/human_run_continue.log', sys.stderr)

# dataset_root_path = '/home/datamanager/dataset/statsklearn/dataset/'
dataset_root_path = 'data/dataset/'
accuracy_import_code : dict= {
        'accuracy_score': 'from sklearn.metrics import accuracy_score',
        'f1_score': 'from sklearn.metrics import f1_score',
        'mean_absolute_error': 'from sklearn.metrics import mean_absolute_error',
    }

model_import_code : dict = {
        'LogisticRegression': 'from sklearn.linear_model.logistic import LogisticRegression',
        'RandomForestClassifier': 'from sklearn.ensemble import RandomForestClassifier',
        # 'LinearRegression': 'from sklearn.linear_model import LinearRegression',
        'KNeighborsClassifier': 'from sklearn.neighbors import KNeighborsClassifier',
        'SVC': 'from sklearn.svm import SVC',
        'DecisionTreeClassifier': 'from sklearn.tree import DecisionTreeClassifier',
    }


error_dict = {}

def exchange_code(code):
    if '.fit_sample(' in code:
        code = code.replace('.fit_sample(', '.fit_resample(')
    if 'from keras.utils import plot_model' in code:
        code = code.replace('from keras.utils import plot_model', "from keras.utils.vis_utils import plot_model")
    # if not os.path.exists('HybridPipeGen/core/tmpdata/runned_notebook/'+ str(self.notebook_id) + '.py'):
    if 'from keras.utils import to_categorical' in code:
        # code = code.replace('from keras.utils import to_categorical\n', 'from keras import utils as np_utils\nnp_utils.to_categorical()\n')
        code = code.replace('from keras.utils import to_categorical', 'from sklearn.model_selection import train_test_split')
    if 'sklearn.cross_validation' in code:
        # code = code.replace('from keras.utils import to_categorical\n', 'from keras import utils as np_utils\nnp_utils.to_categorical()\n')
        code = code.replace('sklearn.cross_validation', 'sklearn.model_selection')
    if 'sklearn.grid_search' in code:
        # code = code.replace('from keras.utils import to_categorical\n', 'from keras import utils as np_utils\nnp_utils.to_categorical()\n')
        code = code.replace('sklearn.grid_search', 'sklearn.model_selection')
    if 'from pandas.tools.plotting' in code:
        # code = code.replace('from keras.utils import to_categorical\n', 'from keras import utils as np_utils\nnp_utils.to_categorical()\n')
        code = code.replace('from pandas.tools.plotting', 'from pandas.plotting')
    if 'convert_objects(convert_numeric=True)' in code:
        code = code.replace('convert_objects(convert_numeric=True)','apply(pd.to_numeric, errors="ignore")')
    if 'from plotly import plotly' in code:
        code = code.replace('from plotly import plotly','from chart_studio import plotly')
    if 'optimizers.SGD' in code:
        code = code.replace('optimizers.SGD','optimizers.gradient_descent_v2.SGD')
    if 'from sklearn.externals import joblib' in code:
        code = code.replace('from sklearn.externals import joblib','import joblib')
    if 'time.clock()' in code:
        code = code.replace("time.clock()","time.perf_counter()")
    if "plotly.plotly" in code:
        code = code.replace("plotly.plotly","chart_studio.plotly")
    if "sklearn.externals.six" in code:
        code = code.replace("sklearn.externals.six","six")
    if "from keras.utils import to_categorical" in code:
        code = code.replace("from keras.utils import to_categorical","from keras.utils.np_utils import to_categorical")
    if "from sklearn.preprocessing import Imputer" in code:
        code = code.replace("from sklearn.preprocessing import Imputer","from sklearn.impute import SimpleImputer as Imputer")
    if "from keras.optimizers import Adam" in code:
        code = code.replace("from keras.optimizers import Adam","from keras.optimizers import adam_v2 as Adam")
    if "from pandas.tools import plotting" in code:
        code = code.replace("from pandas.tools import plotting","from pandas import plotting")
    if "sklearn.externals.joblib" in code:
        code = code.replace("sklearn.externals.joblib","joblib")
    if ".as_matrix()" in code:
        code = code.replace(".as_matrix()",".values")
    if "jaccard_similarity_score" in code:
        code = code.replace("jaccard_similarity_score","jaccard_score")
    if "get_ipython()." in code:
        code = code.replace("get_ipython().","#get_ipython().")
    if "from_pandas_dataframe(" in code:
        code = code.replace("from_pandas_dataframe(","from_pandas_edgelist(")
    if "Perceptron(n_iter" in code:
        code = code.replace("Perceptron(n_iter","Perceptron(max_iter")
    if "pd.scatter_matrix(" in code:
        code = code.replace("pd.scatter_matrix(", "pd.plotting.scatter_matrix(")
    if "from keras.optimizers import SGD" in code:
        code = code.replace("from keras.optimizers import SGD", "from tensorflow.keras.optimizers import SGD")
    if "SMOTE" in code and "ratio" in code:
        code = code.replace("ratio", "sampling_strategy")
    return code

def load_code(path, test):
    with open(path, 'r') as f:
        dict_ = json.load(f)
        if test:
            code = dict_['code']
        else:
            code = dict_['validation_code']

    code = cleaning_origin(code)
        # #print('self.origin_code', self.origin_code)
    # try:
    #     #print("!!!!!!!!!!!!!!!!!!!!!!!")
    #     lines = code.split('\n')
    #     new_code = ''
    #     end_code= ''
    #     for index,line in enumerate(lines):
    #         if index <  len(lines)- 35:
    #             new_code += line
    #             new_code += '\n'
    #         else:
    #             end_code += line
    #             end_code += '\n'

    #     # temp = remove_model(new_code)
    #     # code = temp
    #     code += end_code
    # except:
    #     pass
    lines = code.split('\n')
    new_code = ""
    for index,line in enumerate(lines):
        line = exchange_code(line)  
        new_code += line
        new_code +="\n"
    return new_code


def cleaning_origin(code):
    lines = code.split('\n')
    res_str = ''
    for line in lines:
        line1 = line.strip()
        
        if '#' in line:
            num_p = 0
            num_pp = 0
            index = line.index("#")
            for char in line[0:index]:
                if char == "'":
                    num_p += 1
                if char == '"':
                    num_pp += 1
            if num_p %2 == 0 and num_pp %2 == 0:
                line = line[0:index]
        if len(line) != 0:
            line1 = line.strip()
            if line[-1] == '\\':
                res_str += line[0:-1]
            elif len(line1) > 0:
                if line1[-1] == ',':
                    res_str += line
                else:
                    res_str += line
                    res_str += '\n'
            else:
                res_str += line
                res_str += '\n'
    # #print(res_str)
    # #print("************************8*********************")
    res_str = clean_kuohao(res_str)
    # #print(res_str)
    # #print("************************8**55*******************")
    return res_str

def clean_kuohao(code):
    lines = code.split('\n')
    left_num = 0
    right_num = 0
    left_z_num = 0
    right_z_num = 0
    str_ = ''
    is_in_beizhu = False
    for line in lines:
        num_p = 0
        num_pp = 0
        if line.count('"""') % 2 == 1 or line.count("'''") %2 == 1:
            is_in_beizhu = not is_in_beizhu
        
        if not is_in_beizhu:
            for index,char in enumerate(line):
                if num_pp % 2 ==0 and num_p % 2 ==0:
                    if char == '#':
                        continue
                if char == '"':
                    if index > 0:
                        if line[index-1] != '\\':
                            if num_p % 2 ==0:
                                num_pp += 1
                    else:
                        if num_p % 2 ==0:
                                num_pp += 1
                if char == "'":
                    if index > 0:
                        if line[index-1] != '\\':
                            if num_pp % 2 ==0:
                                num_p += 1
                    else:
                        if num_pp % 2 ==0:
                                num_p += 1
                if char == '(':
                    if num_pp %2 ==0 and num_p %2 == 0:
                        left_num += 1
                if char == ')':
                    if num_pp %2 ==0 and num_p %2 == 0:
                        right_num += 1
                if char == '[':
                    if num_pp %2 ==0 and num_p %2 == 0:
                        left_z_num += 1
                if char == ']':
                    if num_pp %2 ==0 and num_p %2 == 0:
                        right_z_num += 1
                    
        if left_num == right_num and left_z_num == right_z_num:
            str_ += line
            str_ += '\n'
            left_num =0
            right_num = 0
        else:
            str_ += line
    return str_


class Logger(object):
    def __init__(self, filename='Default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

class Preprocessing:
    def __init__(self):
        self.origin_code = ''
        self.code = ''
        self.train_test_split_line = None
        self.info_triple = np.load('HybridPipeGen/core/merged_info_triple.npy', allow_pickle=True).item()
        self.suc = 0
        self.change_suc = 0
        self.train_test_split_no_label = 0
        self.model_index = 0

        # self.save_path = '/home/yxm/stat_result/'
        self.res_data_path = 'HybridPipeGen/core/hi_res_data/'
        self.val_res_data_path = 'HybridPipeGen/core/hi_val_res_data/'

    def load_code(self, ipypath):
        '''input a ipynb path,
        create a python file from this ipynb file'''
        # #print(ipypath)
        with open(ipypath, 'r', encoding='utf-8') as f:
            cells = json.load(f)['cells']
        wstr = ''
        for i in range(len(cells)):
            if cells[i]['cell_type'] == 'markdown':
                for j in cells[i]['source']:
                    wstr += ('# ' + j)
                wstr += '\n\n'
            elif cells[i]['cell_type'] == 'code':
                # wstr += "# In[" + str(cells[i]['execution_count']) + "]\n\n"
                # #print("XXXX")
                # #print(type(cells[i]['source']).__name__)
                if type(cells[i]['source']).__name__ == 'str':
                    code_list = cells[i]['source'].split("\n")
                    for line in code_list:
                        if len(line) != 0:
                            # #print(line[0])
                            if line[0] == '%':
                                line = '#' + line[0]
                        wstr += line
                        wstr += '\n'
                    wstr += '\n'
                else:
                    for line in cells[i]['source']:
                        if len(line) != 0:
                            if line[-1] == '\n':
                                line = line[0:-1]
                        if len(line) != 0:
                            # #print(line[0])
                            if line[0] == '%':
                                line = '#' + line[0]
                        wstr += line
                        wstr += '\n'
                    wstr += '\n'
                # for index,j in enumerate(cells[i]['source']):
                #     # #print(index,j)
                #     if j[0] == '%' and index == 0:
                #         # #print(j)
                #         j = '#' + j
                #     wstr += j

                # wstr += '\n\n'
        return wstr
    def load_origin_code(self, notebook_id, need_remove_model):
        self.train_test_split_index = None
        self.code = ''
        filepath = "data/notebook/" + str(notebook_id) +'.ipynb'
        self.file_path = filepath
        self.notebook_id = notebook_id
        if '.ipynb' in filepath:
            # output_path = filepath.replace(".ipynb",'.py')
            # #print(output_path)
            # output_path = output_path.replace("../data/notebook/",'HybridPipeGen/core/tmpdata/pycode/')
            # #print(output_path)
            self.origin_code = self.load_code(filepath)
            # ipynb2py(filepath, output_path)
        # if '.py' in filepath:
        else:
            output_path = filepath
            with open(output_path, 'r') as src_file:
                self.origin_code = src_file.read()

        # #print('self.origin_code', self.origin_code)
        self.origin_code = cleaning_origin(self.origin_code)
        # #print('self.origin_code', self.origin_code)
        if need_remove_model==1:
            try:
                #print("!!!!!!!!!!!!!!!!!!!!!!!")
                temp = remove_model(self.origin_code)
                self.origin_code = temp
            except:
                pass
        if need_remove_model==2:

            try:
                # #print("???????????????????????????????")
                temp = remove_model_2(self.origin_code)
                self.origin_code = temp
            except:
                pass
        
        # #print('self.origin_code', self.origin_code)
        # except:
            # pass
        self.code = self.origin_code
        # self.file_path = self.save_path + notebook_id
        # self.file_path = self.res_data_path + notebook_id
        # if not os.path.exists(self.file_path):
        #     os.mkdir(self.file_path)
        # self.train_feature_file = self.file_path + '/trainX'
        # self.test_feature_file = self.file_path + '/testX'
        # self.train_label_file = self.file_path + '/trainY'
        # self.test_label_file = self.file_path + '/testY'


    def find_train_test_split(self):
       
        self.origin_code = self.origin_code.replace("\\\n", "")
        code_lines = self.origin_code.split("\n")
        
        not_found = True
        for index,line in enumerate(code_lines):
            # #print(line)
            if "train_test_split(" in line and '=' in line:
                self.suojin = 0
                train_test_index = index
                not_found = False
                for char in line:
                    if ord(char) != 32:
                        break
                    self.suojin += 1
        if not_found:
            return 0
    
        # #print('train_test_index', train_test_index)
        for index,line in enumerate(code_lines):
            if index == train_test_index:
                try:
                    line1 = line.strip()
                    r_node = ast.parse(line1)
                    func_node = r_node.body[0].value
                    target_node_tuple_node = r_node.body[0].targets[0]
                    target_node_list = [item for item in target_node_tuple_node.elts]
                except Exception as e:       
                    # #print(e)   
                    if str(e) == 'unexpected EOF while parsing (<unknown>, line 1)':
                        suc_run = False
                        add_num = 1
                        new_line = line.strip()
                        need_strip = False
                        while suc_run == False:
                            # #print('new_line', new_line)
                            new_line = new_line + (code_lines[index+add_num].strip())
                            try:
                                r_node = ast.parse(new_line)
                                target_node_tuple_node = r_node.body[0].targets[0]
                                target_node_list = [item for item in target_node_tuple_node.elts]
                                func_node = r_node.body[0].value
                                suc_run=True
                            except:
                                add_num += 1
                    else:
                        # #print(self.notebook_id)
                        #print('\033[1;31;40m cant find tts ' + line+ '\033[0m')
                        #print(str(e))
                        continue
                if len(func_node.args) == 2:
                    self.X_varible = astunparse.unparse(func_node.args[0])[:-1]
                    self.y_varible = astunparse.unparse(func_node.args[1])[:-1]
                    # if len(target_node_list) != 4:
                        # print(len(target_node_list))
                        # #print(type(target_node_list[0]).__name__)
                        # #print(notebook_id)
                        #print('\033[1;30;40m' + line+ '\033[0m')
                    if len(target_node_list) == 4:
                        self.x_train_varible = astunparse.unparse(target_node_list[0])[:-1]
                        self.x_test_varible = astunparse.unparse(target_node_list[1])[:-1]
                        self.y_train_varible = astunparse.unparse(target_node_list[2])[:-1]
                        self.y_test_varible = astunparse.unparse(target_node_list[3])[:-1]
                        # #print('\033[1;32;40m' + line+ '\033[0m')
                        # #print(self.X_varible, self.y_varible)
                        # #print(self.x_train_varible, self.x_test_varible, self.y_train_varible, self.y_test_varible)
                        
                        #print('self.line' , line)
                        self.train_test_split_index = index
                        #print('self.train_test_split_index' , self.train_test_split_index)
                        return 1
                else:
                    #print(self.notebook_id)
                    self.train_test_split_no_label+=1
                    #print('\033[1;33;40m' + line+ '\033[0m')
                    self.data_varible = astunparse.unparse(func_node.args[0])[:-1]
                    # if len(target_node_list) != 2:
                        # print(len(target_node_list))
                        # #print(type(target_node_list[0]).__name__)
                        # #print(notebook_id)
                        #print('\033[1;30;40m' + line+ '\033[0m')
                    if len(target_node_list) == 2:
                        self.train_varible = astunparse.unparse(target_node_list[0])[:-1]
                        self.test_varible = astunparse.unparse(target_node_list[1])[:-1]
              
                        # #print('\033[1;32;40m' + line+ '\033[0m')
                        # #print(self.X_varible, self.y_varible)
                        # #print(self.x_train_varible, self.x_test_varible, self.y_train_varible, self.y_test_varible)
                        self.train_test_split_index = index
                        return 2
        #print('train test 0', self.notebook_id)     
        # #print(self.origin_code)    
        return 0
    def change_train_test_split(self, result_id, ratio=0.8, split_random_state=0):
        if self.train_test_split_index == None:
                return False
        #print('result_id', result_id)
        suojinline = ''
        for i in range(0,self.suojin):
            suojinline += ' '
        if result_id == 2:
        
            # with open('datasetlabel/notebook_dataset_no_empty.json', 'r') as f:
                # dataset_label = json.load(f)
            with open('../statsklearn/new_notebook.json', 'r') as f:
                dataset_label = json.load(f)
        
            if str(self.notebook_id) not in dataset_label:
                return False
            for name in dataset_label[str(self.notebook_id)]['column_index']:
                # #print(name)
                # #print(dataset_label[str(self.notebook_id)]['column_index'][name])
                # #print(dataset_label[str(self.notebook_id)]['index'][0])
                if dataset_label[str(self.notebook_id)]['index'][0] == dataset_label[str(self.notebook_id)]['column_index'][name]:
                    label_name = name

            self.x_train_varible = 'x_train_varible'
            self.y_train_varible = 'y_train_varible'
            self.x_test_varible = 'x_test_varible'
            self.y_test_varible = 'y_test_varible'

            self.X_varible = 'x_varible'
            self.y_varible = 'y_varible'
            
            split_x_y_code = self.y_varible + ' = ' + self.data_varible + '["' + label_name + '"]\n'
            split_x_y_code += self.X_varible + ' = ' + self.data_varible + '.drop(["' + label_name + '"], 1)\n'
            # #print(split_x_y_code)
            train_test_split_line = split_x_y_code + 'from sklearn.model_selection import train_test_split\n'

        elif result_id == 1:
            train_test_split_line = suojinline + 'from sklearn.model_selection import train_test_split\n'
        train_test_split_line += suojinline
        train_test_split_line += self.x_train_varible + ', ' + self.x_test_varible + ', ' +  self.y_train_varible + ', ' +  self.y_test_varible
        train_test_split_line = train_test_split_line + " = train_test_split(" + self.X_varible + ', ' + self.y_varible + ', train_size='+str(ratio)+', test_size=1-'+str(ratio)+', random_state='+str(split_random_state)+')' + '\n'
        code_lines = self.origin_code.split("\n")
        # #print(code_lines[self.train_test_split_index])
        #print('self.train_test_split_index',self.train_test_split_index)
        
        self.code = ''
        # #print('')
        for index,line in enumerate(code_lines):
            # #print(index)
            if index == self.train_test_split_index and result_id==1:
                self.code += train_test_split_line
                # break
                continue
            elif index == self.train_test_split_index and result_id==2:
                self.code += train_test_split_line
                self.code += line
                self.code += '\n'
                continue
            if line == '#print(os.listdir("../input"))':
                self.code = self.code + '#' + line
            else: 
                self.code += line
            self.code += '\n'
        # self.code += train_test_split_line
        code_list = self.code.split('\n')
        #print('self.code',code_list[self.train_test_split_index])
        self.change_suc += 1
        # if len(self.code) == 0:
            # print(len(self.code))
        # #print(self.notebook_id, len(self.code))

        # self.end_index = len(self.code.split("\n"))
        if result_id == 2:
            self.end_index = self.train_test_split_index +2
        else:
            self.end_index = self.train_test_split_index - 1
        
        #print(self.code.split('\n')[self.end_index])
        #print('self.end_index',self.end_index)
        with open("HybridPipeGen/core/tmpdata/prenotebook_varibles_index/"+str(self.notebook_id)+".json", 'w') as f:
            json.dump( {'x_varible':self.X_varible, 'end_idnex': self.end_index}, f)

        # for index,line in enumerate(code_list):
            # #print(index, '\033[1;33;40m'+line+'\033[1;31;40m')

    def save_code(self, root_path):
        with open(root_path+ str(self.notebook_id) + '.py', 'w') as f:
            f.write(self.code)
    
         
    def add_model_code(self, metric_type='accuracy_score'):
        save_file_path = 'prenotebook_res/'+ str(self.notebook_id) + '.npy'
        model_type = self.info_triple[self.notebook_id]['model_type']
        if model_type not in model_import_code:
            #print("\033[1;35;40mnot in "+model_type+"\033[1;31;40m")
            return False
        if self.code == '':
            return False
        

        self.code = self.code+'import pandas as pd\n'+ accuracy_import_code[metric_type] + '\n'
        self.code = self.code + model_import_code[model_type] + '\n'
        self.code += '#print("start running model training........")\n'
        if model_type in ['LinearRegression','KNeighborsClassifier']:
            self.code += 'model = ' + model_type + '()\n'
        elif model_type == "LogisticRegression":
            self.code += "model = LogisticRegression(solver='liblinear', random_state=0)\n"
        else:
            self.code += 'model = ' + model_type + '(random_state=0)\n'
        self.code += 'model.fit(' + self.x_train_varible + ', ' + self.y_train_varible + ')\n'
        self.code += 'y_pred = model.predict(' + self.x_test_varible +')\n'
        self.code += 'score = ' +  metric_type +'(' + self.y_test_varible +', y_pred)\n'
        self.code += 'import numpy as np\n'
        self.code += 'np.save("HybridPipeGen/core/tmpdata/' + save_file_path +'", { "' + metric_type+'": score })\n'
        # self.code += 'import pandas as pd\n'
        # self.code += 'if type(' + self.x_train_varible + ').__name__ == "ndarray":\n'
        # self.code += '    np.save("' + self.train_feature_file + '.npy' + '", ' + self.x_train_varible + ')\n'
        # self.code += 'if type(' + self.x_train_varible + ').__name__ == "Series":\n'
        # self.code += '    ' + self.x_train_varible + '.to_csv("' + self.train_feature_file + '.csv' + '",encoding="gbk")\n'
        # self.code += 'if type(' + self.x_train_varible + ').__name__ == "DataFrame":\n'
        # self.code += '    ' + self.x_train_varible + '.to_csv("' + self.train_feature_file + '.csv' + '",encoding="gbk")\n\n'
        
        # self.code += 'if type(' + self.x_test_varible + ').__name__ == "ndarray":\n'
        # self.code += '    np.save("' + self.test_feature_file + '.npy' + '", ' + self.x_test_varible + ')\n'
        # self.code += 'if type(' + self.x_test_varible + ').__name__ == "Series":\n'
        # self.code += '    ' + self.x_test_varible + '.to_csv("' + self.test_feature_file + '.csv' + '",encoding="gbk")\n'
        # self.code += 'if type(' + self.x_test_varible + ').__name__ == "DataFrame":\n'
        # self.code += '    ' + self.x_test_varible + '.to_csv("' + self.test_feature_file + '.csv' + '",encoding="gbk")\n\n'

        # self.code += 'if type(' + self.y_train_varible + ').__name__ == "ndarray":\n'
        # self.code += '    np.save("' + self.train_label_file + '.npy' + '", ' + self.y_train_varible + ')\n'
        # self.code += 'if type(' + self.y_train_varible + ').__name__ == "Series":\n'
        # self.code += '    ' + self.y_train_varible + '.to_csv("' + self.train_label_file + '.csv' + '",encoding="gbk")\n'
        # self.code += 'if type(' + self.y_train_varible + ').__name__ == "DataFrame":\n'
        # self.code += '    ' + self.y_train_varible + '.to_csv("' + self.train_label_file + '.csv' + '",encoding="gbk")\n\n'

        # self.code += 'if type(' + self.y_test_varible + ').__name__ == "ndarray":\n'
        # self.code += '    np.save("' + self.test_label_file + '.npy' + '", ' + self.y_test_varible + ')\n'
        # self.code += 'if type(' + self.y_test_varible + ').__name__ == "Series":\n'
        # self.code += '    ' + self.y_test_varible + '.to_csv("' + self.test_label_file + '.csv' + '",encoding="gbk")\n'
        # self.code += 'if type(' + self.y_test_varible + ').__name__ == "DataFrame":\n'
        # self.code += '    ' + self.y_test_varible + '.to_csv("' + self.test_label_file + '.csv' + '",encoding="gbk")\n\n'
        self.suc += 1
        return True
        

    def found_dataset(self, old_path, notebook_id, root_path, origin_code):
        """
        :param old_path:
        :param notebook_id:
        :param root_path:
        :param origin_code:
        :return:
        如果运行时发现路径不对，找到需要替换的路径
        """
        old_root_path = ''
        if '/' not in old_path:
            result = root_path + '/' + old_path
            old_root_path = old_path
        else:
            for index, i in enumerate(old_path.split('/')):
                if index != len(old_path.split('/')) - 1:
                    old_root_path = old_root_path + i + '/'
                else:
                    if '.' not in i:
                        old_root_path = old_root_path + i
                    if '/' == old_root_path[-1]:
                        old_root_path = old_root_path[0:-1]

            result = root_path
        return origin_code.replace(old_root_path, result)

    def run_one_code(self, notebook_id, origin_code, new_path, try_time, found=False):
        """
        :param origin_code: 需要运行的代码字符串
        :param new_path: 替换路径
        :param try_time: 第几次运行了
        :return: 返回修改过后或者成功运行的代码
        运行代码
        """
        # #print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        try:
            if '/kaggle/input' in origin_code:
                origin_code = origin_code.replace('/kaggle/input', new_path)
            # #print(origin/_code)
            cm = compile(origin_code, '<string>', 'exec')
        except Exception as e:
            #print("compile fail", e)
            return "compile fail"
        #print("\033[0;33;40m try times:" + str(try_time) +"\033[0m")
        can_run = False
        # #print('****************************************************************')
        try:
            ns = {}
            exec(cm,ns)
            print("\033[0;32;40msucceed\033[0m")
            can_run = True
        except Exception as e:
            # traceback.print_exc()
            error_str = str(e)
            #print("\033[0;31;40merror_str\033[0m", error_str)

            new_code = origin_code
            foun = 0
            if "[Errno 2] No such file or directory: " in error_str:
                error_path = error_str.replace("[Errno 2] No such file or directory: " , "")
                error_path = error_path[1:-1]
                new_code = self.found_dataset(error_path, 1, new_path, origin_code)
                # #print('error_path:', error_path)
                foun=1
                #print('path error')
            elif "\"['Unnamed: 0'] not found in axis\"" in error_str:
                new_code = origin_code.replace("'Unnamed: 0'", "'index'")
            elif "does not exist:" in error_str and '[Errno 2] File ' in error_str:
                error_path = error_str.split(':')[-1].strip()
                error_path = error_path[1:-1]
                new_code = self.found_dataset(error_path, 1, new_path, origin_code)
                #print('path error')
                foun=1
            elif "No module named " in error_str and '_tkinter' not in error_str:
                package = error_str.replace("No module named ", "")
                package = package[1:-1]
                command = ' pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ' + package.split('.')[0]
                if 'sklearn' in command or 'scikit_learn' in command:
                    command = 'pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scikit_learn==0.23.2'
                os.system(command)
                #print('package lack error')

            elif  ": No such file or directory" in error_str:
                index1 = error_str.find("'")
                index2 = error_str.find("'", index1+1)
                error_path = error_str[index1+1:index2]
                new_code = self.found_dataset(error_path, 1, new_path, origin_code)
                #print('path error')
            elif "Command '['ls'," in error_str:
                index1 = error_str.find('ls')
                el_line = error_str[index1+6:]
                right_index  = el_line.find('\'')
                error_path = el_line[0:right_index]
    
                new_code = self.found_dataset(error_path, 1, new_path, origin_code)
      
                foun = 1
                #print('command path error')
            elif "File b" in error_str:
                index1 = error_str.find("'")
                index2 = error_str.find("'", index1 + 1)
                error_path = error_str[index1 + 1:index2]
                new_code = self.found_dataset(error_path, 1, new_path, origin_code)
                #print('File b path error')
                foun = 1

            elif "'DataFrame' object has no attribute 'ix'" in error_str or "'Series' object has no attribute 'ix'" in error_str:
                new_code = origin_code.replace('.ix', '.iloc')
                #print('package version error')
            elif "'DataFrame' object has no attribute 'sort'" in error_str:
                new_code = origin_code.replace('.sort(', '.sort_values(')
                #print('package version error')
            else:
                #print("\033[0;31;40merror_str\033[0m", error_str)
                #print('unknown error')
                # error_dict[notebook_id] = str(error_str)
                return "False"
            if try_time < 7:
                if foun ==1:
                    found = True
                res = self.run_one_code(notebook_id, new_code, new_path, try_time + 1,found)
                if res == 'compile fail': 
                    # error_dict[notebook_id] = str(res)
                    return res
                elif  res== 'False':
                    # error_dict[notebook_id] = str(error_str)
                    return res
            else:
                #print('error 9')
                # error_dict[notebook_id] = str(error_str)
                return "False"
        return origin_code

    def deal_with_split_no_var(self):
        code_lines = self.code.split('\n')
        for index,line in enumerate(code_lines):
            if 'train_test_split(' in line:
                train_test_split_index = index
        try:
            r_node = ast.parse(code_lines[train_test_split_index])
            arg0_node = r_node.body[0].value.args[0]
            arg1_node = r_node.body[0].value.args[1]
            if type(arg0_node).__name__ != 'Name':
                add_line1 = 'X = '+ astunparse.unparse(arg0_node)[0:-1]
                
                code_lines[train_test_split_index] = code_lines[train_test_split_index].replace(astunparse.unparse(arg0_node)[0:-1], 'X')
                
                #print('code_lines[train_test_split_index]', code_lines[train_test_split_index])
                code_lines = code_lines[0:train_test_split_index] + [add_line1] + code_lines[train_test_split_index:]
                train_test_split_index += 1
            if type(arg1_node).__name__ != 'Name':
                add_line2 = 'y = '+ astunparse.unparse(arg1_node)
                code_lines[train_test_split_index] = code_lines[train_test_split_index].replace(astunparse.unparse(arg1_node)[0:-1], 'y')
                code_lines = code_lines[0:train_test_split_index] + [add_line2] + code_lines[train_test_split_index:]
                train_test_split_index += 1
            str_ = ''
            for line in code_lines:
                str_ += line
                str_ += '\n'
            self.origin_code = str_
            self.code = str_
            #print(self.origin_code)
        except Exception as e:
            #print('error', e)
            return 

    def profiling_code(self, notebook_id, need_remove_model):
        self.load_origin_code(notebook_id, need_remove_model)
        # #print(self.origin_code)
        self.deal_with_split_no_var()
        result_id = self.find_train_test_split()
        if result_id == 0: # can not find train test split
            self.cant_find_train_test += 1
            if need_remove_model == 1:
                self.load_origin_code(notebook_id, need_remove_model=2)
                result_id = self.find_train_test_split()
                if result_id == 0:
                    return -1
            else:
                return -1
            #break
        # #print('self.origin_code', self.code)
        
        self.change_train_test_split(result_id)
        res = self.add_model_code()
        #print(self.code)
        self.save_code('HybridPipeGen/core/tmpdata/prenotebook_code/')
        return res

    def run_origin_test(self, notebook_id, need_try_again):
        start = time.time()
        self.code = exchange_code(self.code)
        # #print(self.code)
        self.code = self.run_one_code(notebook_id, self.code, dataset_root_path + self.info_triple[notebook_id]['dataset_name'], 0)
        if self.code != '' and self.code != "False" and self.code != "compile fail":
            # #print("saving")
            self.save_code('HybridPipeGen/core/tmpdata/runned_notebook/')
            end = time.time()
            running_time[notebook_id] = end-start
            with open('HybridPipeGen/core/tmpdata/human_running_time_1.json','w') as f:
                json.dump(running_time,f)
        else:
            if need_try_again == 2:
                res = self.profiling_code(notebook_id, need_remove_model=2)
                self.run_origin_test(notebook_id, need_try_again-1)
            elif need_try_again == 1:
                res = self.profiling_code(notebook_id, need_remove_model=0)
                self.run_origin_test(notebook_id, need_try_again-1)
            else:
                self.run_faile += 1
                with open("error_dict.json", 'w') as f:
                    json.dump(error_dict, f)

    def batch_processing(self):
        info_triple = np.load("../statsklearn/NewDataInfoTriple.npy", allow_pickle=True).item()
        # filenames = os.listdir("notebook597")
        # with open('need_rerun.json','r') as f:
            # notebooks_ids = json.load(f)
        # notebooks_ids = list(info_triple.keys())
        # with open("name2id.json", 'r') as f:
        #     name2id = json.load(f)
        # notebooks = [int(filename.split('.')[0]) for filename in filenames]

        self.cant_find_train_test = 0
        linear_reg = 0
        add_faile = 0
        self.run_faile = 0
        all_ = 0
        if os.path.exists('HybridPipeGen/core/tmpdata/human_running_time_1.json'):
            with open('HybridPipeGen/core/tmpdata/human_running_time_1.json','r') as f:
                running_time = json.load(f)
        else:
            running_time = {}
        need_continue = True
        
        # split_len = int(len(notebooks_ids)/9)
        # notebooks_segments = [notebooks_ids[i*split_len: i*split_len + split_len] for i in range(0,8)]
        # notebooks_segments.append(notebooks_ids[8*split_len: ])
        # global specific_split
        # for index,seg in enumerate(notebooks_segments):
        #     if index != specific_split:
        #         continue
        #     for ind,notebook_id in enumerate(seg):
        #         #print(ind)
        #         # if ind <=37:
        #         #     continue 
        #         # if notebook_id == 'bandlote_telco-churn-business-analysis':
        #         #     continue
        #         # notebooks_id_num = name2id[notebook_id]
        #         # if notebooks_id_num == '423':
        #             # need_continue = False
        #             # continue
        #         # if need_continue:
        #             # continue
        #         # if notebook_id in runed:
        #             # continue
        #         all_ += 1
        #         # runed.append(notebook_id)
        #         # np.save('runed.npy', runed)
        #         #print("#########")
        #         #print(notebook_id)
        #         exist_f = open("/home/yxm/staticfg-master/origin_215.txt", "r")
        #         exist = exist_f.readlines()
        #         exitst_ = [x.strip("\n") for x in exist]
        #         if notebook_id not in exitst_:
        #             continue
                
        #         if os.path.exists(self.save_path + notebook_id):
        #             if len(os.listdir(self.save_path + notebook_id)) != 0:
        #                 continue
        #         if not os.path.exists("../data/notebook/" + str(notebook_id) +'.ipynb'):
        #             continue
        #         res = self.profiling_code(notebook_id, need_remove_model=1)
        #         #print('save_code', res)
        #         if self.info_triple[notebook_id]['model_type'] == 'LinearRegression':
        #             linear_reg += 1
        #             if os.path.exists('HybridPipeGen/core/tmpdata/runned_notebook/'+ str(self.notebook_id) + '.py'):
        #                 os.system('rm HybridPipeGen/core/tmpdata/runned_notebook/'+ str(self.notebook_id) + '.py')
        #             if os.path.exists('HybridPipeGen/core/tmpdata/prenotebook_res/'+ str(self.notebook_id) + '.npy'):
        #                 os.system('rm HybridPipeGen/core/tmpdata/prenotebook_res/'+ str(self.notebook_id) + '.npy')
        #         else:
        #             if res == True:
        #                 self.run_origin_test(notebook_id, need_try_again=2)
        #             else:
        #                 add_faile += 1
        #         #print(ind)
        #         #print(notebook_id)
        #         # break
                
        #     #print('cant_find_train_test', self.cant_find_train_test)
        #     #print('linear_reg', linear_reg)
        #     #print('add_faile', add_faile)
        #     #print('run_faile', self.run_faile)
        #     #print('all_', all_)
        # exist_f = open("/home/yxm/staticfg-master/origin.txt", "r")
        # exist = exist_f.readlines()
        # exist_ = [x.strip("\n") for x in exist]
        with open('fix_model.json','r') as f:
            notebooks = json.load(f)
        exist_f = open("/home/yxm/staticfg-master/all_s.txt", "r")
        exist = exist_f.readlines()
        exitst_ = [x.strip("\n") for x in exist]
        # notebooks = list(set(notebooks) & set(exitst_))
        # exist = exist_f.readlines()
        # exist_ = [x.strip("\n") for x in exist]
        #print(len(notebooks))
        for ind in range(len(exitst_)):
            notebook_id = exitst_[ind]
            # notebook_id = ind
            # #print(ind)
            # if notebook_id !='bilal75210_project-data-science':
                # continue
            # #print(ind)
            if notebook_id!="akansha23_heart-diseases-visualisation-and-prediction":
                continue
            # if ind <=408:
            #     continue
            all_ += 1
            #print("#########")
            #print(notebook_id)
            #print("/HybridPipeGen/core/tmpdata/prenotebook_res/" + notebook_id+".npy")
            # if notebook_id != "bparesh_extensive-eda-models-logistic-random-forest":
            #     continue
            # if os.path.exists("/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/prenotebook_res/" + notebook_id+".npy"):
            #     #print("............")
            #     continue
            if not os.path.exists("../data/notebook/" + str(notebook_id) +'.ipynb'):
                continue
            res = self.profiling_code(notebook_id, need_remove_model=1)
            #print('save_code', res)
            if self.info_triple[notebook_id]['model_type'] == 'LinearRegression':
                linear_reg += 1
                if os.path.exists('HybridPipeGen/core/tmpdata/runned_notebook/'+ str(self.notebook_id) + '.py'):
                    os.system('rm HybridPipeGen/core/tmpdata/runned_notebook/'+ str(self.notebook_id) + '.py')
                if os.path.exists('HybridPipeGen/core/tmpdata/prenotebook_res/'+ str(self.notebook_id) + '.npy'):
                    os.system('rm HybridPipeGen/core/tmpdata/prenotebook_res/'+ str(self.notebook_id) + '.npy')
            else:
                if res == True:
                    self.run_origin_test(notebook_id, need_try_again=2)
                else:
                    add_faile += 1
            #print(ind)
            #print(notebook_id)
            # break
        #print('cant_find_train_test', self.cant_find_train_test)
        #print('linear_reg', linear_reg)
        #print('add_faile', add_faile)
        #print('run_faile', self.run_faile)
        #print('all_', all_)
            

        
def run_origin_path(path):
    with open(path, 'r') as f:
        code = f.read()
    pro = Preprocessing()
    notebook_id = int(path.split("/")[-1].split(".")[0])
    
    #print(notebook_id)
    code = cleaning_origin(code)
    code = remove_model(code)
    if 'from keras.utils import to_categorical' in code:
        # code = code.replace('from keras.utils import to_categorical\n', 'from keras import utils as np_utils\nnp_utils.to_categorical()\n')
        code = code.replace('from keras.utils import to_categorical', '')
    if 'from sklearn.cross_validation import train_test_split' in code:
        # code = code.replace('from keras.utils import to_categorical\n', 'from keras import utils as np_utils\nnp_utils.to_categorical()\n')
        code = code.replace('from sklearn.cross_validation import train_test_split', 'from sklearn.model_selection import train_test_split')
    if 'from pandas.tools.plotting import scatter_matrix' in code:
        # code = code.replace('from keras.utils import to_categorical\n', 'from keras import utils as np_utils\nnp_utils.to_categorical()\n')
        code = code.replace('from pandas.tools.plotting import scatter_matrix', 'from pandas.plotting import scatter_matrix')
    
    if 'from sklearn.cross_validation import cross_val_score' in code:
        # code = code.replace('from keras.utils import to_categorical\n', 'from keras import utils as np_utils\nnp_utils.to_categorical()\n')
        code = code.replace('from sklearn.cross_validation import cross_val_score', 'from sklearn.model_selection import cross_val_score')
    if 'import sklearn.cross_validation' in code:
        code = code.replace('import sklearn.cross_validation', "from sklearn.model_selection import cross_validation")

    # #print(code)
    code = pro.run_one_code(notebook_id, code, dataset_root_path + pro.info_triple[notebook_id]['dataset_name'], 0)
    
@func_set_timeout(180)
def run_path(path, replace_code, test, running_id=None, force_rerun=False):
    # #print('path', path)
    with open(path, 'r') as f:
        item_json = json.load(f)
    # #print(item_json.keys())
    # if 'seq' not in item_json.keys():
    #     return 
    # seq = item_json['seq']
    # end_ = False
    # for seq_item in seq:
    #     if seq_item['edge_id'] == 'end':
    #         end_=True
    #         break
    # if end_ == False:
    #     return 
    code = load_code(path, test)
    # #print(code)
    notebook_id = path.rsplit("/")[-2]
    seq_id = path.rsplit("/")[-1].split(".")[0]
    # #print("--------------------------------------------------")
    # if notebook_id == 4102169 and seq_id == 1 or notebook_id == 5753518:
        # return
    # if test:
    #     if running_id !=None:
    #         if not os.path.exists("HybridPipeGen/core/tmpdata/merged_result_1_"+str(running_id)+"/"+str(notebook_id)):
    #             os.mkdir("HybridPipeGen/core/tmpdata/merged_result_1_"+str(running_id)+"/"+str(notebook_id))
    #         if os.path.exists('HybridPipeGen/core/tmpdata/merged_result_1_'+str(running_id)+"/"+str(notebook_id)+"/"+str(seq_id)+".npy") and force_rerun==False:
    #             return
    #         global running_time
    #         if str(notebook_id) in running_time:
    #             if seq_id in running_time[str(notebook_id)] and force_rerun==False:
    #                 return
    #     else:
    #         if not os.path.exists("HybridPipeGen/core/tmpdata/merged_result_1/"+str(notebook_id)):
    #             os.mkdir("HybridPipeGen/core/tmpdata/merged_result_1/"+str(notebook_id))
    #         if os.path.exists('HybridPipeGen/core/tmpdata/merged_result_1/'+str(notebook_id)+"/"+str(seq_id)+".npy") and force_rerun==False:
    #             return
    #         if str(notebook_id) in running_time:
    #             if seq_id in running_time[str(notebook_id)] and force_rerun==False:
    #                 return
    # else:
    #     # if not os.path.exists("HybridPipeGen/core/tmpdata/merge_result_data/"+str(notebook_id)):
    #     #     os.mkdir("HybridPipeGen/core/tmpdata/merge_result_data/"+str(notebook_id))
    #     # if os.path.exists('HybridPipeGen/core/tmpdata/merge_result_data/'+str(notebook_id)+"/"+str(seq_id)+".npy") and force_rerun==False:
    #     #     return

    #     global validation_running_time
        
    #     if os.path.exists("HybridPipeGen/core/tmpdata/validation_running_time_cross.json"):
    #         with open("HybridPipeGen/core/tmpdata/validation_running_time_cross.json", 'r') as f:
    #             validation_running_time = json.load(f)
    #     else:
    #         validation_running_time = {}
    #     if str(notebook_id) in validation_running_time:
    #         # #print(seq_id)
    #         # #print(validation_running_time[str(notebook_id)].keys())
    #         if str(seq_id) in validation_running_time[str(notebook_id)] and force_rerun==False:
                
    #             # #print('continue')
    #             return
    start_time = time.time()
    
    #print(notebook_id)
    pro = Preprocessing()
    if len(replace_code) > 0:
        #print("??????")
        code = code.replace("cross_val_res/"+notebook_id +'.npy', replace_code+notebook_id+'/' + seq_id+'.npy')
        #print('HybridPipeGen/core/tmpdata/'+replace_code + notebook_id)
        #print(not os.path.exists('HybridPipeGen/core/tmpdata/'+replace_code + notebook_id))
        if not os.path.exists('HybridPipeGen/core/tmpdata/'+replace_code + notebook_id):
            #print('mkdir ', 'HybridPipeGen/core/tmpdata/'+replace_code + notebook_id)
            os.mkdir('HybridPipeGen/core/tmpdata/'+replace_code + notebook_id)
    new_c = ''
    lines = code.split('\n')
    for index,line in enumerate(lines):
        if "check_output(" in line:
            temp = line
            lines[index] = "#"+temp
        if 'LogisticRegression()' in line :
            lines[index] = line.replace('LogisticRegression()','LogisticRegression(solver=\'liblinear\')')
        if 'LogisticRegression(random_state=0)' in line:
            lines[index] = line.replace('LogisticRegression()','LogisticRegression(solver=\'liblinear\',random_state=0)')
        if 'coef_' in line:
            temp = line
            lines[index] = "#"+temp
        # if 'display(' in line:
        #     temp = line
        #     lines[index] = "#"+temp
        # if 'pred' in line and 'read_csv' not in line:
        #     temp = line
        #     lines[index] = "#"+temp
        # #print(line)
    # #print("----------------3---------------------------------------------------------")
    # #print(lines)
    for line in lines:
        # #print(line)
        if 'np.save' in line and ('_data_x.npy"' in line or '_data_y.npy' in line):
            new_c += '#'+line
            new_c += '\n'
        else:
            new_c += line
            new_c += '\n'
    # #print(new_c)
    # #print("--------------------4-------------------------")
    # #print(notebook_id)
    # info_triple = np.load('HybridPipeGen/core/merged_info_triple.npy', allow_pickle=True).item()
    # #print(info_triple[notebook_id]['dataset_name'])
    code = pro.run_one_code(notebook_id, new_c, dataset_root_path + pro.info_triple[notebook_id]['dataset_name'], 0)
    end = time.time()
    if test:
        # global running_time
        if running_id != None:
            running_time[str(notebook_id)][seq_id] = end-start_time
            with open("HybridPipeGen/core/tmpdata/running_time_"+str(running_id)+".json", 'w') as f:
                json.dump(running_time, f)
        else:
            running_time[str(notebook_id)][seq_id] = end-start_time
            with open("HybridPipeGen/core/tmpdata/running_time.json", 'w') as f:
                json.dump(running_time, f)
    # else:
    #     # global validation_running_time
    #     # global validation_running_time
    #     if running_id != None:
    #         if os.path.exists("HybridPipeGen/core/tmpdata/validation_running_time_"+str(running_id)+".json"):
    #             with open("HybridPipeGen/core/tmpdata/validation_running_time_"+str(running_id)+".json", 'r') as f:
    #                 #print("HybridPipeGen/core/tmpdata/validation_running_time_"+str(running_id)+".json")
    #                 validation_running_time = json.load(f)
    #         if str(notebook_id) not in validation_running_time:
    #             validation_running_time[str(notebook_id)] = {}
    #         validation_running_time[str(notebook_id)][seq_id] = end-start_time
    #         with open("HybridPipeGen/core/tmpdata/validation_running_time_"+str(running_id)+".json", 'w') as f:
    #             json.dump(validation_running_time, f)
    #     else:
    #         if os.path.exists("HybridPipeGen/core/tmpdata/validation_running_time_cross.json"):
    #             with open("HybridPipeGen/core/tmpdata/validation_running_time_cross.json", 'r') as f:
    #                 validation_running_time = json.load(f)
    #         if str(notebook_id) not in validation_running_time:
    #             validation_running_time[str(notebook_id)] = {}
    #         validation_running_time[str(notebook_id)][seq_id] = end-start_time
    #         # #print("xxxxxxxxxx")
    #         # #print(validation_running_time)
    #         with open("HybridPipeGen/core/tmpdata/validation_running_time_cross.json", 'w') as f:
    #             json.dump(validation_running_time, f)

def run_notebook_only():
    filelist = os.listdir('prenotebook_code')
    for filename in filelist:
        os.system('python prenotebook_code/'+filename)

running_time = {}
validation_running_time = {}
def run_test(batch_id, running_id=None , spe_notebook_id=None, spe_seq_id=None, force_rerun=False):
    filenamelist = os.listdir('merge_code')
    if batch_id != -1:
        filenamelist.sort()
    filenamelist1 = filenamelist[0:int(len(filenamelist)/4)]
    filenamelist2 = filenamelist[int(len(filenamelist)/4):int(len(filenamelist)/2)]
    filenamelist3 = filenamelist[int(len(filenamelist)/2):int(3*len(filenamelist)/4)]
    filenamelist0 = filenamelist[int(3*len(filenamelist)/4):]
    # chosed_list = random.choices(filenamelist, k=5)
    # ticks = time.time()/
    f_list = [filenamelist0,filenamelist1,filenamelist2,filenamelist3]
    global running_time

    if os.path.exists("running_time_"+str(batch_id)+".json"):
        with open("running_time_"+str(batch_id)+".json", 'r') as f:
            running_time = json.load(f)

    if batch_id == -1:
        one_filenamelist = filenamelist
    else:
        one_filenamelist = f_list[batch_id]
    for notebook_id in one_filenamelist:
        filelist = os.listdir('merge_code/'+notebook_id)
        filelist.sort()
        
        if notebook_id not in running_time:
            running_time[notebook_id] = {}
        for item in filelist:
            if spe_seq_id is not None:
                if item.split('.')[0] != str(spe_seq_id):
                    continue
            if spe_notebook_id is not None:
                if notebook_id != str(spe_notebook_id):
                    continue
            if notebook_id == str(4102169):
                continue
            #print("item,", item)
            # with eventlet.Timeout(20, False):
            if running_id !=None:

                run_path('merge_code_'+str(running_id)+'/'+notebook_id+'/'+item, batch_id, test=True, running_id=running_id, force_rerun=force_rerun)
            else:
                run_path('merge_code/'+notebook_id+'/'+item, batch_id, test=True, running_id=running_id, force_rerun=force_rerun)
            break
        break



def run_origin_validation():
    filenamelist = os.listdir('../staticfg-master/HybridPipeGen/core/tmpdata/validation_prenotebook_code')
    # with open("../staticfg-master/validation_merged_best_index.json",'r') as f:
        # validation_merged_best_index = json.load(f)
    split_len = int(len(filenamelist)/9)
    notebooks_segments = [filenamelist[i*split_len: i*split_len + split_len] for i in range(0,8)]
    notebooks_segments.append(filenamelist[8*split_len: ])
    global specific_split
    for index,seg in enumerate(notebooks_segments):
        if index != specific_split:
            continue
        for notebook_id_py in seg:
            
            notebook_id = notebook_id_py.split('.')[0]
            #print(notebook_id)
            # if notebook_id != '4444309':
                # continue
            if os.path.exists('HybridPipeGen/core/tmpdata/validation_prenotebook_res/'+str(notebook_id)+'.npy'):
                continue
            notebook_id_py = notebook_id + '.py'

            run_origin_path('HybridPipeGen/core/tmpdata/validation_prenotebook_code/'+notebook_id_py)
            # os.system('python validation_prenotebook_code/'+notebook_id_py)

def load_code_1(path, test):
    if '.json' in path:
        with open(path, 'r') as f:
            dict_ = json.load(f)
            code = dict_['code']
    elif '.py' in path:
        with open(path, 'r') as f:
            code = f.read()

    code = cleaning_origin(code)
        # #print('self.origin_code', self.origin_code)
    # try:
    #     #print("!!!!!!!!!!!!!!!!!!!!!!!")
    #     lines = code.split('\n')
    #     new_code = ''
    #     end_code= ''
    #     for index,line in enumerate(lines):
    #         if index <  len(lines)- 35:
    #             new_code += line
    #             new_code += '\n'
    #         else:
    #             end_code += line
    #             end_code += '\n'

    #     # temp = remove_model(new_code)
    #     # code = temp
    #     code += end_code
    # except:
    #     pass
    lines = code.split('\n')
    new_code = ""
    for index,line in enumerate(lines):
        line = exchange_code(line)  
        new_code += line
        new_code +="\n"
    return new_code
def run_path_1(path, replace_code, test, running_id=None, force_rerun=False):
    # with open(path, 'r') as f:

    code = load_code_1(path, test)
    # #print(code)
    notebook_id = path.split("/")[-2]
    seq_id = path.split("/")[-1].split(".")[0]
    #print(seq_id)
    #print(notebook_id)
    pro = Preprocessing()
    if len(replace_code) > 0:
        #print("??????")
        code = code.replace("prenotebook_res/"+notebook_id +'.npy', replace_code+notebook_id+'/' + seq_id+'.npy')
        #print('HybridPipeGen/core/tmpdata/'+replace_code + notebook_id)
        #print(not os.path.exists('HybridPipeGen/core/tmpdata/'+replace_code + notebook_id))
        if not os.path.exists('HybridPipeGen/core/tmpdata/'+replace_code + notebook_id):
            #print('mkdir ', 'HybridPipeGen/core/tmpdata/'+replace_code + notebook_id)
            os.mkdir('HybridPipeGen/core/tmpdata/'+replace_code + notebook_id)
    new_c = ''
    lines = code.split('\n')
    for index,line in enumerate(lines):
        if "check_output(" in line:
            temp = line
            lines[index] = "#"+temp
        # #print(line)
    # #print("----------------3---------------------------------------------------------")
    for line in lines:
        
        if 'np.save' in line and ('_data_x.npy"' in line or '_data_y.npy' in line):
            new_c += '#'+line
            new_c += '\n'
        else:
            new_c += line
            new_c += '\n'
    # #print(new_c)
    # #print("--------------------4-------------------------")
    code = pro.run_one_code(notebook_id, new_c, dataset_root_path + pro.info_triple[notebook_id]['dataset_name'], 0)


def run_notebook_path(path, replace_code, test, running_id=None, force_rerun=False):
    # with open(path, 'r') as f:

    code = load_code_1(path, test)
    # #print(code)
    notebook_id = path.split("/")[-1].split('.')[0]

    #print(notebook_id)
    pro = Preprocessing()
    if len(replace_code) > 0:
        #print("??????")
        code = code.replace("prenotebook_res/"+notebook_id +'.npy', replace_code+notebook_id+'.npy')
        #print('HybridPipeGen/core/tmpdata/'+replace_code + notebook_id)

    new_c = ''
    lines = code.split('\n')
    for index,line in enumerate(lines):
        if "check_output(" in line:
            temp = line
            lines[index] = "#"+temp
        # #print(line)
    # #print("----------------3---------------------------------------------------------")
    for line in lines[0:-28]:
        
        if 'np.save' in line and ('_data_x.npy"' in line or '_data_y.npy' in line):
            new_c += '#'+line
            new_c += '\n'
        else:
            new_c += line
            new_c += '\n'
    # #print(new_c)
    # #print("--------------------4-------------------------")
    code = pro.run_one_code(notebook_id, new_c, dataset_root_path + pro.info_triple[notebook_id]['dataset_name'], 0)

<<<<<<< HEAD
def run_max_hybrid():
=======
def run_max_hai():
>>>>>>> 667bbf9cb98902e69354ddf34ecd5389817d14f7
    res = []
    if os.path.exists("rl_step2_test_time.json"):
        with open("rl_step2_test_time.json", 'r') as f:
            validation_running_time = json.load(f)
    else:
        validation_running_time = {}
    with open('max_index_rl.json','r',encoding='utf8')as fp:
        json_data = json.load(fp)
        for notebook_id in json_data:
            # if len(os.listdir("HybridPipeGen/core/tmpdata/merge_max_result_planB/"+str(notebook_id)))!=0:
            #     continue
            if str(notebook_id) not in validation_running_time:
                validation_running_time[str(notebook_id)] = {}
            L = list(json_data[notebook_id].items())
            if os.path.exists("HybridPipeGen/core/tmpdata/merge_max_result_rl/"+str(notebook_id)):
                path = os.listdir("HybridPipeGen/core/tmpdata/merge_max_result_rl/"+str(notebook_id))
                if os.path.exists("HybridPipeGen/core/tmpdata/merge_max_result_rl/"+str(notebook_id)+'/'+L[0][0]):
                    continue
                elif len(path)!=0:
                    os.remove("HybridPipeGen/core/tmpdata/merge_max_result_rl/"+str(notebook_id)+'/'+path[0])
            for index,seq_id_file_ in  enumerate(json_data[notebook_id]):              
                if os.path.exists("HybridPipeGen/core/tmpdata/merge_max_result_rl/"+str(notebook_id)) and len(os.listdir("HybridPipeGen/core/tmpdata/merge_max_result_rl/"+str(notebook_id)))!=0:
                    break
                if seq_id_file_ != "origin.npy":
                    item_name = seq_id_file_.split(".npy")[0]
                    item = item_name + ".json"
                    if os.path.exists("HybridPipeGen/core/tmpdata/merge_max_result_rl/"+str(notebook_id)+'/'+seq_id_file_):
                        continue
                    start_time = time.time()
                    run_path_1('rl_test_merge_code/'+notebook_id+'/'+item, replace_code = 'merge_max_result_rl/',test=False)
                    end = time.time()
                    validation_running_time[str(notebook_id)] = end-start_time
                    with open("planB_step1_time.json", 'w') as f:
                        json.dump(validation_running_time, f)
                else:
                    if not os.path.exists("HybridPipeGen/core/tmpdata/merge_max_result_rl/"+str(notebook_id)):
                        os.mkdir("HybridPipeGen/core/tmpdata/merge_max_result_rl/"+str(notebook_id))
                    #print(notebook_id)
                    res.append(notebook_id)
                    shutil.copyfile('HybridPipeGen/core/tmpdata/prenotebook_res/'+notebook_id+'.npy', 'HybridPipeGen/core/tmpdata/merge_max_result_rl/'+notebook_id+'/origin.npy')
                    break
    #print(len(res))
    #print(res)

<<<<<<< HEAD
def run_max_hybrid_add3():
=======
def run_max_hai_add3():
>>>>>>> 667bbf9cb98902e69354ddf34ecd5389817d14f7
    res = []
    with open('planB_cross_max_index_add3_222.json','r',encoding='utf8')as fp:
        json_data = json.load(fp)
        for notebook_id in json_data:
            # if len(os.listdir("HybridPipeGen/core/tmpdata/merge_max_result_planB/"+str(notebook_id)))!=0:
            #     continue
            if os.path.exists("HybridPipeGen/core/tmpdata/merge_max_result_planB_add3/"+str(notebook_id)):
                path = os.listdir("HybridPipeGen/core/tmpdata/merge_max_result_planB_add3/"+str(notebook_id))
                # #print(json_data[notebook_id])
                L = list(json_data[notebook_id].items())
                # #print(L[0][1])
                if os.path.exists("HybridPipeGen/core/tmpdata/merge_max_result_planB_add3/"+str(notebook_id)+'/'+L[0][0]):
                    continue
                elif len(path)!=0:
                    os.remove("HybridPipeGen/core/tmpdata/merge_max_result_planB_add3/"+str(notebook_id)+'/'+path[0])
            for index,seq_id_file_ in  enumerate(json_data[notebook_id]):              
                if os.path.exists("HybridPipeGen/core/tmpdata/merge_max_result_planB_add3/"+str(notebook_id)) and len(os.listdir("HybridPipeGen/core/tmpdata/merge_max_result_planB_add3/"+str(notebook_id)))!=0:
                    break
                
                if seq_id_file_ != "origin.npy":
                    item_name = seq_id_file_.split(".npy")[0]
                    item = item_name + ".json"
                    if os.path.exists("HybridPipeGen/core/tmpdata/merge_max_result_planB_add3/"+str(notebook_id)+'/'+seq_id_file_):
                        continue
                    run_path_1('planB_test_merge_code_add_rule3/'+notebook_id+'/'+item, replace_code = 'merge_max_result_planB_add3/',test=False)
                    
                else:
                    if not os.path.exists("HybridPipeGen/core/tmpdata/merge_max_result_planB_add3/"+str(notebook_id)):
                        os.mkdir("HybridPipeGen/core/tmpdata/merge_max_result_planB_add3/"+str(notebook_id))
                    #print(notebook_id)
                    res.append(notebook_id)
                    shutil.copyfile('HybridPipeGen/core/tmpdata/prenotebook_res/'+notebook_id+'.npy', 'HybridPipeGen/core/tmpdata/merge_max_result_planB_add3/'+notebook_id+'/origin.npy')
                    break
    #print(len(res))
    #print(res)

def run_validation():
    # notebooks = os.listdir('HybridPipeGen/core/tmpdata/prenotebook_res/')
    with open('return_cross.json','r') as f:
            notebooks = json.load(f)
    exist_f = open("/home/yxm/staticfg-master/shibai.txt", "r")
    exist = exist_f.readlines()
    exitst_ = [x.strip("\n") for x in exist]
    
    notebook_m = os.listdir('merge_code_1')
    # notebooks = list(set(notebooks) & set(notebook_m))
    # notebooks = list(set(notebooks) & set(exitst_))
    notebooks = os.listdir('HybridPipeGen/core/tmpdata/cross_val_res')
    notebooks = list(set(notebooks) & set(notebook_m))
    for notebook_id in exitst_:
        
        # notebook_id = notebook_id_file.split('.')[0]
        exist_f = open("/home/yxm/staticfg-master/origin_1756.txt", "r")
        exist = exist_f.readlines()
        exitst_ = [x.strip("\n") for x in exist]
        # if notebook_id !='7752056' :
        #         continue
        # if notebook_id not in exitst_:
            # continue
        # if notebook_id == 'adrkr7_kernel5ee8529b4e' or notebook_id == 'd34th4ck3r_kernel649ffd0ea8':
        #     continue
        # exist_f = open("/home/yxm/staticfg-master/all_same.txt", "r")
        # exist = exist_f.readlines()
        # exitst_ = [x.strip("\n") for x in exist]
        # if notebook_id not in exitst_:
        #     continue
        # if notebook_id == "arunimsamudra_heart-disease-data-visualisation-and-prediction" :#太慢了xgb
        #     continue     

        filelist = os.listdir('HybridPipeGen/core/tmpdata/cross_validation_code/'+notebook_id)
        # filelist.sort()
        # if not os.path.exists("HybridPipeGen/core/tmpdata/merge_validation_result_1600/"+notebook_id):
            # os.mkdir("HybridPipeGen/core/tmpdata/merge_validation_result_1600/"+notebook_id)
        # if notebook_id != "1395715" :
        #     continue 
        # if len(filelist)<=1000:
        #     continue
        for item in filelist:
            # if item == "7.json" and notebook_id == "azmatsiddique_social-network-ads-gridsearch-eda":
            #     continue 


            # if item != "origin.json":
            #     continue

                
            # if item.split('.')[0] != analyze_score[notebook_id]['best_index'].split('.')[0]:
                # continue
            if os.path.exists("HybridPipeGen/core/tmpdata/cross_val_res/"+str(notebook_id)+'/'+item.split('.')[0]+'.npy'):
                continue
            # if notebook_id != str(322282) or item !='25.json':
            #     continue
            #print("item,", item)
            #print("notebook,",notebook_id)
            # if notebook_id == "drfrank_heart-disease-visualization-and-machine-learning" and item == "42.json":
            #     continue
            start_time = time.time()
            try:
                run_path('HybridPipeGen/core/tmpdata/cross_validation_code/'+notebook_id+'/'+item, replace_code = 'cross_val_res/',test=False)
            # except:
            except func_timeout.exceptions.FunctionTimedOut:
                if os.path.exists("HybridPipeGen/core/tmpdata/validation_running_time_cross.json"):
                    with open("HybridPipeGen/core/tmpdata/validation_running_time_cross.json", 'r') as f:
                        validation_running_time = json.load(f)
                if str(notebook_id) not in validation_running_time:
                    validation_running_time[str(notebook_id)] = {}
                end = time.time()
                seq_id = item.split('.json')[0]
                validation_running_time[str(notebook_id)][seq_id] = end-start_time
                # #print("xxxxxxxxxx")
                # #print(validation_running_time)
                with open("HybridPipeGen/core/tmpdata/validation_running_time_cross.json", 'w') as f:
                    json.dump(validation_running_time, f)
                #print('执行函数超时')
            gc.collect()
            # break
        # break
        # gc.collect()
def run_validation_planB():
    # notebooks = os.listdir('HybridPipeGen/core/tmpdata/prenotebook_res/')
    notebooks = os.listdir('HybridPipeGen/core/tmpdata/planB_cross_validation_code')
    notebooks.sort()
    if os.path.exists("HybridPipeGen/core/tmpdata/planB_validation_running_time_cross.json"):
        with open("HybridPipeGen/core/tmpdata/planB_validation_running_time_cross.json", 'r') as f:
            validation_running_time = json.load(f)
    else:
        validation_running_time = {}
   
    for index,notebook_id in enumerate(notebooks):
        # if index == 16 or index ==34 or index ==99 or index == 100 or index == 150:
        #     #print(notebook_id)
            # continue
        if notebook_id != 'adikeshri_identifying-gender-from-voice-shortest-guide':
            continue
        # if index<=150:
        #     continue
        filelist = os.listdir('HybridPipeGen/core/tmpdata/planB_cross_validation_code/'+notebook_id)
        if str(notebook_id) not in validation_running_time:
            validation_running_time[str(notebook_id)] = {}
        for item in filelist:
            if os.path.exists("HybridPipeGen/core/tmpdata/planB_cross_val_res/"+str(notebook_id)+'/'+item.split('.')[0]+'.npy'):
                continue
            # if notebook_id != str(322282) or item !='25.json':
            #     continue
            #print('index',index)
            #print("item,", item)
            #print("notebook,",notebook_id)
            # if notebook_id == "drfrank_heart-disease-visualization-and-machine-learning" and item == "42.json":
            #     continue
            start_time = time.time()
            seq_id = item.split('.json')[0]
            try:
                run_path('HybridPipeGen/core/tmpdata/planB_cross_validation_code/'+notebook_id+'/'+item, replace_code = 'planB_cross_val_res/',test=False)
            # except:
            except func_timeout.exceptions.FunctionTimedOut:
                if os.path.exists("HybridPipeGen/core/tmpdata/planB_validation_running_time_cross.json"):
                    with open("HybridPipeGen/core/tmpdata/planB_validation_running_time_cross.json", 'r') as f:
                        validation_running_time = json.load(f)
                if str(notebook_id) not in validation_running_time:
                    validation_running_time[str(notebook_id)] = {}
                end = time.time()
                validation_running_time[str(notebook_id)][seq_id] = end-start_time
                with open("HybridPipeGen/core/tmpdata/planB_validation_running_time_cross.json", 'w') as f:
                    json.dump(validation_running_time, f)
                #print('执行函数超时')
            else:
                end = time.time()
                validation_running_time[str(notebook_id)][seq_id] = end-start_time
                with open("HybridPipeGen/core/tmpdata/planB_validation_running_time_cross.json", 'w') as f:
                    json.dump(validation_running_time, f)
            gc.collect()        
 
def run_validation_rl():
    # notebooks = os.listdir('HybridPipeGen/core/tmpdata/prenotebook_res/')
    notebooks = os.listdir('HybridPipeGen/core/tmpdata/rl_cross_validation_code')
    notebooks.sort()
    if os.path.exists("HybridPipeGen/core/tmpdata/rl_validation_running_time_cross.json"):
        with open("HybridPipeGen/core/tmpdata/rl_validation_running_time_cross.json", 'r') as f:
            validation_running_time = json.load(f)
    else:
        validation_running_time = {}
   
    for index,notebook_id in enumerate(notebooks):
        # if index == 16 or index ==34 or index ==99 or index == 100 or index == 150:
        #     #print(notebook_id)
            # continue
        # #print(index)
        filelist = os.listdir('HybridPipeGen/core/tmpdata/rl_cross_validation_code/'+notebook_id)
        if str(notebook_id) not in validation_running_time:
            validation_running_time[str(notebook_id)] = {}
        for item in filelist:
            if os.path.exists("HybridPipeGen/core/tmpdata/rl_cross_val_res/"+str(notebook_id)+'/'+item.split('.')[0]+'.npy'):
                continue
            # if notebook_id != str(322282) or item !='25.json':
            #     continue
            #print('index',index)
            #print("item,", item)
            #print("notebook,",notebook_id)
            # if notebook_id == "drfrank_heart-disease-visualization-and-machine-learning" and item == "42.json":
            #     continue
            start_time = time.time()
            seq_id = item.split('.json')[0]
            try:
                run_path('HybridPipeGen/core/tmpdata/rl_cross_validation_code/'+notebook_id+'/'+item, replace_code = 'rl_cross_val_res/',test=False)
            # except:
            except func_timeout.exceptions.FunctionTimedOut:
                if os.path.exists("HybridPipeGen/core/tmpdata/rl_validation_running_time_cross.json"):
                    with open("HybridPipeGen/core/tmpdata/rl_validation_running_time_cross.json", 'r') as f:
                        validation_running_time = json.load(f)
                if str(notebook_id) not in validation_running_time:
                    validation_running_time[str(notebook_id)] = {}
                end = time.time()
                validation_running_time[str(notebook_id)][seq_id] = end-start_time
                with open("HybridPipeGen/core/tmpdata/rl_validation_running_time_cross.json", 'w') as f:
                    json.dump(validation_running_time, f)
                #print('执行函数超时')
            else:
                end = time.time()
                validation_running_time[str(notebook_id)][seq_id] = end-start_time
                with open("HybridPipeGen/core/tmpdata/rl_validation_running_time_cross.json", 'w') as f:
                    json.dump(validation_running_time, f)
            gc.collect()
        
def generate_notebook_id():
    info_triple = np.load("../statsklearn/NewDataInfoTriple.npy", allow_pickle=True).item()
        # filenames = os.listdir("notebook597")
    notebooks_ids = list(info_triple.keys())
    id_ = 1
    nnid2nid = {}
    for nid in notebooks_ids:
        nnid2nid[nid] = id_
        id_ += 1

    with open("name2id.json", 'w') as f:
        json.dump(nnid2nid, f)

def how_much_run():
    with open('../statsklearn/new_notebook.json', 'r') as f:
        notebook_dataset_no_empty = json.load(f)

    has_label_set = set([notebook_id.split('.')[0] for notebook_id in notebook_dataset_no_empty.keys()])
    # filelist = os.listdir('../staticfg-master/prenotebook_res/')
    # has_notebook_res = set([int(filename.split('.')[0]) for filename in filelist])
    
    # notebooks = list(has_label_set & has_notebook_res)
    info_triple = np.load('../statsklearn/NewDataInfoTriple.npy', allow_pickle=True).item()
    notebooks = list(has_label_set & set(info_triple.keys()))
    #print(len(notebooks))
    # info_triple = np.load('/home/chensibei/statsklearn/stat_data_info_triple.npy', allow_pickle=True).item()
   
    all_ = 0
    d_exist = 0
    h_exist = 0
    run_fail = 0

    for notebook_id in notebooks:
        # try:
        all_ += 1
        if os.path.exists("../staticfg-master/deepline_only_HybridPipeGen/core/tmpdata/"+ str(notebook_id) + "_score.json"):
            d_exist += 1
        if os.path.exists("../staticfg-master/HybridPipeGen/core/tmpdata/prenotebook_res/"+ str(notebook_id) + ".npy"):
            h_exist += 1
    #print('runed', len(os.listdir("../staticfg-master/HybridPipeGen/core/tmpdata/prenotebook_res/")))
    #print('deepline exist', d_exist)
    #print('human exist', h_exist)
    #print('all', all_)
    runed_list = os.listdir('HybridPipeGen/core/tmpdata/prenotebook_res')
    with open("runed1.json",'w') as f:
        json.dump(runed_list,f)
def run_validation_new():
    notebooks = os.listdir('HybridPipeGen/core/tmpdata/merge_validation_code_new')
    for notebook_id_file in notebooks:
        notebook_id = notebook_id_file.split('.')[0]
        exist_f = open("/home/yxm/staticfg-master/test_error.txt", "r")
        exist = exist_f.readlines()
        exitst_ = [x.strip("\n") for x in exist]
        if notebook_id not in exitst_:
            continue 
        if notebook_id == "adichamoli_support-vector-regression" :#跑完
            continue
        if notebook_id == "sumitkumar02_eda-and-classifying-heart-disease-patients":#跑完
            continue
        if notebook_id == "apollopower_ibm-attrition-visualization-random-forests":
            continue 
        if notebook_id == "adeyoyintemidayo_diabetes-prediction": #跑完
            continue 
        # if notebook_id == "mysecret_hw-churn-modelling":#总被杀
        #     continue
        # if notebook_id =="sidsiddu_kernal-svm":#卡
        #     continue
        # if notebook_id != "azmatsiddique_social-network-ads-gridsearch-eda" and notebook_id != "sidsiddu_kernal-svm" and notebook_id != "mysecret_hw-churn-modelling":#被杀
        #     continue
        if notebook_id == "arya24_notebook1430e96cf2":#跑完
            continue
        if notebook_id == 'online0147_10-19-19':#跑完
            continue
        if notebook_id == 'atilayyinanc_telco-churn':#跑完
            continue
        filelist = os.listdir('HybridPipeGen/core/tmpdata/merge_validation_code_new/'+notebook_id)
        # filelist.sort()
        # if notebook_id != "championrunner_graduate-admissions" :
        #     continue 
        for item in filelist:
            # if item != "origin.json":
            #     continue
            if item == "7.json" and notebook_id == "sidsiddu_kernal-svm":
                continue
            if os.path.exists("HybridPipeGen/core/tmpdata/merge_validation_result_new/"+str(notebook_id)+'/'+item.split('.')[0]+'.npy'):
                continue
            #print("item,", item)
            #print("notebook,",notebook_id)
            try:
                run_path('HybridPipeGen/core/tmpdata/merge_validation_code_new/'+notebook_id+'/'+item, replace_code = 'merge_validation_result_new/',test=False)
            # except:
            except func_timeout.exceptions.FunctionTimedOut:
                print('执行函数超时')
            except:
                print("其他错误")
            gc.collect()
def load_code_base(path, test):
    with open(path, 'r') as f:
        dict_ = json.load(f)
        code = dict_['code']

    code = cleaning_origin(code)
        # #print('self.origin_code', self.origin_code)
    lines = code.split('\n')
    new_code = ""
    for index,line in enumerate(lines):
        line = exchange_code(line)  
        new_code += line
        new_code +="\n"
    return new_code
def run_path_base(path, replace_code, test, running_id=None, force_rerun=False):
    # with open(path, 'r') as f:
    # info_triple = np.load("merged_info_triple.npy", allow_pickle=True).item()
    with open('task_all.json','r')as f:
        tasks = json.load(f)
    
    code = load_code_base(path, test)
    # #print(code)
    task = path.split("/")[-1].split('.')[0]
    notebook_id = list(tasks[task]['notebook_list'])[0]
    #print(task)
    pro = Preprocessing()
    if len(replace_code) > 0:
        #print("??????")
        code = code.replace("prenotebook_res/"+notebook_id +'.npy', replace_code + task +'.npy')
        #print('HybridPipeGen/core/tmpdata/'+replace_code + task)
    #print(code)
    code = pro.run_one_code(task, code, dataset_root_path + tasks[task]['dataset'], 0)
def run_base():
    res = []
    path = os.listdir('HybridPipeGen/core/tmpdata/base_code')
    for task in path:
        # if task != 'adammaus_predicting-churn-for-bank-customers_SVC_13.json':
        #     continue
        task_id = task.split('.json')[0]
        if os.path.exists("HybridPipeGen/core/tmpdata/base_result/"+str(task_id)+'.npy'):
            continue
        run_path_base('HybridPipeGen/core/tmpdata/base_code/'+task, replace_code = 'HybridPipeGen/core/tmpdata/base_result/',test=False)
        res.append(task)
    #print(len(res))
    #print(res)



def final_test_hi():
    res = []
    run_code_filelist = os.listdir('transdata/this_prenotebook_code/')
    
    for notebook_id_py in run_code_filelist:
        #print('notebook_id_py', notebook_id_py)
        if os.path.exists('final_hi_res/' + notebook_id_py.split('.')[0] + '.npy'):
            continue
        run_notebook_path('transdata/this_prenotebook_code/'+notebook_id_py, replace_code = 'final_hi_res/',test=False)

<<<<<<< HEAD
def final_planB_hybrid():
=======
def final_planB_hai():
>>>>>>> 667bbf9cb98902e69354ddf34ecd5389817d14f7
    res = []
    run_code_filelist = os.listdir('transdata/this_planB_test_merge_code_max/')
    origin_notebooks = []
    for notebook_id in run_code_filelist:
        seq_list = os.listdir('transdata/this_planB_test_merge_code_max/' + notebook_id)
<<<<<<< HEAD
        if not os.path.exists('HybridPipeGen/core/tmpdata/final_planB_hybrid'):
            os.mkdir('HybridPipeGen/core/tmpdata/final_planB_hybrid')
=======
        if not os.path.exists('HybridPipeGen/core/tmpdata/final_planB_hai'):
            os.mkdir('HybridPipeGen/core/tmpdata/final_planB_hai')
>>>>>>> 667bbf9cb98902e69354ddf34ecd5389817d14f7
        if notebook_id != 'kaggledroid_predicting-heart-disease-using-ensemble':
            continue
        if len(seq_list) == 0:
            origin_notebooks.append(notebook_id)
            #print('xxxx')
            continue
        
        seq_id = seq_list[0].split('.')[0]
        #print('notebook_id', notebook_id)
<<<<<<< HEAD
        if os.path.exists('HybridPipeGen/core/tmpdata/final_planB_hybrid/'+notebook_id + '/' + seq_id + '.npy'):
            continue
        run_path_1('transdata/this_planB_test_merge_code_max/'+notebook_id + '/' + seq_id + '.json', replace_code = 'final_planB_hybrid/',test=False)
    # with open('origin_planB_notebooks.json','w') as f:
        # json.dump(origin_notebooks, f)

def final_rl_hybrid():
=======
        if os.path.exists('HybridPipeGen/core/tmpdata/final_planB_hai/'+notebook_id + '/' + seq_id + '.npy'):
            continue
        run_path_1('transdata/this_planB_test_merge_code_max/'+notebook_id + '/' + seq_id + '.json', replace_code = 'final_planB_hai/',test=False)
    # with open('origin_planB_notebooks.json','w') as f:
        # json.dump(origin_notebooks, f)

def final_rl_hai():
>>>>>>> 667bbf9cb98902e69354ddf34ecd5389817d14f7
    res = []
    run_code_filelist = os.listdir('transdata/this_rl_test_merge_code_max/')
    origin_notebooks = []
    for notebook_id in run_code_filelist:
        seq_list = os.listdir('transdata/this_rl_test_merge_code_max/' + notebook_id)
<<<<<<< HEAD
        if not os.path.exists('HybridPipeGen/core/tmpdata/final_rl_hybrid'):
            os.mkdir('HybridPipeGen/core/tmpdata/final_rl_hybrid')
=======
        if not os.path.exists('HybridPipeGen/core/tmpdata/final_rl_hai'):
            os.mkdir('HybridPipeGen/core/tmpdata/final_rl_hai')
>>>>>>> 667bbf9cb98902e69354ddf34ecd5389817d14f7
        if len(seq_list) == 0:
            origin_notebooks.append(notebook_id)
            # print
            continue
        seq_id = seq_list[0].split('.')[0]
        #print('notebook_id', notebook_id)
<<<<<<< HEAD
        if os.path.exists('HybridPipeGen/core/tmpdata/final_rl_hybrid/'+notebook_id + '/' + seq_id + '.npy'):
            continue
        run_path_1('transdata/this_rl_test_merge_code_max/'+notebook_id + '/' + seq_id + '.json', replace_code = 'final_rl_hybrid/',test=False)
=======
        if os.path.exists('HybridPipeGen/core/tmpdata/final_rl_hai/'+notebook_id + '/' + seq_id + '.npy'):
            continue
        run_path_1('transdata/this_rl_test_merge_code_max/'+notebook_id + '/' + seq_id + '.json', replace_code = 'final_rl_hai/',test=False)
>>>>>>> 667bbf9cb98902e69354ddf34ecd5389817d14f7
    # with open('origin_rl_notebooks.json','w') as f:
        # json.dump(origin_notebooks, f)

def run_one_validation_rl(notebook_id):
    filelist = os.listdir('HybridPipeGen/core/tmpdata/rl_cross_validation_code/'+notebook_id)
    if not os.path.exists('HybridPipeGen/core/tmpdata/rl_cross_val_res'):
        os.mkdir('HybridPipeGen/core/tmpdata/rl_cross_val_res')
    for item in filelist:
        # if os.path.exists("HybridPipeGen/core/tmpdata/rl_cross_val_res/"+str(notebook_id)+'/'+item.split('.')[0]+'.npy'):
        #     continue
        #print("item,", item)
        #print("notebook,",notebook_id)
        start_time = time.time()
        seq_id = item.split('.json')[0]
        try:
            run_path('HybridPipeGen/core/tmpdata/rl_cross_validation_code/'+notebook_id+'/'+item, replace_code = 'rl_cross_val_res/',test=False)
        except func_timeout.exceptions.FunctionTimedOut:
            print('执行函数超时')
        except:
            print('其他错误')
        gc.collect()
<<<<<<< HEAD
def run_one_max_hybrid(notebook_id):
=======
def run_one_max_hai(notebook_id):
>>>>>>> 667bbf9cb98902e69354ddf34ecd5389817d14f7
    res = []
    with open('HybridPipeGen/core/tmpdata/max_index.json','r',encoding='utf8')as fp:
        json_data = json.load(fp)
    L = list(json_data[notebook_id].items())
    if not os.path.exists('HybridPipeGen/core/tmpdata/merge_max_result_rl'):
        os.mkdir('HybridPipeGen/core/tmpdata/merge_max_result_rl')
    if os.path.exists("HybridPipeGen/core/tmpdata/merge_max_result_rl/"+str(notebook_id)):
        path = os.listdir("HybridPipeGen/core/tmpdata/merge_max_result_rl/"+str(notebook_id))
        # if os.path.exists("HybridPipeGen/core/tmpdata/merge_max_result_rl/"+str(notebook_id)+'/'+L[0][0]):
        #     return
        if len(path)!=0:
            os.remove("HybridPipeGen/core/tmpdata/merge_max_result_rl/"+str(notebook_id)+'/'+path[0])
    for index,seq_id_file_ in  enumerate(json_data[notebook_id]):              
        if os.path.exists("HybridPipeGen/core/tmpdata/merge_max_result_rl/"+str(notebook_id)) and len(os.listdir("HybridPipeGen/core/tmpdata/merge_max_result_rl/"+str(notebook_id)))!=0:
            break
        if seq_id_file_ != "origin.npy":
            item_name = seq_id_file_.split(".npy")[0]
            item = item_name + ".json"
            if os.path.exists("HybridPipeGen/core/tmpdata/merge_max_result_rl/"+str(notebook_id)+'/'+seq_id_file_):
                continue
            run_path_1('HybridPipeGen/core/tmpdata/rl_test_merge_code/'+notebook_id+'/'+item, replace_code = 'merge_max_result_rl/',test=False)
        else:
            shutil.copyfile('HybridPipeGen/core/tmpdata/prenotebook_res/'+notebook_id+'.npy', 'HybridPipeGen/core/tmpdata/merge_max_result_rl/'+notebook_id+'/origin.npy')
            break
# if __name__ == "__main__":
    # sys.stdout = Logger('running.log')
    # sys.stderr = Logger('running.log')
    # generate_notebook_id()
    # pro = Preprocessing()
    # pro.batch_processing()
    # pro.profiling_code('royrangan7_heart-disease', True)
    # how_much_run()
    # run_test(-1,spe_notebook_id=10826870, spe_seq_id=5)
    # test_time(0)
    # test_time(1)
    # test_time(2)
    # test_time(3)
    # run_validation()
    # run_validation_planB()
    # run_validation_rl()
    # # run_validation_new()
    #run_origin_validation()
    # #print(pro.train_test_split_no_label)
<<<<<<< HEAD
    # run_max_hybrid_add3()
    # run_max_hybrid()
=======
    # run_max_hai_add3()
    # run_max_hai()
>>>>>>> 667bbf9cb98902e69354ddf34ecd5389817d14f7
    # run_base()
    # run_notebook_only()
    # def change_train_test_split(self, ):
    #     4133598.npy 3489576.npy 3900293.npy rerun
    # final_test_hi()
<<<<<<< HEAD
    # final_planB_hybrid()
    # final_rl_hybrid()
=======
    # final_planB_hai()
    # final_rl_hai()
>>>>>>> 667bbf9cb98902e69354ddf34ecd5389817d14f7
