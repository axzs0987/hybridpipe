import json
import numpy as np
import os
from HybridPipeGen.core.preprocessing import *
from HybridPipeGen.core.NotebookGraph import profile_hipipe
from HybridPipeGen.core.merge import *
from MLPipeGen.MLPipe import MLPipe
import pandas as pd
import shutil

class HybridPipe:
    def __init__(self, notebook_path, data_path, label_index, model):
        notebook_id = notebook_path.split('/')[-1].split('.')[0]
        # print(notebook_id)
        self.notebook_id = notebook_id
        self.data_path = data_path
        self.label_index = label_index
        self.model = model
        self.update_info()

    def update_info(self):
        info_triple = np.load('HybridPipeGen/core/merged_info_triple.npy',allow_pickle=True).item()
        with open("HybridPipeGen/core/merged_dataset_label.json",'r') as f:
            merged_dataset_label = json.load(f)
        with open("MLPipeGen/core/jsons/classification_task_dic.json",'r') as f:
            classification_task_dic = json.load(f)
        with open("MLPipeGen/core/jsons/test_index.json",'r') as f:
            test_index = json.load(f)
        task_id = str(len(classification_task_dic))
        test_index.append(task_id)
        task_name = self.data_path.split('/')[-2]+'_'+self.model+'_'+str(self.label_index)
        exist_task = False
        for task in classification_task_dic:
            if task_name == classification_task_dic[task]['task_name']:
                exist_task = True
                break
        if not exist_task:
            classification_task_dic[task_id] = {}
            classification_task_dic[task_id]['dataset'] = self.data_path.split('/')[-2]
            classification_task_dic[task_id]['csv_file'] = self.data_path.split('/')[-1]
            classification_task_dic[task_id]['label'] = str(self.label_index)
            classification_task_dic[task_id]['model'] = self.model
            classification_task_dic[task_id]['task_name'] = task_name
            
        with open("MLPipeGen/core/jsons/classification_task_dic.json",'w') as f:
            json.dump(classification_task_dic, f)    
        if self.notebook_id not in info_triple:
            info_triple[self.notebook_id] = {}
            info_triple[self.notebook_id]['dataset_name'] = self.data_path
            info_triple[self.notebook_id]['dataset_id'] = self.data_path
            info_triple[self.notebook_id]['model_type'] = self.model
        if self.notebook_id not in merged_dataset_label:
            data = pd.read_csv(self.data_path)
            columns = data.columns
            column_index = {}
            for index,col in enumerate(columns):
                column_index[col] = index
            merged_dataset_label[self.notebook_id] = {}
            merged_dataset_label[self.notebook_id]['dataset'] = self.data_path
            merged_dataset_label[self.notebook_id]['column_index'] = column_index
            merged_dataset_label[self.notebook_id]['index'] = [self.label_index]

        np.save('HybridPipeGen/core/merged_info_triple.npy', info_triple)
        with open("HybridPipeGen/core/merged_dataset_label.json",'w') as f:
            json.dump(merged_dataset_label, f)
        with open("MLPipeGen/core/jsons/classification_task_dic.json",'w') as f:
            json.dump(classification_task_dic, f)
        with open("MLPipeGen/core/jsons/test_index.json",'w') as f:
            json.dump(test_index, f)

    def combine(self):
        profile_hipipe(self.notebook_id)
        if not os.path.exists('HybridPipeGen/core/tmpdata/rl_test_merge_code'):
            os.mkdir('HybridPipeGen/core/tmpdata/rl_test_merge_code')
        if not os.path.exists('HybridPipeGen/core/tmpdata/rl_test_merge_code_py'):
            os.mkdir('HybridPipeGen/core/tmpdata/rl_test_merge_code_py')
        if not os.path.exists('HybridPipeGen/core/tmpdata/rl_cross_validation_code'):
            os.mkdir('HybridPipeGen/core/tmpdata/rl_cross_validation_code')
        if not os.path.exists('HybridPipeGen/core/tmpdata/rl_cross_validation_code_py'):
            os.mkdir('HybridPipeGen/core/tmpdata/rl_cross_validation_code_py')
        merger = Merger(4)
        merger.merging_one_notebook_rl(self.notebook_id, self.ai_sequence)
        transform_one_validation_rl(self.notebook_id)
        transform_one_origin_validation_code(self.notebook_id)

    def generate_mlpipe(self):
        mlpipe = MLPipe(self.data_path, self.label_index)
        self.ai_sequence = mlpipe.inference()


    def evaluate_hi(self):
        pro = Preprocessing()
        if not os.path.exists('HybridPipeGen/core/tmpdata/prenotebook_code'):
            os.mkdir('HybridPipeGen/core/tmpdata/prenotebook_code')
        if not os.path.exists('HybridPipeGen/core/tmpdata/runned_notebook'):
            os.mkdir('HybridPipeGen/core/tmpdata/runned_notebook')
        if not os.path.exists('HybridPipeGen/core/tmpdata/prenotebook_res'):
            os.mkdir('HybridPipeGen/core/tmpdata/prenotebook_res')
        if not os.path.exists('HybridPipeGen/core/tmpdata/prenotebook_varibles_index'):
            os.mkdir('HybridPipeGen/core/tmpdata/prenotebook_varibles_index')
        res = pro.profiling_code(self.notebook_id, need_remove_model=1)
        add_faile = 0
        #print('save_code', res)
        print("\033[0;33;40mstart run HIPipe on test dataset:\033[0m")
        if res == True:
            pro.run_origin_test(self.notebook_id, need_try_again=2)
        else:
            add_faile += 1
        print('\n')

    def evalaute_hybrid(self):
        run_one_validation_rl(self.notebook_id)
        print('\n')
        self.select_best_hybrid()
        run_one_max_hybrid(self.notebook_id)
        print('\n')

    def select_best_hybrid(self):
        max_index ={}
        max_index[self.notebook_id] = {}
        note_validation_path = os.listdir("HybridPipeGen/core/tmpdata/rl_cross_val_res/"+self.notebook_id)
        for note_validation in note_validation_path:
            note_score = np.load("HybridPipeGen/core/tmpdata/rl_cross_val_res/"+self.notebook_id+"/"+note_validation, allow_pickle=True).item()
            note_mean = np.mean(note_score['accuracy_score'])
            if note_mean != note_mean:
                continue
            max_index[self.notebook_id][note_validation] = note_mean
        L=list(max_index[self.notebook_id].items())
        L.sort(key=lambda x:x[1],reverse=True)
        max_index[self.notebook_id] = dict(L)
        with open('HybridPipeGen/core/tmpdata/max_index.json','w',encoding='utf8')as f1:
            json.dump(max_index,f1,ensure_ascii=False)
            
    def output(self,hybrid_name,save_fig =False):
        hi_score = np.load("HybridPipeGen/core/tmpdata/prenotebook_res/"+self.notebook_id+'.npy', allow_pickle=True).item()
        note_test_path = os.listdir("HybridPipeGen/core/tmpdata/merge_max_result_rl/"+self.notebook_id)
        hybrid_score = np.load("HybridPipeGen/core/tmpdata/merge_max_result_rl/"+self.notebook_id+'/'+note_test_path[0], allow_pickle=True).item()
        hybrid_index = note_test_path[0].split('.npy')[0]
        if hybrid_index!='origin':
            shutil.copyfile('HybridPipeGen/core/tmpdata/rl_test_merge_code_py/'+self.notebook_id +'/'+hybrid_index+'.py', hybrid_name)
        else:
            shutil.copyfile('copy HybridPipeGen/core/tmpdata/prenotebook_code/'+self.notebook_id +'.py', hybrid_name)
        print('notebook:',self.notebook_id)
        print('accuracy of HIPipe:',hi_score['accuracy_score'])
        print('accuracy of best HybridPipe',note_test_path[0].split('.npy')[0],':',hybrid_score['accuracy_score'])
        # if save_fig == False:
        shutil.rmtree('HybridPipeGen/core/tmpdata/')
        os.mkdir('HybridPipeGen/core/tmpdata/')
        # else:
        #     tmp_path= os.listdir('HybridPipeGen/core/tmpdata/')
        #     for tmp_file in tmp_path:
        #         if tmp_file != 'profiling_result':
        #             if os.path.isdir('HybridPipeGen/core/tmpdata/'+tmp_file):
        #                 shutil.rmtree('HybridPipeGen/core/tmpdata/'+tmp_file)
        #             else:
        #                 os.remove('HybridPipeGen/core/tmpdata/'+tmp_file)
        #         else:
        #             os.remove('HybridPipeGen/core/tmpdata/profiling_result/example')


