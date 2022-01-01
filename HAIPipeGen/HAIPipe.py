import json
import numpy as np
import os
from HAIPipeGen.core.preprocessing import *
from HAIPipeGen.core.NotebookGraph import profile_hipipe
from HAIPipeGen.core.merge import *
import shutil

class HAIPipe:
    def __init__(self, notebook_id):
        self.notebook_id = notebook_id

    def combine(self):
        profile_hipipe()
        if not os.path.exists('HAIPipeGen/tmpdata/rl_test_merge_code'):
            os.mkdir('HAIPipeGen/tmpdata/rl_test_merge_code')
        if not os.path.exists('HAIPipeGen/tmpdata/rl_test_merge_code_py'):
            os.mkdir('HAIPipeGen/tmpdata/rl_test_merge_code_py')
        if not os.path.exists('HAIPipeGen/tmpdata/rl_cross_validation_code'):
            os.mkdir('HAIPipeGen/tmpdata/rl_cross_validation_code')
        if not os.path.exists('HAIPipeGen/tmpdata/rl_cross_validation_code_py'):
            os.mkdir('HAIPipeGen/tmpdata/rl_cross_validation_code_py')
        merger = Merger(4)
        merger.merging_one_notebook_rl(self.notebook_id)
        transform_one_validation_rl(self.notebook_id)
        transform_one_origin_validation_code(self.notebook_id)

    def evaluate_hi(self):
        pro = Preprocessing()
        if not os.path.exists('HAIPipeGen/tmpdata/prenotebook_code'):
            os.mkdir('HAIPipeGen/tmpdata/prenotebook_code')
        if not os.path.exists('HAIPipeGen/tmpdata/runned_notebook'):
            os.mkdir('HAIPipeGen/tmpdata/runned_notebook')
        if not os.path.exists('HAIPipeGen/tmpdata/prenotebook_res'):
            os.mkdir('HAIPipeGen/tmpdata/prenotebook_res')
        if not os.path.exists('HAIPipeGen/tmpdata/prenotebook_varibles_index'):
            os.mkdir('HAIPipeGen/tmpdata/prenotebook_varibles_index')
        res = pro.profiling_code(self.notebook_id, need_remove_model=1)
        add_faile = 0
        #print('save_code', res)
        
        if res == True:
            pro.run_origin_test(self.notebook_id, need_try_again=2)
        else:
            add_faile += 1

    def evalaute_hai(self):
        run_one_validation_rl(self.notebook_id)
        self.max_index()
        run_one_validation_rl(self.notebook_id)

    def max_index(self):
        max_index ={}
        max_index[self.notebook_id] = {}
        note_validation_path = os.listdir("HAIPipeGen/tmpdata/rl_cross_val_res/"+self.notebook_id)
        for note_validation in note_validation_path:
            note_score = np.load("HAIPipeGen/tmpdata/rl_cross_val_res/"+self.notebook_id+"/"+note_validation, allow_pickle=True).item()
            note_mean = np.mean(note_score['accuracy_score'])
            if note_mean != note_mean:
                continue
            max_index[self.notebook_id][note_validation] = note_mean
        L=list(max_index[self.notebook_id].items())
        L.sort(key=lambda x:x[1],reverse=True)
        max_index[self.notebook_id] = dict(L)
        with open('HAIPipeGen/tmpdata/max_index.json','w',encoding='utf8')as f1:
            json.dump(max_index,f1,ensure_ascii=False)
    def output(self,hai_name):
        hi_score = np.load("HAIPipeGen/tmpdata/prenotebook_res/"+self.notebook_id+'.npy', allow_pickle=True).item()
        note_test_path = os.listdir("HAIPipeGen/tmpdata/merge_max_result_rl/"+self.notebook_id)
        hai_score = np.load("HAIPipeGen/tmpdata/merge_max_result_rl/"+self.notebook_id+'/'+note_test_path[0], allow_pickle=True).item()
        hai_index = note_test_path[0].split('.npy')[0]
        if hai_index!='origin':
            shutil.copyfile('HAIPipeGen/tmpdata/rl_test_merge_code_py/'+self.notebook_id +'/'+hai_index+'.py', hai_name)
        else:
            shutil.copyfile('copy HAIPipeGen/tmpdata/prenotebook_code/'+self.notebook_id +'.py', hai_name)
        print('notebook:',self.notebook_id)
        print('accuracy of HIPipe:',hi_score['accuracy_score'])
        print('accuracy of HAIPipe:',hai_score['accuracy_score'])
        shutil.rmtree('HAIPipeGen/tmpdata/')
        os.mkdir('HAIPipeGen/tmpdata/')


