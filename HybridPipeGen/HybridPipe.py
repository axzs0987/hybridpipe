import json
import numpy as np
import os
from HybridPipeGen.core.preprocessing import *
from HybridPipeGen.core.NotebookGraph import profile_hipipe
from HybridPipeGen.core.merge import *
import shutil

class HybridPipe:
    def __init__(self, notebook_id):
        self.notebook_id = notebook_id

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
        merger.merging_one_notebook_rl(self.notebook_id)
        transform_one_validation_rl(self.notebook_id)
        transform_one_origin_validation_code(self.notebook_id)

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
        
        if res == True:
            pro.run_origin_test(self.notebook_id, need_try_again=2)
        else:
            add_faile += 1

    def evalaute_hai(self):
        run_one_validation_rl(self.notebook_id)
        self.max_index()
        run_one_max_hai(self.notebook_id)

    def max_index(self):
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
    def output(self,hai_name,save_fig =False):
        hi_score = np.load("HybridPipeGen/core/tmpdata/prenotebook_res/"+self.notebook_id+'.npy', allow_pickle=True).item()
        note_test_path = os.listdir("HybridPipeGen/core/tmpdata/merge_max_result_rl/"+self.notebook_id)
        hai_score = np.load("HybridPipeGen/core/tmpdata/merge_max_result_rl/"+self.notebook_id+'/'+note_test_path[0], allow_pickle=True).item()
        hai_index = note_test_path[0].split('.npy')[0]
        if hai_index!='origin':
            shutil.copyfile('HybridPipeGen/core/tmpdata/rl_test_merge_code_py/'+self.notebook_id +'/'+hai_index+'.py', hai_name)
        else:
            shutil.copyfile('copy HybridPipeGen/core/tmpdata/prenotebook_code/'+self.notebook_id +'.py', hai_name)
        print('notebook:',self.notebook_id)
        print('accuracy of HIPipe:',hi_score['accuracy_score'])
        print('accuracy of HybridPipe:',hai_score['accuracy_score'])
        if save_fig == False:
            shutil.rmtree('HybridPipeGen/core/tmpdata/')
            os.mkdir('HybridPipeGen/core/tmpdata/')
        else:
            tmp_path= os.listdir('HybridPipeGen/core/tmpdata/')
            for tmp_file in tmp_path:
                if tmp_file != 'profiling_result':
                    if os.path.isdir('HybridPipeGen/core/tmpdata/'+tmp_file):
                        shutil.rmtree('HybridPipeGen/core/tmpdata/'+tmp_file)
                    else:
                        os.remove('HybridPipeGen/core/tmpdata/'+tmp_file)
                else:
                    os.remove('HybridPipeGen/core/tmpdata/profiling_result/example')


