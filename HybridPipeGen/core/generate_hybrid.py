import json
import numpy as np
import os
from HybridPipeGen.core.preprocessing import run_one_hi,run_one_validation_rl,run_one_max_hybrid
from HybridPipeGen.core.NotebookGraph import build_one_graph
from HybridPipeGen.core.merge import merge_one_code
import shutil
def max_index(notebook_id):
    max_index ={}
    max_index[notebook_id] = {}
    note_validation_path = os.listdir("HybridPipeGen/core/tmpdata/rl_cross_val_res/"+notebook_id)
    for note_validation in note_validation_path:
        note_score = np.load("HybridPipeGen/core/tmpdata/rl_cross_val_res/"+notebook_id+"/"+note_validation, allow_pickle=True).item()
        note_mean = np.mean(note_score['accuracy_score'])
        if note_mean != note_mean:
            continue
        max_index[notebook_id][note_validation] = note_mean
    L=list(max_index[notebook_id].items())
    L.sort(key=lambda x:x[1],reverse=True)
    max_index[notebook_id] = dict(L)
    with open('HybridPipeGen/core/tmpdata/max_index.json','w',encoding='utf8')as f1:
        json.dump(max_index,f1,ensure_ascii=False)
def output(notebook_id,hybrid_name):
    hi_score = np.load("HybridPipeGen/core/tmpdata/prenotebook_res/"+notebook_id+'.npy', allow_pickle=True).item()
    note_test_path = os.listdir("HybridPipeGen/core/tmpdata/merge_max_result_rl/"+notebook_id)
    hybrid_score = np.load("HybridPipeGen/core/tmpdata/merge_max_result_rl/"+notebook_id+'/'+note_test_path[0], allow_pickle=True).item()
    hybrid_index = note_test_path[0].split('.npy')[0]
    if hybrid_index!='origin':
        shutil.copyfile('HybridPipeGen/core/tmpdata/rl_test_merge_code_py/'+notebook_id +'/'+hybrid_index+'.py', hybrid_name)
    else:
        shutil.copyfile('copy HybridPipeGen/core/tmpdata/prenotebook_code/'+notebook_id +'.py', hybrid_name)
    print('notebook_id',notebook_id)
    print('hi_score:',hi_score)
    print('hybrid_score:',hybrid_score)
    shutil.rmtree('HybridPipeGen/core/tmpdata')
def generate_one_hybrid(notebook_id,hybrid_name='hybrid.py'):
    if not os.path.exists('HybridPipeGen/core/tmpdata'):
        os.mkdir('HybridPipeGen/core/tmpdata')
    run_one_hi(notebook_id)
    build_one_graph(notebook_id)
    merge_one_code(notebook_id)
    run_one_validation_rl(notebook_id)
    max_index(notebook_id)
    run_one_max_hybrid(notebook_id)
    output(notebook_id,hybrid_name)
if __name__ == "__main__":
    generate_one_hybrid('datascientist25_gender-recognition-by-voice-using-machine-learning')

