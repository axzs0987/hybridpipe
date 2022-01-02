import json
import numpy as np
def add_hi_mean():
    with open('/home/yxm/staticfg-master/clean_task_no_1_fix_label_s.json','r')as f:
        clean_task = json.load(f)
    with open('/home/chensibei/firefly_seq/jsons/train_classification_task_dic.json','r')as f:
        rl_task = json.load(f)
    for task in clean_task:
        hi = []
        for notebook_id in clean_task[task]['notebook_list']:
            hi.append(clean_task[task]['notebook_list'][notebook_id]['hi'])
            clean_task[task]['hi_mean'] = np.mean(hi)
            # if np.mean(hi)-clean_task[task]['hi_mean']!=0:
            #     print(task)
            #     print(np.mean(hi),clean_task[task]['hi_mean'])
    for task_id in rl_task:
        task_name = rl_task[task_id]['dataset']+'_'+rl_task[task_id]['model']+'_'+rl_task[task_id]['label']
        rl_task[task_id]['hi_mean'] = clean_task[task_name]['hi_mean']
    with open('/home/chensibei/firefly_seq/jsons/train_classification_task_dic.json','w')as f:
        json.dump(rl_task,f,ensure_ascii=False)
if __name__ == '__main__':
    add_hi_mean()