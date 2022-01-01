import numpy as np
import os,sys
import json
from HAIPipeGen.operation_code import *
import pprint
def find_same():
    HI = np.load("/home/yxm/statsklearn/NewDataSeq.npy", allow_pickle=True).item()
    notebook_hi = list(HI.keys())
    print(len(notebook_hi))
    res = []
    AI_path = "deepline_only_new_data/"
    AI_path_list = os.listdir(AI_path)
    all_ = 0
    for file_ in AI_path_list:
        if file_.endswith("_seq.json"):
            all_ +=1
            notebook_id = file_.split("_seq.")[0]
            if notebook_id in notebook_hi:
                res.append(notebook_id)
    # print(all_)
    all_final = []
    f = open("all_final.txt", "r")
    exist = f.readlines()

    exist_ = [x.strip("\n") for x in exist]
    for file_ in res:
        if file_ in exist_:
            all_final.append(file_)
    with open("all_same.txt","w") as f1:
        for x in all_final:
            f1.write(x+"\n")
    return all_final
score_dic = {}
def Human_score():
    global score_dic
    path = os.listdir("new_data/prenotebook_res/")
    with open('score_del_meitu.json','r') as f:
            notebooks = json.load(f)
    exist_f = open("/root/staticfg-master/all_s.txt", "r")
    exist = exist_f.readlines()
    exitst_ = [x.strip("\n") for x in exist]
    notebooks = list(set(notebooks) & set(exitst_))
    for file_ in path:
        notebook_id = file_.split(".npy")[0]
        if notebook_id in notebooks:
            note_score = np.load("new_data/prenotebook_res/"+file_, allow_pickle=True).item()
            score_dic[notebook_id] = {}
            score_dic[notebook_id]["human"]=note_score['accuracy_score']
    # print(score_dic)
    # Hu = np.load("new_data/prenotebook_res/aadharsh0428_heart-disease-major-project-kd.npy", allow_pickle=True).item()
    # print(Hu)
    # print(type(Hu))
    # print(Hu['accuracy_score'])
def deepline_score():
    global score_dic
    path = os.listdir("deepline_only_new_data/")
    with open('score_del_meitu.json','r') as f:
            notebooks = json.load(f)
    exist_f = open("/root/staticfg-master/all_s.txt", "r")
    exist = exist_f.readlines()
    exitst_ = [x.strip("\n") for x in exist]
    notebooks = list(set(notebooks) & set(exitst_))
    for file_ in path:
        if file_.endswith("_score.json"):
            notebook_id = file_.split("_score.json")[0]
            if notebook_id in notebooks:
                with open("deepline_only_new_data/"+file_,"r",encoding="utf8")as fp:
                    note_score = json.load(fp)
                    # score_dic[notebook_id] = {}
                    score_dic[notebook_id]["deepline"]=note_score['accuracy_score']
    # print(score_dic)
max_index = {}
def hai_score(note_all):
    global score_dic
    global max_index
    path = os.listdir("new_data/merge_validation_result/")
    f = open(note_all, "r")
    exist = f.readlines()
    all_note = [x.strip("\n") for x in exist]
    for file_ in path:
        notebook_id = file_
        if notebook_id in all_note:
            note_validation_path = os.listdir("new_data/merge_validation_result/"+file_)
            max_score = 0
            for note_validation in note_validation_path:
                note_score = np.load("new_data/merge_validation_result/"+file_+"/"+note_validation, allow_pickle=True).item()
                if note_validation == "origin.npy":
                    score_dic[notebook_id]["origin"] = note_score['accuracy_score']
                if note_score['accuracy_score']>max_score:
                    max_index[notebook_id] = note_validation
                    max_score = note_score['accuracy_score']
            score_dic[notebook_id]["hai"] = max_score
    # print(score_dic)
    
# test_dic = {}
max_index_new = {} 
def modify_hai(test_error):
    global score_dic
    global max_index
    global max_index_new
    f = open(test_error, "r")
    exist = f.readlines()
    note_test_new = [x.strip("\n") for x in exist]
    for notebook_id in score_dic:
        if notebook_id in note_test_new:
            # print(notebook_id)
            try:
                if max_index_new[notebook_id] == "origin.npy":
                    score_dic[notebook_id]['hai'] = score_dic[notebook_id]['human']
                else:
                    try:
                        note_score = np.load("new_data/merge_max_result_new/"+notebook_id+"/"+max_index_new[notebook_id], allow_pickle=True).item()
                        score_dic[notebook_id]['hai'] = note_score['accuracy_score']
                    except:
                        print("new_test没成功：",notebook_id)
            except:
                print("new_validation没成功：",notebook_id)
        else:
            try:
                if max_index[notebook_id] == "origin.npy":
                    score_dic[notebook_id]['hai'] = score_dic[notebook_id]['human']
                else:
                    try:
                        note_score = np.load("new_data/merge_max_result/"+notebook_id+"/"+max_index[notebook_id], allow_pickle=True).item()
                        score_dic[notebook_id]['hai'] = note_score['accuracy_score']
                    except:
                        print("old_test没成功",notebook_id)
            except:
                print("值为0：",notebook_id)
                # test_dic[notebook_id] = {}
                # note_validation_path = os.listdir("new_data/merge_validation_result/"+notebook_id)
                # for note_validation in note_validation_path:
                #     note_score = np.load("new_data/merge_validation_result/"+notebook_id+"/"+note_validation, allow_pickle=True).item()
                #     test_dic[notebook_id][note_validation] = note_score
    
                   
def max_index_800(note_all):
    global score_dic
    global max_index_new
    path = os.listdir("new_data/merge_validation_result_new/")
    f = open(note_all, "r")
    exist = f.readlines()
    all_note = [x.strip("\n") for x in exist]
    cnt = 0
    for file_ in path:
        notebook_id = file_
        if notebook_id in all_note:
            note_validation_path = os.listdir("new_data/merge_validation_result_new/"+file_)
            max_score = 0
            for note_validation in note_validation_path:
                note_score = np.load("new_data/merge_validation_result_new/"+file_+"/"+note_validation, allow_pickle=True).item()
                # if note_validation == "origin.npy":
                #     score_dic[notebook_id]["origin"] = note_score['accuracy_score']
                if note_score['accuracy_score']>max_score:
                    max_index_new[notebook_id] = note_validation
                    max_score = note_score['accuracy_score']
            # score_dic[notebook_id]["hai"] = max_score
    # print(score_dic)
def duibi():
    score_1600 = {}
    with open('need_rerun.json','r') as f:
            notebooks = json.load(f)
    exist_f = open("/root/staticfg-master/all_s.txt", "r")
    exist = exist_f.readlines()
    exitst_ = [x.strip("\n") for x in exist]
    notebooks = list(set(notebooks) & set(exitst_))
    cnt = 0
    higer = 0
    lower = 0
    equal = 0
    for notebook_id in notebooks:
        if not os.path.exists("new_data/merge_validation_result_1600/"+notebook_id):
            cnt+=1
            print("1600result没有",notebook_id)  
            continue
        path = os.listdir("new_data/merge_validation_result_1600/"+notebook_id)
        if len(path)<1:
            continue
        score_1600[notebook_id] = {}
        max_score_1600 = -1
        max_score = -1
        max_index = -1
        max_index_1600 = -1

        try:
            note_validation_path = os.listdir("new_data/merge_validation_result_new/"+notebook_id)
            for note_validation in note_validation_path:
                if os.path.exists("new_data/merge_validation_result_1600/"+notebook_id+"/"+note_validation):
                    note_score = np.load("new_data/merge_validation_result_1600/"+notebook_id+"/"+note_validation, allow_pickle=True).item()
                else:
                    note_score = np.load("new_data/merge_validation_result_new/"+notebook_id+"/"+note_validation, allow_pickle=True).item()
                if note_score['accuracy_score']>max_score_1600:
                    max_index_1600 = note_validation
                    max_score_1600 = note_score['accuracy_score']
            score_1600[notebook_id]['new'] = max_score_1600
            score_1600[notebook_id]['new_index'] =  max_index_1600
            for note_validation in note_validation_path:
                note_score = np.load("new_data/merge_validation_result_new/"+notebook_id+"/"+note_validation, allow_pickle=True).item()
                if note_score['accuracy_score']>max_score:
                    max_index = note_validation
                    max_score = note_score['accuracy_score']
            score_1600[notebook_id]['old'] = max_score
            score_1600[notebook_id]['old_index'] =  max_index
            if max_score_1600 > max_score:
                higer +=1
            elif max_score_1600 == max_score:
                equal +=1
            else:
                lower +=1
        except:
            note_validation_path = os.listdir("new_data/merge_validation_result/"+notebook_id)
            for note_validation in note_validation_path:
                if os.path.exists("new_data/merge_validation_result_1600/"+notebook_id+"/"+note_validation):
                    note_score = np.load("new_data/merge_validation_result_1600/"+notebook_id+"/"+note_validation, allow_pickle=True).item()
                else:
                    note_score = np.load("new_data/merge_validation_result/"+notebook_id+"/"+note_validation, allow_pickle=True).item()
                if note_score['accuracy_score']>max_score_1600:
                    max_index_1600 = note_validation
                    max_score_1600 = note_score['accuracy_score']
            score_1600[notebook_id]['new'] = max_score_1600
            score_1600[notebook_id]['new_index'] =  max_index_1600
            for note_validation in note_validation_path:
                note_score = np.load("new_data/merge_validation_result/"+notebook_id+"/"+note_validation, allow_pickle=True).item()
                if note_score['accuracy_score']>max_score:
                    max_index = note_validation
                    max_score = note_score['accuracy_score']
            score_1600[notebook_id]['old'] = max_score
            score_1600[notebook_id]['ols_index'] =  max_index
            if max_score_1600 > max_score:
                higer +=1
            elif max_score_1600 == max_score:
                equal +=1
            else:
                lower +=1
    old = []
    new = []
    for notebook_id in score_1600:
        old.append(score_1600[notebook_id]['old'])
        new.append(score_1600[notebook_id]['new']) 
    print(len(old),len(new))
    print('old ',np.mean(old))
    print("new ",np.mean(new))
    print("higer",higer)
    print("equal",equal)
    print("lower",lower)
    print(cnt)
    with open('score_1600_duibi.json','a',encoding='utf8')as fp:
        json.dump(score_1600,fp,ensure_ascii=False)
def max_index_1600():
    global score_dic
    global max_index
    with open('need_rerun.json','r') as f:
            notebooks = json.load(f)
    exist_f = open("/root/staticfg-master/all_s.txt", "r")
    exist = exist_f.readlines()
    exitst_ = [x.strip("\n") for x in exist]
    notebooks = list(set(notebooks) & set(exitst_))
    cnt = 0

    for notebook_id in notebooks:
        try:
            note_validation_path = os.listdir("new_data/merge_validation_result_1600/"+notebook_id)
            max_score = -1
            for note_validation in note_validation_path:
                note_score = np.load("new_data/merge_validation_result_1600/"+notebook_id+"/"+note_validation, allow_pickle=True).item()
                if note_score['accuracy_score']>max_score:
                    max_index[notebook_id] = note_validation
                    max_score = note_score['accuracy_score'] 
        except:
            cnt+=1
            print(notebook_id)   
    print(cnt)
    with open('max_index_1600.json','a',encoding='utf8')as f1:
        json.dump(max_index,f1,ensure_ascii=False)    
def modify_hai_1600():
    global score_dic
    with open('score_del_meitu.json','r') as f:
            notebooks = json.load(f)
    exist_f = open("/root/staticfg-master/all_s.txt", "r")
    exist = exist_f.readlines()
    exitst_ = [x.strip("\n") for x in exist]
    notebooks = list(set(notebooks) & set(exitst_))
    cnt_validation_origin = 0
    validation_origin = {}
    cnt_test_equal = 0
    test_equal = {}
    cnt = 0
    for notebook_id in notebooks:
        cnt+=1
        max_index = -1
        max_score = -1
        hai_score = -1
        max_root = ""
        try:
            note_validation_path = os.listdir("new_data/merge_validation_result_new/"+notebook_id)
            for note_validation in note_validation_path:
                if os.path.exists("new_data/merge_validation_result_1600/"+notebook_id+"/"+note_validation):
                    note_score = np.load("new_data/merge_validation_result_1600/"+notebook_id+"/"+note_validation, allow_pickle=True).item()
                    print("new_data/merge_validation_result_1600/"+notebook_id+"/"+note_validation+" "+str(note_score['accuracy_score']))
                    if note_score['accuracy_score']>max_score:
                        max_index = note_validation
                        max_root = "new_data/merge_max_result_1600/"
                        max_score = note_score['accuracy_score']
                
                else:
                    note_score = np.load("new_data/merge_validation_result_new/"+notebook_id+"/"+note_validation, allow_pickle=True).item()
                    print("new_data/merge_validation_result_new/"+notebook_id+"/"+note_validation+" "+str(note_score['accuracy_score']))
                    if note_score['accuracy_score']>max_score:
                        max_index = note_validation
                        max_root = "new_data/merge_max_result_new/"
                        max_score = note_score['accuracy_score']

        except:
            note_validation_path = os.listdir("new_data/merge_validation_result/"+notebook_id)
            for note_validation in note_validation_path:
                if os.path.exists("new_data/merge_validation_result_1600/"+notebook_id+"/"+note_validation):
                    
                    note_score = np.load("new_data/merge_validation_result_1600/"+notebook_id+"/"+note_validation, allow_pickle=True).item()
                    print("new_data/merge_validation_result_new_160022/"+notebook_id+"/"+note_validation+" "+str(note_score['accuracy_score']))
                    if note_score['accuracy_score']>max_score:
                        max_index = note_validation
                        max_root = "new_data/merge_max_result_1600/"
                        max_score = note_score['accuracy_score']
                else:
                    note_score = np.load("new_data/merge_validation_result/"+notebook_id+"/"+note_validation, allow_pickle=True).item()
                    print("new_data/merge_validation_result/"+notebook_id+"/"+note_validation+" "+str(note_score['accuracy_score']))
                    if note_score['accuracy_score']>max_score:
                        max_index = note_validation
                        max_root = "new_data/merge_max_result/"
                        max_score = note_score['accuracy_score']
        score_dic[notebook_id]['max_index'] = max_index
        score_dic[notebook_id]['max_root'] = max_root

        if max_index == "origin.npy":
            score_dic[notebook_id]['hai'] = score_dic[notebook_id]['human']
            validation_origin[notebook_id] = {}
            validation_origin[notebook_id]['human'] = score_dic[notebook_id]['human']
            validation_origin[notebook_id]['deepline'] = score_dic[notebook_id]['deepline']
            validation_origin[notebook_id]['hai'] = score_dic[notebook_id]['hai']
            validation_origin[notebook_id]['max_index'] = score_dic[notebook_id]['max_index']
            validation_origin[notebook_id]['max_root'] = score_dic[notebook_id]['max_root']
            cnt_validation_origin += 1
            
            
        else:
            path = os.listdir(max_root+notebook_id)
            hai_score = np.load(max_root+notebook_id+"/"+path[0], allow_pickle=True).item()
            # hai_score = np.load(max_root+notebook_id+"/"+max_index, allow_pickle=True).item()
            score_dic[notebook_id]['hai'] = hai_score['accuracy_score']
            if hai_score['accuracy_score'] == score_dic[notebook_id]['human']:
                test_equal[notebook_id] = {}
                test_equal[notebook_id]['human'] = score_dic[notebook_id]['human']
                test_equal[notebook_id]['deepline'] = score_dic[notebook_id]['deepline']
                test_equal[notebook_id]['hai'] = score_dic[notebook_id]['hai']
                test_equal[notebook_id]['max_index'] = score_dic[notebook_id]['max_index']
                test_equal[notebook_id]['max_root'] = score_dic[notebook_id]['max_root']
                test_equal[notebook_id]['max_score'] = max_score
                cnt_test_equal+=1
    # with open('analyze_validation_origin.json','a',encoding='utf8')as fp:
    #     json.dump(validation_origin,fp,ensure_ascii=False)
    # with open('analyze_test_equal.json','a',encoding='utf8')as fp:
    #     json.dump(test_equal,fp,ensure_ascii=False)
    print(cnt)
    print(cnt_validation_origin)
    print(cnt_test_equal)
            # if os.path.exists(max_root+notebook_id):
            #     path = os.listdir(max_root+notebook_id)
            #     hai_score = np.load(max_root+notebook_id+"/"+path[0], allow_pickle=True).item()
            #     score_dic[notebook_id]['hai'] = hai_score['accuracy_score']
            # else:
            #     score_dic[notebook_id]['hai'] = score_dic[notebook_id]['human']
            # hai_score = np.load(max_root+notebook_id+"/"+max_index, allow_pickle=True).item()
            # score_dic[notebook_id]['hai'] = hai_score['accuracy_score']
    
def look_score(notebook_id):
    try:
        note_validation_path = os.listdir("new_data/merge_validation_result_new/"+notebook_id)
        for note_validation in note_validation_path:
            if os.path.exists("new_data/merge_validation_result_1600/"+notebook_id+"/"+note_validation):
                note_score = np.load("new_data/merge_validation_result_1600/"+notebook_id+"/"+note_validation, allow_pickle=True).item()
                print("new_data/merge_validation_result_1600/"+notebook_id+"/"+note_validation+" "+str(note_score['accuracy_score']))
                
            else:
                note_score = np.load("new_data/merge_validation_result_new/"+notebook_id+"/"+note_validation, allow_pickle=True).item()
                print("new_data/merge_validation_result_new/"+notebook_id+"/"+note_validation+" "+str(note_score['accuracy_score']))

    except:
        note_validation_path = os.listdir("new_data/merge_validation_result/"+notebook_id)
        for note_validation in note_validation_path:
            if os.path.exists("new_data/merge_validation_result_1600/"+notebook_id+"/"+note_validation):
                note_score = np.load("new_data/merge_validation_result_1600/"+notebook_id+"/"+note_validation, allow_pickle=True).item()
                print("new_data/merge_validation_result_1600/"+notebook_id+"/"+note_validation+" "+str(note_score['accuracy_score']))
                
            else:
                note_score = np.load("new_data/merge_validation_result/"+notebook_id+"/"+note_validation, allow_pickle=True).item()
                print("new_data/merge_validation_result/"+notebook_id+"/"+note_validation+" "+str(note_score['accuracy_score']))
    print("----------------------------------")
    try:
        path = os.listdir("new_data/merge_max_result_new/"+notebook_id)
        note_score = np.load("new_data/merge_max_result_new/"+notebook_id+"/"+path[0], allow_pickle=True).item()
        print("new_data/merge_max_result_new/"+notebook_id+"/"+path[0]+ str(note_score['accuracy_score']))
    except:
        print("没new")

    try:
        path = os.listdir("new_data/merge_max_result/"+notebook_id)
        note_score = np.load("new_data/merge_max_result/"+notebook_id+"/"+path[0], allow_pickle=True).item()
        print("new_data/merge_max_result/"+notebook_id+"/"+path[0]+ str(note_score['accuracy_score']))
    except:
        print("没test")
    try:
        path = os.listdir("new_data/merge_max_result_1600/"+notebook_id)
        note_score = np.load("new_data/merge_max_result_1600/"+notebook_id+"/"+path[0], allow_pickle=True).item()
        print("new_data/merge_max_result_1600/"+notebook_id+"/"+path[0]+ str(note_score['accuracy_score']))
    except:
        print("没1600")
def look_1600():
 
    path_merge = os.listdir("merge_code_new_data_1600")
    path = os.listdir("new_data/prenotebook_graph")

    res_all =[]
    for notebook_id in path_merge:
        if notebook_id+".pkl" in path:
            res_all.append(notebook_id)
    path_result = os.listdir("new_data/merge_validation_result_1600")
    res = []
    for notebook_id in path_result:
        note_path = os.listdir("new_data/merge_validation_result_1600/" + notebook_id)
        if len(note_path)>0:
            res.append(notebook_id)
    with open('score_del_meitu.json','r',encoding='utf8')as fp:
        score = json.load(fp)
    print("merge成功 ",len(res_all))
    print("validation成功 ",len(res))
    same = 0
    higher = 0
    lower = 0
    for notebook_id in res:
        if score[notebook_id]['hai']==score[notebook_id]['human']:
            same +=1
        if score[notebook_id]['hai']>score[notebook_id]['human']:
            higher +=1
        if score[notebook_id]['hai']<score[notebook_id]['human']:
            lower +=1   
    print("same",same)
    print("higher",higher)
    print("lower",lower)
    
def stat_same():
    global score_dic
    with open('dedup_score_del_meitu_1264.json','r') as f:
        score_dic = json.load(f)
    exist_f = open("/root/staticfg-master/all_s.txt", "r")
    exist = exist_f.readlines()
    exitst_ = [x.strip("\n") for x in exist]
    notebooks = list(set(score_dic) & set(exitst_))
    cnt_validation_origin = 0
    validation_origin = {}
    cnt_test_equal = 0
    cnt_validation_same = 0
    cnt_validation_low = 0
    cnt_other_validation = 0
    cnt_validation_same_all = 0
    cnt_ai_higher = 0
    test_equal = {}
    cnt = 0
    for notebook_id in notebooks:
        cnt+=1
        max_index = -1
        max_score = -1
        hai_score = -1
        max_root = ""
        try:
            note_validation_path = os.listdir("new_data/merge_validation_result_new/"+notebook_id)
            for note_validation in note_validation_path:
                if os.path.exists("new_data/merge_validation_result_1600/"+notebook_id+"/"+note_validation):
                    note_score = np.load("new_data/merge_validation_result_1600/"+notebook_id+"/"+note_validation, allow_pickle=True).item()
                    print("new_data/merge_validation_result_1600/"+notebook_id+"/"+note_validation+" "+str(note_score['accuracy_score']))
                    if note_score['accuracy_score']>max_score:
                        max_index = note_validation
                        max_root = "new_data/merge_max_result_1600/"
                        max_score = note_score['accuracy_score']
                
                else:
                    note_score = np.load("new_data/merge_validation_result_new/"+notebook_id+"/"+note_validation, allow_pickle=True).item()
                    print("new_data/merge_validation_result_new/"+notebook_id+"/"+note_validation+" "+str(note_score['accuracy_score']))
                    if note_validation == "origin.npy":
                        score_dic[notebook_id]['origin'] = note_score['accuracy_score']
                    elif note_score['accuracy_score']>max_score:
                        max_index = note_validation
                        max_root = "new_data/merge_max_result_new/"
                        max_score = note_score['accuracy_score']

        except:
            note_validation_path = os.listdir("new_data/merge_validation_result/"+notebook_id)
            for note_validation in note_validation_path:
                if os.path.exists("new_data/merge_validation_result_1600/"+notebook_id+"/"+note_validation):
                    
                    note_score = np.load("new_data/merge_validation_result_1600/"+notebook_id+"/"+note_validation, allow_pickle=True).item()
                    print("new_data/merge_validation_result_new_160022/"+notebook_id+"/"+note_validation+" "+str(note_score['accuracy_score']))
                    if note_score['accuracy_score']>max_score:
                        max_index = note_validation
                        max_root = "new_data/merge_max_result_1600/"
                        max_score = note_score['accuracy_score']
                else:
                    note_score = np.load("new_data/merge_validation_result/"+notebook_id+"/"+note_validation, allow_pickle=True).item()
                    print("new_data/merge_validation_result/"+notebook_id+"/"+note_validation+" "+str(note_score['accuracy_score']))
                    if note_validation == "origin.npy":
                        score_dic[notebook_id]['origin'] = note_score['accuracy_score']
                    elif note_score['accuracy_score']>max_score:
                        max_index = note_validation
                        max_root = "new_data/merge_max_result/"
                        max_score = note_score['accuracy_score']
        score_dic[notebook_id]['max_index'] = max_index
        score_dic[notebook_id]['max_root'] = max_root
        if score_dic[notebook_id]['hi'] == score_dic[notebook_id]['hai']:
            if max_score <= score_dic[notebook_id]['origin']:
                cnt_validation_same +=1
            # elif max_score < score_dic[notebook_id]['origin']:
            #     cnt_validation_low +=1
        elif max_score <= score_dic[notebook_id]['origin'] :
            cnt_other_validation +=1
        if max_score <= score_dic[notebook_id]['origin']:
            cnt_validation_same_all +=1
            if score_dic[notebook_id]['ai-deepline'] > score_dic[notebook_id]['hi']:
                cnt_ai_higher +=1

    print(cnt)
    print(cnt_validation_same)
    print(cnt_other_validation)
    print(cnt_validation_same_all)
    print(cnt_ai_higher)

    
def look_task_mean():
    with open('clean_task.json', 'r') as f:
        task_add_base = json.load(f)  
    hai_mean = 0 
    hi_mean = 0
    all_ = 0
    deepline_mean = 0
    ai_seq = 0   
    base_mean = 0
    autosklearn_mean = 0
    hi_std_mean = 0
    for task in task_add_base:
        print(task)
        # if task_add_base[task]['model'] != 'RandomForestClassifier':
        # if task_add_base[task]['model'] != 'DecisionTreeClassifier':
        # if task_add_base[task]['model'] != 'LogisticRegression':
        # if task_add_base[task]['model'] != 'KNeighborsClassifier':
        # if task_add_base[task]['model'] != 'SVC':
            # continue
        hi_mean += task_add_base[task]['hi_mean']
        deepline_mean += task_add_base[task]['deepline_mean']
        ai_seq += task_add_base[task]['ai_seq_mean']
        autosklearn_mean += task_add_base[task]['autosklearn']
        hai_mean += task_add_base[task]['hai_mean']
        base_mean += task_add_base[task]['base']
        if len( task_add_base[task]['notebook_list']) > 1:
            print(len( task_add_base[task]['notebook_list']))
            print('hi_std', task_add_base[task]['hi_std'])
            print('hai_std', task_add_base[task]['hai_std'])
            print('ai_seq', task_add_base[task]['ai_seq_std'])

        # hi_std_mean += task_add_base[task]['hi_std']
        # hai_std_mean += task_add_base[task]['hai_std']
        # deepline_std_mean += task_add_base[task]['deepline_std']
        all_ += 1
    print('hi_mean', hi_mean/all_)
    print('hai_mean', hai_mean/all_)
    print('deepline_mean', deepline_mean/all_)
    print('ai_seq', ai_seq/all_)
    print('autosklearn_mean', autosklearn_mean/all_)
    # print('base_mean', base_mean/all_)
    # print('base_mean', base_mean/all_)
    # print('base_mean', base_mean/all_)

    print('all_', all_)


def look_task():
    with open('task_add_validation.json', 'r') as f:
        task_add_base = json.load(f)
    clean_data = {}
    fail = 0
    fail_task = 0
    dataset = []
    for i in task_add_base:
        hi_mean = 0
        deepline_mean = 0
        ai_seq_mean = 0
        num = 0
        hai_mean = 0
        hi_list = []
        ai_seq_list = []
        deepline_list = []
        hai_list = []
        print("XXXXXXXXX")
        print(i)
        if 'base' not in task_add_base[i]:
            continue
        print(task_add_base[i]['base'])
        new_notebooklist = {}
        autosklearn = 0
        for j in task_add_base[i]['notebook_list']:
            print(j)
            item = task_add_base[i]['notebook_list'][j]
            
            if 'add_seq' not in item:
                fail += 1
                continue
            hi_mean += item['hi']
            hai_mean += item['hai']
            ai_seq_mean += item['add_seq']
            deepline_mean += item['ai-deepline']
            num += 1
            new_notebooklist[j] = item
            autosklearn = item['ai-autosklearn']
            hi_list.append(item['hi'])
            ai_seq_list.append(item['add_seq'])
            deepline_list.append(item['ai-deepline'])
            hai_list.append(item['hai'])
        if len(new_notebooklist) == 0:
            continue
        
        task_add_base[i]['notebook_list'] = new_notebooklist
        task_add_base[i]['hai_mean'] = hai_mean/num
        task_add_base[i]['hi_mean'] = hi_mean/num
        task_add_base[i]['deepline_mean'] = deepline_mean/num
        task_add_base[i]['ai_seq_mean'] = ai_seq_mean/num

        task_add_base[i]['hai_std'] = np.array(hai_list).std()
        task_add_base[i]['hi_std'] = np.array(hi_list).std()
        task_add_base[i]['deepline_std'] = np.array(deepline_list).std()
        task_add_base[i]['ai_seq_std'] = np.array(ai_seq_list).std()
        split_ = i.split('_')
        dataset_name = ''
        for char in split_[0:-2]:
            dataset_name += char
        if dataset_name  not in dataset:
            dataset.append(dataset_name)
        data_ind = 0
        for ind,ds in enumerate(dataset):
            if dataset_name == ds:
                data_ind = ind
                break
        task_add_base[i]['data_index'] = data_ind
        task_add_base[i]['dataset'] = dataset_name
        task_add_base[i]['model'] = split_[-2]
        task_add_base[i]['label'] = split_[-1]
        task_add_base[i]['autosklearn'] = autosklearn
        clean_data[i] = task_add_base[i]
    with open('clean_task.json', 'w') as f:
        json.dump(clean_data,f)
    print(task_add_base)
    print(fail)

def look_svc_not_imp():        
    with open('clean_task.json','r') as f:
        clean_task = json.load(f)
    for task in clean_task:
        if clean_task[task]['model'] != 'SVC':
            continue
        for notebook_id in clean_task[task]['notebook_list']:
            notebook_obj = clean_task[task]['notebook_list'][notebook_id]
            if notebook_obj['hai'] <= notebook_obj['hi']:
                print('#########')
                print(task)
                print(notebook_obj)

def fix_dataset():
    with open('clean_task_no_1_fix_label.json', 'r') as f:
        clean_task = json.load(f)
    for task in clean_task:
        dataset_name = ''
        split_ = task.split('_')
        for index,char in enumerate(split_[0:-2]):
            dataset_name += char
            if index != len(split_[0:-2])-1:
                dataset_name += '_'
        clean_task[task]['dataset'] = dataset_name
    with open('clean_task_no_1_fix_label.json', 'w') as f:
        json.dump(clean_task, f)

def add_time():
    with open('clean_task.json','r') as f:
        clean_task = json.load(f)
    with open('new_data/validation_running_time_1.json', 'r') as f:
        run_time1 = json.load(f)
    with open('new_data/validation_running_time.json', 'r') as f:
        run_time = json.load(f)
    with open('validation_running_time_0.json', 'r') as f:
        run_time2 = json.load(f)
    with open('validation_running_time_-1.json', 'r') as f:
        run_time3 = json.load(f)
    with open('validation_running_time_1.json', 'r') as f:
        run_time4 = json.load(f)
    with open('validation_running_time_2.json', 'r') as f:
        run_time5 = json.load(f)
    for item in run_time2:
        if item not in run_time1:
            run_time1[item] = run_time2[item]
    for item in run_time3:
        if item not in run_time1:
            run_time1[item] = run_time3[item]
    for item in run_time4:
        if item not in run_time1:
            run_time1[item] = run_time4[item]
    for item in run_time5:
        if item not in run_time1:
            run_time1[item] = run_time5[item]
    for item in run_time:
        if item not in run_time1:
            run_time1[item] = run_time[item]
    for task in clean_task:
        for notebook_id in clean_task[task]['notebook_list']:
            if notebook_id in run_time1:
                sum_time = 0
                for seq_item in run_time1[notebook_id]:
                    sum_time = run_time1[notebook_id][seq_item]
                clean_task[task]['notebook_list'][notebook_id]['time'] = sum_time
    with open('clean_task.json','w') as f:
        json.dump(clean_task, f)

def add_seq_in_json():
    with open('clean_task.json','r') as f:
        clean_task = json.load(f)
    for task in clean_task:
        for notebook_id in clean_task[task]['notebook_list']:
            if os.path.exists("deepline_only_new_data"+ "/" + str(notebook_id) + "_seq.json"):
                filepath = "deepline_only_new_data"+ "/" + str(notebook_id) + "_seq.json"
            else:
                filepath = "deepline_only_new_1"+ "/" + str(notebook_id) + "_seq.json"
            operations = []
            with open(filepath, 'r') as f:
                operations = json.load(f)
            can_add_operaiton = []
            res = []
            # print('operations', operations)
            for ope_type in OperationType:
                can_add_operaiton += OperationType[ope_type]
            for item in operations:
                if item != -1 and item != 'BLANK' and item != 'FINISH' and item in can_add_operaiton and item not in res:
                    res.append(item)
            clean_task[task]['notebook_list'][notebook_id]['seq'] = res
    with open('clean_task.json','w') as f:
        json.dump(clean_task,f)

def add_related_score():
    with open('clean_task.json','r') as f:
        clean_task = json.load(f)
    new_res = {}
    for task in clean_task:
        all_related_imp = 0
        deepline_related_imp = 0
        ai_seq_related_imp = 0
        autosklearn_related_imp = 0
        # print(task)
        if clean_task[task]['hi_mean'] == 0:
            continue
        new_nb = {}
        for notebook_id in clean_task[task]['notebook_list']:
            notebook = clean_task[task]['notebook_list'][notebook_id]
            clean_task[task]['notebook_list'][notebook_id]['hi_related'] = (notebook['hi'] - clean_task[task]['hi_mean']) / clean_task[task]['hi_mean']
            clean_task[task]['notebook_list'][notebook_id]['hai_related'] = (notebook['hai'] - clean_task[task]['hi_mean']) / clean_task[task]['hi_mean']
            clean_task[task]['notebook_list'][notebook_id]['deepline_related'] = (notebook['ai-deepline'] - clean_task[task]['hi_mean']) / clean_task[task]['hi_mean']
            clean_task[task]['notebook_list'][notebook_id]['dseq_related'] = (notebook['add_seq'] - clean_task[task]['hi_mean']) / clean_task[task]['hi_mean']
            clean_task[task]['notebook_list'][notebook_id]['autosklearn_related'] = (notebook['ai-autosklearn'] - clean_task[task]['hi_mean']) / clean_task[task]['hi_mean']
            all_related_imp += clean_task[task]['notebook_list'][notebook_id]['hai_related']
            deepline_related_imp += clean_task[task]['notebook_list'][notebook_id]['deepline_related']
            ai_seq_related_imp += clean_task[task]['notebook_list'][notebook_id]['dseq_related']
            autosklearn_related_imp += clean_task[task]['notebook_list'][notebook_id]['autosklearn_related']
            if notebook['hi'] != 1:
                new_nb[notebook_id] = notebook.copy()
        print(clean_task[task])
        all_related_imp /= len(clean_task[task]['notebook_list'])
        deepline_related_imp /= len(clean_task[task]['notebook_list'])
        ai_seq_related_imp /= len(clean_task[task]['notebook_list'])
        autosklearn_related_imp /= len(clean_task[task]['notebook_list'])
        
        clean_task[task]['related_hai_improve'] = all_related_imp
        clean_task[task]['related_deepline_improve'] = deepline_related_imp
        clean_task[task]['related_ai_seq_improve'] = ai_seq_related_imp
        clean_task[task]['related_autosklearn_improve'] = autosklearn_related_imp
        
        new_res[task] = clean_task[task].copy()
        new_res[task]['notebook_list'] = new_nb
    with open('clean_task.json','w') as f:
        json.dump(clean_task, f)
    with open('clean_task_no_1.json','w') as f:
        json.dump(new_res, f)

def generate_clean_task_planb():
    with open('clean_task_add_planB.json','r') as f:
        clean_task = json.load(f)
    res = {}
    for task in clean_task:
        new_notebook_list = {}
        for notebook_id in clean_task[task]['notebook_list']:
            notebook = clean_task[task]['notebook_list'][notebook_id]
            if 'planB' not in notebook:
                continue
            new_notebook_list[notebook_id] = notebook
        new_task = clean_task[task].copy()
        new_task['notebook_list'] = new_notebook_list
        res[task] = new_task
    with open('clean_task_planB.json','w') as f:
        json.dump(res, f)

def add_abs_score():
    with open('clean_task_rl_200.json','r') as f:
        clean_task = json.load(f)
    clean_task_new = {}
    for task in clean_task:
        all_related_imp = 0
        all_related_imp_deepline = 0
        all_related_imp_autosklearn = 0
        all_related_imp_planb = 0
        all_related_imp_rl = 0

        # print(task)
        need_continue = False
        hi_mean = 0

        notebook_list = {}
        if len(clean_task[task]['notebook_list']) == 0:
            continue
        for notebook_id in clean_task[task]['notebook_list']:
            hi_mean += clean_task[task]['notebook_list'][notebook_id]['hi']
        clean_task[task]['hi_mean']= hi_mean/len(clean_task[task]['notebook_list'])
        for notebook_id in clean_task[task]['notebook_list']:    
            notebook = clean_task[task]['notebook_list'][notebook_id]
            notebook_list[notebook_id] = notebook
            clean_task[task]['notebook_list'][notebook_id]['hi_abs'] = (notebook['hi'] - clean_task[task]['hi_mean'])
            clean_task[task]['notebook_list'][notebook_id]['hai_abs'] = (notebook['hai'] - clean_task[task]['hi_mean'])
            clean_task[task]['notebook_list'][notebook_id]['deepline_abs'] = (notebook['ai-deepline'] - clean_task[task]['hi_mean'])
            clean_task[task]['notebook_list'][notebook_id]['autosklearn_abs'] = (notebook['ai-autosklearn'] - clean_task[task]['hi_mean'])
            clean_task[task]['notebook_list'][notebook_id]['planB_abs'] = (notebook['planB'] - clean_task[task]['hi_mean'])
            clean_task[task]['notebook_list'][notebook_id]['rl_abs'] = (notebook['rl'] - clean_task[task]['hi_mean'])
            all_related_imp += clean_task[task]['notebook_list'][notebook_id]['hai_abs']
            all_related_imp_planb += clean_task[task]['notebook_list'][notebook_id]['planB_abs']
            all_related_imp_deepline += clean_task[task]['notebook_list'][notebook_id]['deepline_abs']
            all_related_imp_autosklearn += clean_task[task]['notebook_list'][notebook_id]['autosklearn_abs']
            all_related_imp_rl += clean_task[task]['notebook_list'][notebook_id]['rl_abs']
            
       

        clean_task[task]['abs_planB_improve'] = all_related_imp_planb/len(clean_task[task]['notebook_list'])
        clean_task[task]['abs_deepline_56_improve'] = all_related_imp_deepline/len(clean_task[task]['notebook_list'])
        clean_task[task]['abs_autosklearn_improve'] = all_related_imp_autosklearn/len(clean_task[task]['notebook_list'])
        clean_task[task]['abs_rl_improve'] = all_related_imp_rl/len(clean_task[task]['notebook_list'])
        
        clean_task[task]['abs_planB_seq_improve'] = clean_task[task]['planB_ai_seq'] - clean_task[task]['hi_mean']
        clean_task[task]['abs_rl_seq_improve'] = clean_task[task]['rl_ai_seq'] - clean_task[task]['hi_mean']
        clean_task[task]['abs_deepline_same_improve'] = clean_task[task]['deepline'] - clean_task[task]['hi_mean']

        clean_task[task]['abs_planB_imp_add_himean'] = all_related_imp_planb/len(clean_task[task]['notebook_list']) + clean_task[task]['hi_mean']
        clean_task[task]['abs_deepline_56_imp_add_himean'] = all_related_imp_deepline/len(clean_task[task]['notebook_list']) + clean_task[task]['hi_mean']
        clean_task[task]['abs_autosklearn_imp_add_himean'] = all_related_imp_autosklearn/len(clean_task[task]['notebook_list']) + clean_task[task]['hi_mean']
        clean_task[task]['abs_rl_imp_add_himean'] = all_related_imp_rl/len(clean_task[task]['notebook_list']) + clean_task[task]['hi_mean']
        
        clean_task_new[task] = clean_task[task].copy()
        
        clean_task_new[task]['notebook_list'] = notebook_list
    with open('clean_task_rl_200.json','w') as f:
        json.dump(clean_task_new, f)


def look_rf():
    improve = 0
    not_improve = 0
    equal = 0
    all_ = 0
    with open('clean_task_no_1.json','r') as f:
        clean_task = json.load(f)

    dataset = []
    bad_dataset = []
    new_res = {}
    for task in clean_task:

        all_related_imp = 0
        # if clean_task[task]['model'] == 'DecisionTreeClassifier':
        if (clean_task[task]['model'] == 'RandomForestClassifier' or clean_task[task]['model'] == 'SVC') and clean_task[task]['hi_mean'] < 0.7:
        # if clean_task[task]['model'] == 'SVC':
        # if clean_task[task]['model'] == 'KNeighborsClassifier':
        # if clean_task[task]['model'] == 'LogisticRegression':
            # pprint.pprint(clean_task[task])
            all_not_imp = True
            dataset.append(clean_task[task]['dataset'])
            for notebook in clean_task[task]['notebook_list']:
                if clean_task[task]['notebook_list'][notebook]['hai'] > clean_task[task]['notebook_list'][notebook]['hi']:
                    improve+=1
                elif clean_task[task]['notebook_list'][notebook]['hai'] == clean_task[task]['notebook_list'][notebook]['hi'] and clean_task[task]['notebook_list'][notebook]['hi'] != 1:
                    equal += 1
                elif clean_task[task]['notebook_list'][notebook]['hai'] < clean_task[task]['notebook_list'][notebook]['hi']:
                    not_improve += 1
                all_+=1
        # for notebook in clean_task[task]['notebook_list']:

            # if clean_task[task]['notebook_list'][notebook]['max_validation_score'] > clean_task[task]['notebook_list'][notebook]['origin']:
                # all_not_imp = False
                # break
        # if all_not_imp == False:
        #     # continue
            # bad_dataset.append(clean_task[task]['dataset'])
            # pprint.pprint(clean_task[task])
            new_res[task] = clean_task[task]
    print(len(bad_dataset))
    print(len(dataset))
    print(len(set(bad_dataset)))
    print(len(set(dataset)))

    all_count = improve + equal + not_improve
    print(improve/all_count)
    print(equal/all_count)
    print(not_improve/all_count)
    print(all_)
    with open('svcrf_lower_0.7.json','w') as f:
        json.dump(new_res, f)
def look_val():
    files = os.listdir('new_data/merge_validation_result_new/ilysainath_titanic-ml2')
    for item in files:
        score = np.load('new_data/merge_validation_result_new/ilysainath_titanic-ml2/' + item,allow_pickle=True).item()
        print(item,score)

def look_svcrf():
    with open('svcrf_lower_0.7.json','r') as f:
        res = json.load(f)
    svc_task_num = 0
    rf_task_num = 0
    svc_nb_num = 0
    rf_nb_num = 0
    for task in res:
        if res[task]['model_type'] == 'RandomForestClassifier':
            rf_task_num += 1
            rf_nb_num += len(res[task]['notebook_list'])
        if res[task]['model_type'] == 'SVC':
            svc_task_num += 1
            svc_nb_num += len(res[task]['notebook_list'])
    print('rf_task_num', rf_task_num)
    print('rf_nb_num', rf_nb_num)
    print('svc_task_num', svc_task_num)
    print('svc_nb_num', svc_nb_num)

def last_end():
    with open('clean_task_no_1.json', 'r') as f:
        clean_task = json.load(f)

    related_hai_improve = 0
    related_deepline_improve = 0
    related_ai_seq_improve = 0
    related_autosklearn_improve = 0
    all_ =0
    for task in clean_task:
        # if clean_task[task]['model_type'] == 'RandomForestClassifier' and clean_task[task]['hi_mean'] < 0.7:
        # if clean_task[task]['model_type'] == 'RandomForestClassifier' and clean_task[task]['hi_mean'] >= 0.7 and clean_task[task]['hi_mean'] < 0.8:
        # if clean_task[task]['model_type'] == 'RandomForestClassifier' and clean_task[task]['hi_mean'] >= 0.8 and clean_task[task]['hi_mean'] < 0.9:
        # if clean_task[task]['model_type'] == 'RandomForestClassifier' and clean_task[task]['hi_mean'] >= 0.9 and clean_task[task]['hi_mean'] < 1:
        #  if clean_task[task]['model_type'] == 'DecisionTreeClassifier' and clean_task[task]['hi_mean'] < 0.7:
        # if clean_task[task]['model_type'] == 'DecisionTreeClassifier' and clean_task[task]['hi_mean'] >= 0.7 and clean_task[task]['hi_mean'] < 0.8:
        # if clean_task[task]['model_type'] == 'DecisionTreeClassifier' and clean_task[task]['hi_mean'] >= 0.8 and clean_task[task]['hi_mean'] < 0.9:
        # if clean_task[task]['model_type'] == 'DecisionTreeClassifier' and clean_task[task]['hi_mean'] >= 0.9 and clean_task[task]['hi_mean'] < 1:
        #  if clean_task[task]['model_type'] == 'KNeighborsClassifier' and clean_task[task]['hi_mean'] < 0.7:
        # if clean_task[task]['model_type'] == 'KNeighborsClassifier' and clean_task[task]['hi_mean'] >= 0.7 and clean_task[task]['hi_mean'] < 0.8:
        # if clean_task[task]['model_type'] == 'KNeighborsClassifier' and clean_task[task]['hi_mean'] >= 0.8 and clean_task[task]['hi_mean'] < 0.9:
        # if clean_task[task]['model_type'] == 'KNeighborsClassifier' and clean_task[task]['hi_mean'] >= 0.9 and clean_task[task]['hi_mean'] < 1:
        # if clean_task[task]['model_type'] == 'LogisticRegression' and clean_task[task]['hi_mean'] < 0.7:
        # if clean_task[task]['model_type'] == 'LogisticRegression' and clean_task[task]['hi_mean'] >= 0.7 and clean_task[task]['hi_mean'] < 0.8:
        # if clean_task[task]['model_type'] == 'LogisticRegression' and clean_task[task]['hi_mean'] >= 0.8 and clean_task[task]['hi_mean'] < 0.9:
        # if clean_task[task]['model_type'] == 'LogisticRegression' and clean_task[task]['hi_mean'] >= 0.9 and clean_task[task]['hi_mean'] < 1:
        # if clean_task[task]['model_type'] == 'SVC' and clean_task[task]['hi_mean'] < 0.7:
        # if clean_task[task]['model_type'] == 'SVC' and clean_task[task]['hi_mean'] >= 0.7 and clean_task[task]['hi_mean'] < 0.8:
        # if clean_task[task]['model_type'] == 'SVC' and clean_task[task]['hi_mean'] >= 0.8 and clean_task[task]['hi_mean'] < 0.9:
        if clean_task[task]['model_type'] == 'SVC' and clean_task[task]['hi_mean'] >= 0.9 and clean_task[task]['hi_mean'] < 1:
            related_hai_improve += clean_task[task]['related_hai_improve']
            related_deepline_improve += clean_task[task]['related_deepline_improve']
            related_ai_seq_improve += clean_task[task]['related_ai_seq_improve']
            related_autosklearn_improve += clean_task[task]['related_autosklearn_improve']
            all_ += 1
    print('related_hai_improve',related_hai_improve/all_)
    print('related_deepline_improve', related_deepline_improve/all_)
    print('related_ai_seq_improve', related_ai_seq_improve/all_)
    print('related_autosklearn_improve', related_autosklearn_improve/all_)
            # break
def compare_firefly():
    with open('test_reward_24000_56000.json', 'r') as f:
        test_reward = json.load(f)
    with open('clean_task_add_planB.json', 'r') as f:
        clean_task = json.load(f)
    with open('../firefly_seq/jsons/classification_task_dic.json', 'r') as f:
        classification_task_dic = json.load(f)
    with open('planB_cross.json','r') as f:
        planB_test = json.load(f)
    print('planB_test', len(planB_test))
    firefly_better = 0
    firefly_equal = 0
    firefly_bad = 0

    firefly = 0
    base = 0
    planB = 0
    all_ = 0
    planB_ai_seq =0

    ope_num = {}
    planB_ope_num = {}
    for task_id in test_reward:
        task_name = classification_task_dic[task_id]['dataset']+'_'+classification_task_dic[task_id]['model']+"_"+classification_task_dic[task_id]['label'] 
        reward = test_reward[task_id]['reward']['56000']
        if reward == -1 or 'planB_ai_seq' not in clean_task[task_name]:
            continue
        if task_name not in planB_test:
            continue
        print(clean_task[task_name]['planB_ai_seq'])
        best_seq_id = list(planB_test[task_name].keys())[0]
        operations = get_all_aiseq()[int(best_seq_id)]
        # print()
        # print('deepline_mean', clean_task[task_name]['deepline_mean'])
        if reward > clean_task[task_name]['base']:
            firefly_better += 1
            print("XXXXXX")
            print('firefly', reward)
            print('base', clean_task[task_name]['base'])
            print(test_reward[task_id]['seq']['56000'])
        elif reward == clean_task[task_name]['base']:
            firefly_equal += 1
        else:
            firefly_bad += 1
        # if clean_task[task_name]['model_type'] == 'LogisticRegression':
        all_ += 1
        firefly += test_reward[task_id]['reward']['56000']
        base += clean_task[task_name]['base']
        planB_ai_seq += clean_task[task_name]['planB_ai_seq']
        for ope in test_reward[task_id]['seq']['56000']:
            if ope not in ope_num:
                ope_num[ope] = 0
            ope_num[ope] += 1
        for ope in operations:
            if ope not in planB_ope_num:
                planB_ope_num[ope] = 0
            planB_ope_num[ope] += 1


    print('firefly_better', firefly_better)
    print('firefly_equal', firefly_equal)
    print('firefly_bad', firefly_bad)
    print('firefly', firefly/all_)
    print('base', base/all_)
    print('planB_ai_seq', planB_ai_seq/all_)
    pprint.pprint(ope_num)
    pprint.pprint(planB_ope_num)
def get_all_aiseq():
    seqs = [
        ['Scaler', 'FeatureEngine_simple', 'FeatureSelection'],
        ['FeatureEngine_simple', 'FeatureSelection', 'Scaler'],
        ['FeatureSelection', 'Scaler', 'FeatureEngine_simple'],
        ['FeatureSelection', 'FeatureEngine_simple', 'Scaler'],
    ]
    ai_sequences = []
    for seq in seqs:
        temp_list = []
        for index, lop in enumerate(seq):
            for pop in OperationType[lop]:
                if index == 0:
                    temp_list.append([pop])
                else:
                    for index1,temp_seq in enumerate(temp_list1):
                        temp_seq1 = temp_seq.copy()
                        temp_seq1.append(pop)
                        temp_list.append(temp_seq1)
            temp_list1 = temp_list.copy()
        ai_sequences += temp_list
    # pprint.pprint(ai_sequences)
    ai_sequences_new = [i for i in ai_sequences if len(i) ==3]
    # print(len(ai_sequences_new))
    return ai_sequences_new



def look_planB_train(spec_task=None, prefix=None):
    ai_sequences = get_all_aiseq()
    with open('clean_task_no_1_fix_label_s.json', 'r') as f:
        clean_data = json.load(f)
    with open('planB_train_sort.json', 'r') as f:
        planB_train_score = json.load(f)
    seq_mean = {}
    # print(planB_train_score)
    seq_sort = {}

    seq_improve_ratio = {}
    seq_num = {}
    for task in planB_train_score:
        if spec_task is not None:
            if task in spec_task:
                continue
        sorted_score = sorted(planB_train_score[task].items(), key=lambda x: x[1], reverse=True)
        # print('sorted_score', sorted_score)
        for seq_id in planB_train_score[task]:
            if seq_id == 'mean':
                continue
            if seq_id not in seq_mean:
                seq_mean[seq_id] = []
                seq_sort[seq_id] = []
                seq_improve_ratio[seq_id] = 0
                seq_num[seq_id] = 0
            if planB_train_score[task][seq_id] > clean_data[task]['base']:
                seq_improve_ratio[seq_id] += 1
            seq_num[seq_id] += 1
            for index, item in enumerate(sorted_score):
                if item[0] == seq_id:
                    seq_sort[seq_id].append(index)
            
            seq_mean[seq_id].append(planB_train_score[task][seq_id])
    # print(seq_mean)
    seq_weight_score = {}
    for seq_id in seq_mean:
        
        seq_mean[seq_id] = np.array(seq_mean[seq_id]).mean()
        seq_sort[seq_id] = np.array(seq_sort[seq_id]).mean()
        seq_improve_ratio[seq_id] = seq_improve_ratio[seq_id]/seq_num[seq_id]
        seq_weight_score[seq_id] = (seq_mean[seq_id] + seq_improve_ratio[seq_id])/2
    
    
    seq_mean = sorted(seq_mean.items(), key = lambda x: x[1], reverse=True)
    # pprint.pprint(seq_mean)
    seq_sort = sorted(seq_sort.items(), key = lambda x: x[1])

    seq_improve_ratio = sorted(seq_improve_ratio.items(), key = lambda x: x[1], reverse=True)
    seq_weight_score = sorted(seq_weight_score.items(), key = lambda x: x[1], reverse=True)
    # pprint.pprint(seq_sort)
    str_ = ''
    res = {}
    for index,item in enumerate(seq_weight_score):
        for item0 in seq_improve_ratio:
            if item0[0] == item[0]:
                mean_improve_ratio = item0[1]
        for item1 in seq_sort:
            if item1[0] == item[0]:
                mean_sort = item1[1]
        for item2 in seq_mean:
            if item2[0] == item[0]:
                mean_score = item2[1]
        # print(item[0], item[1], mean_sort, mean_improve_ratio, ai_sequences[int(item[0])])
        str_ += str(index) + ',' + str(item[0])+',' + str(item[1]) +',' + str(mean_score)  +',' +  str(mean_improve_ratio)  +',' +  str(mean_sort)  +',' +  str(ai_sequences[int(item[0])]) + '\n'
    # print(len(planB_train_score))
        res[item[0]] = {
            'weight_score': item[1],
            'mean_score': mean_score,
            'mean_improve_ratio': mean_improve_ratio,
            'mean_sort': mean_sort,
            'ope_seq': ai_sequences[int(item[0])]
        }
    if prefix==None:
        with open('planB_analyze_on_324.json','w') as f:
            # f.write(str_)
            json.dump(res, f)
    else:
        with open('planB_analyze_on_324_'+prefix+'.json','w') as f:
            # f.write(str_)
            json.dump(res, f)
def look_sequence_between_task():
    with open('planB_train_tasks.json','r') as f:
        planB_train_tasks = json.load(f)
    with open('clean_task_no_1_fix_label_s.json', 'r') as f:
        clean_data = json.load(f)
    rf_tasks = []
    knn_tasks = []
    dt_tasks = []
    lr_tasks = []
    svc_tasks = []

    for task in clean_data:
        if task not in planB_train_tasks:
            continue
        if clean_data[task]['model_type'] == 'RandomForestClassifier':
            rf_tasks.append(task)
        if clean_data[task]['model_type'] == 'KNeighborsClassifier':
            knn_tasks.append(task)
        if clean_data[task]['model_type'] == 'LogisticRegression':
            lr_tasks.append(task)
        if clean_data[task]['model_type'] == 'DecisionTreeClassifier':
            dt_tasks.append(task)
        if clean_data[task]['model_type'] == 'SVC':
            svc_tasks.append(task)
    look_planB_train(rf_tasks, 'rf')
    look_planB_train(knn_tasks, 'knn')
    look_planB_train(lr_tasks, 'lr')
    look_planB_train(dt_tasks, 'dt')
    look_planB_train(svc_tasks, 'svc')


def look_test_seq():
    with open('planB_crosss.json','r') as f:
        test_result = json.load(f)

    with open('clean_task_no_1_fix_label.json','r') as f:
        clean_data = json.load(f)
    mean_add_seq = 0
    mean_base = 0
    mean_best_score = 0
    all_num = 0
    for task in test_result:
        
        if len(list(test_result[task].keys())) == 0:
            continue
        # print(task, list(test_result[task].keys()))
        best_seq_id = list(test_result[task].keys())[0]
        best_score = test_result[task][best_seq_id]
        base = clean_data[task]['base']
        add_seq = clean_data[task]['ai_seq_mean']
        mean_add_seq += add_seq
        mean_base += base
        mean_best_score += best_score
        all_num += 1
        print(task, base, add_seq, best_score)
    print('mean_add_seq', mean_add_seq/all_num)
    print('mean_base', mean_base/all_num)
    print('mean_best_score', mean_best_score/all_num)

def look_top_3():
    with open('planB_cross.json','r') as f:
        test_result = json.load(f)
    ai_sequences = get_all_aiseq()
    for task in test_result:
        seq_list = list(test_result[task].keys())
        seq_list = seq_list[0:3]
        print('########')
        print(task)
        for seq_id in seq_list:
            print(ai_sequences[int(seq_id)], test_result[task][seq_id])

def generate_firefiy_ai_seq():
    with open('planB_test_tasks.json','r') as f:
        test_tasks = json.load(f)
    with open('clean_task_no_1_fix_label.json','r') as f:
        clean_data = json.load(f)
    with open('test_reward_56000.json','r') as f:
        test_reward = json.load(f)
    # test_notebooks = []
    # for task in clean_data:
    #     if task in test_tasks:
    #         for notebook_id in clean_data[task]['notebook_list']:
    #             test_notebooks.append(task)
    k_fold = 3
    with open('../firefiy_seq/jsons/classification_task_dic.json', 'r') as f:
        classification_task_dic = json.load(f)
    res = {}
    for task in clean_data:
        if task in test_tasks:
            for tid in classification_task_dic:
                if classification_task_dic['dataset'] + '_' + classification_task_dic['model'] + '_' + classification_task_dic['label'] == task:
                    found_tid = tid
                    break
            for notebook_id in clean_data[task]['notebook_list']:
                seq = test_reward[tid]['seq']['56000']
                res[notebook_id] = seq
    with open('firefly_sequence.json','r') as f:
        json.dump(res, f)
    
   


def table3():
    with open('clean_task_rl_200.json','r') as f:
        clean_data = json.load(f)

    hai = []
    hi = []
    deepline = []
    autosklearn = []
    planb = []
    planb_corss, hi_cross, hai_cross = [],[],[]

    planb2hai = 0
    planb2hi = 0
    planb2deepline = 0
    planb2autosklearn = 0
    hai2hi = 0
    all_ = 0

    orign_nan = 0

    hard_case_equal_to_hi = 0
    hard_case = 0

    rf_planb2hi = 0
    dt_planb2hi = 0

    rf_hi, rf_hai, rf_deepline, rf_autosklearn, rf_planb, rf_hi_cross, rf_planb_cross,rf_hai_cross = [],[],[],[],[],[],[],[]
    dt_hi, dt_hai, dt_deepline, dt_autosklearn, dt_planb, dt_hi_cross, dt_planb_cross, dt_hai_cross = [],[],[],[],[],[],[],[]
    lr_hi, lr_hai, lr_deepline, lr_autosklearn, lr_planb, lr_hi_cross, lr_planb_cross, lr_hai_cross = [],[],[],[],[],[],[],[]
    knn_hi, knn_hai, knn_deepline, knn_autosklearn, knn_planb, knn_hi_cross, knn_planb_cross, knn_hai_cross = [],[],[],[],[],[],[],[]
    svc_hi, svc_hai, svc_deepline, svc_autosklearn, svc_planb, svc_hi_cross, svc_planb_cross, svc_hai_cross = [],[],[],[],[],[],[],[]
    for task in clean_data:
        for notebook_id in clean_data[task]['notebook_list']:
            notebook = clean_data[task]['notebook_list'][notebook_id]
            if 'planB' not in notebook:
                continue
            if 'origin_cross' not in notebook:
                notebook['origin_cross'] = 'nan'
                orign_nan+=1
            if abs(notebook['hi']-notebook['planB']) > 0.4:
                continue
            if clean_data[task]['model_type'] == 'RandomForestClassifier':
                
                # if notebook['origin_cross'] != 'nan':
                    rf_hai.append(notebook['hai'])
                    rf_hi.append(notebook['hi'])
                    rf_deepline.append(notebook['ai-deepline'])
                    rf_autosklearn.append(notebook['ai-autosklearn'])
                    rf_planb.append(notebook['planB'])
                # if notebook['origin_cross'] != 'nan':
                    rf_hi_cross.append(notebook['origin_cross'])
                    rf_planb_cross.append(notebook['max_planB'])
                    rf_hai_cross.append(notebook['max_hai'])
                    if notebook['max_validation_score'] > notebook['origin']:
                        rf_planb2hi += 1


            if clean_data[task]['model_type'] == 'DecisionTreeClassifier':
                # if notebook['origin_cross'] != 'nan':
                    dt_hai.append(notebook['hai'])
                    dt_hi.append(notebook['hi'])
                    dt_deepline.append(notebook['ai-deepline'])
                    dt_autosklearn.append(notebook['ai-autosklearn'])
                    dt_planb.append(notebook['planB'])
                    
                    dt_hi_cross.append(notebook['origin_cross'])
                    dt_planb_cross.append(notebook['max_planB'])
                    dt_hai_cross.append(notebook['max_hai'])
                    if notebook['max_validation_score'] > notebook['origin']:
                        dt_planb2hi += 1
                
            if clean_data[task]['model_type'] == 'KNeighborsClassifier':
                # if notebook['origin_cross'] != 'nan':
                    knn_hai.append(notebook['hai'])
                    knn_hi.append(notebook['hi'])
                    knn_deepline.append(notebook['ai-deepline'])
                    knn_autosklearn.append(notebook['ai-autosklearn'])
                    knn_planb.append(notebook['planB'])
                # if notebook['origin_cross'] != 'nan':
                    knn_hi_cross.append(notebook['origin_cross'])
                    knn_planb_cross.append(notebook['max_planB'])
                    knn_hai_cross.append(notebook['max_hai'])
            if clean_data[task]['model_type'] == 'LogisticRegression':
                # if notebook['origin_cross'] != 'nan':
                    lr_hai.append(notebook['hai'])
                    lr_hi.append(notebook['hi'])
                    lr_deepline.append(notebook['ai-deepline'])
                    lr_autosklearn.append(notebook['ai-autosklearn'])
                    lr_planb.append(notebook['planB'])
                # if notebook['origin_cross'] != 'nan':
                    lr_hi_cross.append(notebook['origin_cross'])
                    lr_planb_cross.append(notebook['max_planB'])
                    lr_hai_cross.append(notebook['max_hai'])
            if clean_data[task]['model_type'] == 'SVC':
                # if notebook['origin_cross'] != 'nan':
                    svc_hai.append(notebook['hai'])
                    svc_hi.append(notebook['hi'])
                    svc_deepline.append(notebook['ai-deepline'])
                    svc_autosklearn.append(notebook['ai-autosklearn'])
                    svc_planb.append(notebook['planB'])
                # if notebook['origin_cross'] != 'nan':
                    svc_hi_cross.append(notebook['origin_cross'])
                    svc_planb_cross.append(notebook['max_planB'])
                    svc_hai_cross.append(notebook['max_hai'])
            all_ += 1
            if notebook['origin_cross'] != 'nan':
                hai.append(notebook['hai'])
                hi.append(notebook['hi'])
                deepline.append(notebook['ai-deepline'])
                autosklearn.append(notebook['ai-autosklearn'])
                planb.append(notebook['planB'])
                planb_corss.append(notebook['max_planB'])
                hi_cross.append(notebook['origin_cross'])
                hai_cross.append(notebook['max_hai'])

                if notebook['hi'] >= notebook['planB'] and notebook['origin_cross'] < notebook['max_planB'] and notebook['hi'] - notebook['planB'] <= 0.05 and notebook['hi'] - notebook['planB'] > 0.02:
                    print("xxxxxx")
                    print(notebook_id)
                    print(notebook['hi'], notebook['planB'], notebook['origin_cross'], notebook['max_planB'])
                    hard_case += 1
                    if notebook['hi'] == notebook['planB']:
                        hard_case_equal_to_hi += 1
            else:
                orign_nan += 1


            if notebook['hai'] > notebook['hi']:
                hai2hi += 1
            if notebook['planB'] > notebook['hi']:
                planb2hi += 1
            if notebook['planB'] > notebook['hai']:
                planb2hai += 1
            if notebook['planB'] > notebook['ai-deepline']:
                planb2deepline += 1
            if notebook['planB'] > notebook['ai-autosklearn']:
                planb2autosklearn += 1

    

    print(orign_nan)
    print('hard_case', hard_case)
    print('hard_case_equal_to_hi/hard_case', hard_case_equal_to_hi/hard_case)
    print('table3')
    print('      mean', '.25', '.5', '.75')
    print('hi', round(np.array(hi).mean(),3), round(np.quantile(np.array(hi), 0.25),3), round(np.quantile(np.array(hi), 0.5),3), round(np.quantile(np.array(hi), 0.75),3))
    print('hai', round(np.array(hai).mean(),3), round(np.quantile(np.array(hai), 0.25),3), round(np.quantile(np.array(hai), 0.5),3), round(np.quantile(np.array(hai), 0.75),3))
    print('deepline', round(np.array(deepline).mean(),3), round(np.quantile(np.array(deepline), 0.25),3), round(np.quantile(np.array(deepline), 0.5),3), round(np.quantile(np.array(deepline), 0.75),3))
    print('autosklearn', round(np.array(autosklearn).mean(),3), round(np.quantile(np.array(autosklearn), 0.25),3), round(np.quantile(np.array(autosklearn), 0.5),3), round(np.quantile(np.array(autosklearn), 0.75),3))
    print('planb', round(np.array(planb).mean(),3), round(np.quantile(np.array(planb), 0.25),3), round(np.quantile(np.array(planb), 0.5),3), round(np.quantile(np.array(planb), 0.75),3))
    print('planbcross', round(np.array(planb_corss).mean(),3), round(np.quantile(np.array(planb_corss), 0.25),3), round(np.quantile(np.array(planb_corss), 0.5),3), round(np.quantile(np.array(planb_corss), 0.75),3))
    print('hicross', round(np.array(hi_cross).mean(),3), round(np.quantile(np.array(hi_cross), 0.25),3), round(np.quantile(np.array(hi_cross), 0.5),3), round(np.quantile(np.array(hi_cross), 0.75),3))
    print('haicross', round(np.array(hai_cross).mean(),3), round(np.quantile(np.array(hai_cross), 0.25),3), round(np.quantile(np.array(hai_cross), 0.5),3), round(np.quantile(np.array(hai_cross), 0.75),3))
    
    print('figure8')
    print('% hai > hi', hai2hi / all_)
    print('% planb > hai', planb2hai / all_)
    print('% planb > hi', planb2hi / all_)
    print('% planb > deepline', planb2deepline/ all_)
    print('% planb > autosklearn', planb2autosklearn / all_)
    print('rf % planb > hi', rf_planb2hi / len(rf_hi))
    print('dt % planb > hi', dt_planb2hi / len(dt_hi))

    print('group by model')
    print('rf      mean', '.25', '.5', '.75')
    print('hi', round(np.array(rf_hi).mean(),3), round(np.quantile(np.array(rf_hi), 0.25),3), round(np.quantile(np.array(rf_hi), 0.5),3), round(np.quantile(np.array(rf_hi), 0.75),3))
    print('hai', round(np.array(rf_hai).mean(),3), round(np.quantile(np.array(rf_hai), 0.25),3), round(np.quantile(np.array(rf_hai), 0.5),3), round(np.quantile(np.array(rf_hai), 0.75),3))
    print('deepline', round(np.array(rf_deepline).mean(),3), round(np.quantile(np.array(rf_deepline), 0.25),3), round(np.quantile(np.array(rf_deepline), 0.5),3), round(np.quantile(np.array(rf_deepline), 0.75),3))
    print('autosklearn', round(np.array(rf_autosklearn).mean(),3), round(np.quantile(np.array(rf_autosklearn), 0.25),3), round(np.quantile(np.array(rf_autosklearn), 0.5),3), round(np.quantile(np.array(rf_autosklearn), 0.75),3))
    print('planb', round(np.array(rf_planb).mean(),3), round(np.quantile(np.array(rf_planb), 0.25),3), round(np.quantile(np.array(rf_planb), 0.5),3), round(np.quantile(np.array(rf_planb), 0.75),3))
    print('planbcross', round(np.array(rf_planb_cross).mean(),3), round(np.quantile(np.array(rf_planb_cross), 0.25),3), round(np.quantile(np.array(rf_planb_cross), 0.5),3), round(np.quantile(np.array(rf_planb_cross), 0.75),3))
    print('hicross', round(np.array(rf_hi_cross).mean(),3), round(np.quantile(np.array(rf_hi_cross), 0.25),3), round(np.quantile(np.array(rf_hi_cross), 0.5),3), round(np.quantile(np.array(rf_hi_cross), 0.75),3))
    print('haicross', round(np.array(rf_hai_cross).mean(),3), round(np.quantile(np.array(rf_hai_cross), 0.25),3), round(np.quantile(np.array(rf_hai_cross), 0.5),3), round(np.quantile(np.array(rf_hai_cross), 0.75),3))
    print('len rf', len(rf_hi))
    print('dt      mean', '.25', '.5', '.75')
    print('hi', round(np.array(dt_hi).mean(),3), round(np.quantile(np.array(dt_hi), 0.25),3), round(np.quantile(np.array(dt_hi), 0.5),3), round(np.quantile(np.array(dt_hi), 0.75),3))
    print('hai', round(np.array(dt_hai).mean(),3), round(np.quantile(np.array(dt_hai), 0.25),3), round(np.quantile(np.array(dt_hai), 0.5),3), round(np.quantile(np.array(dt_hai), 0.75),3))
    print('deepline', round(np.array(dt_deepline).mean(),3), round(np.quantile(np.array(dt_deepline), 0.25),3), round(np.quantile(np.array(dt_deepline), 0.5),3), round(np.quantile(np.array(dt_deepline), 0.75),3))
    print('autosklearn', round(np.array(dt_autosklearn).mean(),3), round(np.quantile(np.array(dt_autosklearn), 0.25),3), round(np.quantile(np.array(dt_autosklearn), 0.5),3), round(np.quantile(np.array(dt_autosklearn), 0.75),3))
    print('planb', round(np.array(dt_planb).mean(),3), round(np.quantile(np.array(dt_planb), 0.25),3), round(np.quantile(np.array(dt_planb), 0.5),3), round(np.quantile(np.array(dt_planb), 0.75),3))
    print('planbcross', round(np.array(dt_planb_cross).mean(),3), round(np.quantile(np.array(dt_planb_cross), 0.25),3), round(np.quantile(np.array(dt_planb_cross), 0.5),3), round(np.quantile(np.array(dt_planb_cross), 0.75),3))
    print('hicross', round(np.array(dt_hi_cross).mean(),3), round(np.quantile(np.array(dt_hi_cross), 0.25),3), round(np.quantile(np.array(dt_hi_cross), 0.5),3), round(np.quantile(np.array(dt_hi_cross), 0.75),3))
    print('haicross', round(np.array(dt_hai_cross).mean(),3), round(np.quantile(np.array(dt_hai_cross), 0.25),3), round(np.quantile(np.array(dt_hai_cross), 0.5),3), round(np.quantile(np.array(dt_hai_cross), 0.75),3))
    print('len dt', len(dt_hi))
    print('knn      mean', '.25', '.5', '.75')
    print('hi', round(np.array(knn_hi).mean(),3), round(np.quantile(np.array(knn_hi), 0.25),3), round(np.quantile(np.array(knn_hi), 0.5),3), round(np.quantile(np.array(knn_hi), 0.75),3))
    print('hai', round(np.array(knn_hai).mean(),3), round(np.quantile(np.array(knn_hai), 0.25),3), round(np.quantile(np.array(knn_hai), 0.5),3), round(np.quantile(np.array(knn_hai), 0.75),3))
    print('deepline', round(np.array(knn_deepline).mean(),3), round(np.quantile(np.array(knn_deepline), 0.25),3), round(np.quantile(np.array(knn_deepline), 0.5),3), round(np.quantile(np.array(knn_deepline), 0.75),3))
    print('autosklearn', round(np.array(knn_autosklearn).mean(),3), round(np.quantile(np.array(knn_autosklearn), 0.25),3), round(np.quantile(np.array(knn_autosklearn), 0.5),3), round(np.quantile(np.array(knn_autosklearn), 0.75),3))
    print('planb', round(np.array(knn_planb).mean(),3), round(np.quantile(np.array(knn_planb), 0.25),3), round(np.quantile(np.array(knn_planb), 0.5),3), round(np.quantile(np.array(knn_planb), 0.75),3))

    print('lr      mean', '.25', '.5', '.75')
    print('hi', round(np.array(lr_hi).mean(),3), round(np.quantile(np.array(lr_hi), 0.25),3), round(np.quantile(np.array(lr_hi), 0.5),3), round(np.quantile(np.array(lr_hi), 0.75),3))
    print('hai', round(np.array(lr_hai).mean(),3), round(np.quantile(np.array(lr_hai), 0.25),3), round(np.quantile(np.array(lr_hai), 0.5),3), round(np.quantile(np.array(lr_hai), 0.75),3))
    print('deepline', round(np.array(lr_deepline).mean(),3), round(np.quantile(np.array(lr_deepline), 0.25),3), round(np.quantile(np.array(lr_deepline), 0.5),3), round(np.quantile(np.array(lr_deepline), 0.75),3))
    print('autosklearn', round(np.array(lr_autosklearn).mean(),3), round(np.quantile(np.array(lr_autosklearn), 0.25),3), round(np.quantile(np.array(lr_autosklearn), 0.5),3), round(np.quantile(np.array(lr_autosklearn), 0.75),3))
    print('planb', round(np.array(lr_planb).mean(),3), round(np.quantile(np.array(lr_planb), 0.25),3), round(np.quantile(np.array(lr_planb), 0.5),3), round(np.quantile(np.array(lr_planb), 0.75),3))

    print('svc      mean', '.25', '.5', '.75')
    print('hi', round(np.array(svc_hi).mean(),3), round(np.quantile(np.array(svc_hi), 0.25),3), round(np.quantile(np.array(svc_hi), 0.5),3), round(np.quantile(np.array(svc_hi), 0.75),3))
    print('hai', round(np.array(svc_hai).mean(),3), round(np.quantile(np.array(svc_hai), 0.25),3), round(np.quantile(np.array(svc_hai), 0.5),3), round(np.quantile(np.array(svc_hai), 0.75),3))
    print('deepline', round(np.array(svc_deepline).mean(),3), round(np.quantile(np.array(svc_deepline), 0.25),3), round(np.quantile(np.array(svc_deepline), 0.5),3), round(np.quantile(np.array(svc_deepline), 0.75),3))
    print('autosklearn', round(np.array(svc_autosklearn).mean(),3), round(np.quantile(np.array(svc_autosklearn), 0.25),3), round(np.quantile(np.array(svc_autosklearn), 0.5),3), round(np.quantile(np.array(svc_autosklearn), 0.75),3))
    print('planb', round(np.array(svc_planb).mean(),3), round(np.quantile(np.array(svc_planb), 0.25),3), round(np.quantile(np.array(svc_planb), 0.5),3), round(np.quantile(np.array(svc_planb), 0.75),3))



def look_validaiton_test():
    with open('clean_task_add_planB.json','r') as f:
        clean_data = json.load(f)

    hi_val = []
    hai_val = []
    hi_test = []
    hai_test = []

    rf_hi_val, rf_hai_val, rf_hi_test, rf_hai_test = [],[],[],[]
    dt_hi_val, dt_hai_val, dt_hi_test, dt_hai_test = [],[],[],[]
    for task in clean_data:
        for notebook_id in clean_data[task]['notebook_list']:
            notebook = clean_data[task]['notebook_list'][notebook_id]
        if 'origin_cross' not in notebook:
            continue
        if notebook['origin_cross'] == 'nan':
            continue

        hi_val.append(notebook['origin_cross'])
        hai_val.append(notebook['max_hai'])
        hi_test.append(notebook['hi'])
        hai_test.append(notebook['hai'])

        if clean_data[task]['model_type'] == 'RandomForestClassifier':
            rf_hi_val.append(notebook['origin_cross'])
            rf_hai_val.append(notebook['max_hai'])
            rf_hi_test.append(notebook['hi'])
            rf_hai_test.append(notebook['hai'])
        if clean_data[task]['model_type'] == 'DecisionTreeClassifier':
            dt_hi_val.append(notebook['origin_cross'])
            dt_hai_val.append(notebook['max_hai'])
            dt_hi_test.append(notebook['hi'])
            dt_hai_test.append(notebook['hai'])

    print('hi_val', np.array(hi_val).mean())
    print('hai_val', np.array(hai_val).mean())
    print('hi_test', np.array(hi_test).mean())
    print('hai_test', np.array(hai_test).mean())

    print()
    print('rf_hi_val', np.array(rf_hi_val).mean())
    print('rf_hai_val', np.array(rf_hai_val).mean())
    print('rf_hi_test', np.array(rf_hi_test).mean())
    print('rf_hai_test', np.array(rf_hai_test).mean())
    
    print('dt_hi_val', np.array(dt_hi_val).mean())
    print('dt_hai_val', np.array(dt_hai_val).mean())
    print('dt_hi_test', np.array(dt_hi_test).mean())
    print('dt_hai_test', np.array(dt_hai_test).mean())

def overview():
    with open('clean_task_add_planB.json','r') as f:
        all_tasks = json.load(f)
    with open('clean_task_rl_200.json','r') as f:
        test_tasks = json.load(f)

    print(len(all_tasks)-len(test_tasks))
    print(len(test_tasks))
    train_notebooks = 0
    test_notebooks = 0

    train_model = {}
    test_model = {}

    train_model_n = {}
    test_model_n = {}

    train_dataset = set()
    test_dataset = set()

    all_col_num = []
    all_row_num = []
    train_col_num = []
    train_row_num = []
    test_col_num = []
    test_row_num = []
    for task in all_tasks:
        all_col_num.append(all_tasks[task]['col_num'])
        all_row_num.append(all_tasks[task]['row_num'])
        if task not in test_tasks.keys():
            train_notebooks += len(all_tasks[task]['notebook_list'])
            train_dataset.add(all_tasks[task]['dataset'])
            if all_tasks[task]['model_type'] not in train_model:
                train_model[all_tasks[task]['model_type']] = 0
                train_model_n[all_tasks[task]['model_type']] = 0
            train_model[all_tasks[task]['model_type']] +=1 
            train_model_n[all_tasks[task]['model_type']] += len(all_tasks[task]['notebook_list'])
            train_col_num.append(all_tasks[task]['col_num'])
            train_row_num.append(all_tasks[task]['row_num'])
        else:
            test_notebooks += len(all_tasks[task]['notebook_list'])
            test_dataset.add(all_tasks[task]['dataset'])
            if all_tasks[task]['model_type'] not in test_model:
                test_model[all_tasks[task]['model_type']] = 0
                test_model_n[all_tasks[task]['model_type']] = 0
            test_model[all_tasks[task]['model_type']] +=1 
            test_model_n[all_tasks[task]['model_type']] += len(all_tasks[task]['notebook_list'])
            test_col_num.append(all_tasks[task]['col_num'])
            test_row_num.append(all_tasks[task]['row_num'])
    print(train_notebooks)
    print(test_notebooks)

    print(len(train_dataset))
    print(len(test_dataset))

    print(train_model)
    print(test_model)

    print(train_model_n)
    print(test_model_n)

    all_col_num = np.array(all_col_num)
    all_row_num = np.array(all_row_num)
    train_col_num = np.array(train_col_num)
    train_row_num = np.array(train_row_num)
    test_col_num = np.array(test_col_num)
    test_row_num = np.array(test_row_num)
    print(all_col_num.mean())
    print(all_row_num.mean())
    print(train_col_num.mean())
    print(train_row_num.mean())
    print(test_col_num.mean())
    print(test_row_num.mean())

    print(all_col_num.min())
    print(all_row_num.min())
    print(train_col_num.min())
    print(train_row_num.min())
    print(test_col_num.min())
    print(test_row_num.min())

    print(all_col_num.max())
    print(all_row_num.max())
    print(train_col_num.max())
    print(train_row_num.max())
    print(test_col_num.max())
    print(test_row_num.max())
    
def look_can_remove():
    with open("/home/yxm/staticfg-master/clean_task_rl_200.json",'r') as f:
        clean_data = json.load(f)
    test_notebooks = []
    for task in clean_data:
        for notebook_id in clean_data[task]['notebook_list']:
            test_notebooks.append(notebook_id)
    notebook2seq = np.load('/home/yxm/staticfg-master/NewDataSeq1.npy', allow_pickle=True).item()
    # print(notebook2seq) 
    all_ = 0
    has_ope = 0
    for notebook_id in notebook2seq:
        if notebook_id not in test_notebooks:
            continue
        # print(notebook_id, notebook2seq[notebook_id])
        cleaned = []
        for ope in notebook2seq[notebook_id]:
            if ope not in OperationType['Scaler'] and ope not in OperationType['FeatureEngine'] and ope not in OperationType['FeatureSelection']:
                continue
            cleaned.append(ope)
        # cleaned = set(cleaned)
        if len(cleaned) > 0:
            has_ope += 1
            print(has_ope)
            print(notebook_id, cleaned)
            # os.system('mv new_data/delete_prenotebook_code/'+notebook_id+'.py new_data/delete_more_2/')
        all_ += 1
    print(has_ope)
    print(all_)

def get_deepline():
    with open("clean_task_rl_200.json",'r') as f:
        clean_task = json.load(f)
    rf_res = np.load('/home/chensibei/staticfg-master/deepline/test_reward_rf_log.npy',allow_pickle=True).item()
    dt_res = np.load('/home/chensibei/staticfg-master/deepline/test_reward_dt_log.npy',allow_pickle=True).item()
    knn_res = np.load('/home/chensibei/staticfg-master/deepline/test_reward_knn_log.npy',allow_pickle=True).item()
    lr_res = np.load('/home/chensibei/staticfg-master/deepline/test_reward_lr_log.npy',allow_pickle=True).item()
    svc_res = np.load('/home/chensibei/staticfg-master/deepline/test_reward_svc_log.npy',allow_pickle=True).item()
    
    rf_time = np.load('/home/chensibei/staticfg-master/deepline/time_rf_log.npy',allow_pickle=True).item()
    dt_time = np.load('/home/chensibei/staticfg-master/deepline/time_dt_log.npy',allow_pickle=True).item()
    knn_time = np.load('/home/chensibei/staticfg-master/deepline/time_knn_log.npy',allow_pickle=True).item()
    lr_time = np.load('/home/chensibei/staticfg-master/deepline/time_lr_log.npy',allow_pickle=True).item()
    svc_time = np.load('/home/chensibei/staticfg-master/deepline/time_svc_log.npy',allow_pickle=True).item()
    
    
    with open('/home/chensibei/staticfg-master/deepline/all_test_lj_job.json','r') as f:
        datasets = json.load(f)
    
    dataset2rfscore = {}
    dataset2dtscore = {}
    dataset2knnscore = {}
    dataset2lrscore = {}
    dataset2svcscore = {}

    dataset2rftime = {}
    dataset2dttime = {}
    dataset2knntime = {}
    dataset2lrtime = {}
    dataset2svctime = {}
    for index,dataset in enumerate(datasets):
        dataset2rfscore[dataset] = rf_res[index]
        dataset2dtscore[dataset] = dt_res[index]
        dataset2knnscore[dataset] = knn_res[index]
        dataset2lrscore[dataset] = lr_res[index]
        dataset2svcscore[dataset] = svc_res[index]

        dataset2rftime[dataset] = rf_time[index]
        dataset2dttime[dataset] = dt_time[index]
        dataset2knntime[dataset] = knn_time[index]
        dataset2lrtime[dataset] = lr_time[index]
        dataset2svctime[dataset] = svc_time[index]
    
    pprint.pprint(dataset2svctime)
    # print('dataset2rfscore', dataset2rfscore)
    count = 0
    for task in clean_task:
        if clean_task[task]['model_type'] == 'RandomForestClassifier':
            clean_task[task]['deepline'] = max(dataset2rfscore[clean_task[task]['dataset']],0)
            clean_task[task]['deepline_step1_time'] = dataset2rftime[clean_task[task]['dataset']]
            count += 1
            print(clean_task[task]['deepline_step1_time'] )
        if clean_task[task]['model_type'] == 'DecisionTreeClassifier':
            clean_task[task]['deepline'] = max(dataset2dtscore[clean_task[task]['dataset']],0)
            clean_task[task]['deepline_step1_time'] = dataset2dttime[clean_task[task]['dataset']]
            count += 1
            print(clean_task[task]['deepline_step1_time'] )
        if clean_task[task]['model_type'] == 'KNeighborsClassifier':
            clean_task[task]['deepline'] = max(dataset2knnscore[clean_task[task]['dataset']],0)
            clean_task[task]['deepline_step1_time'] = dataset2knntime[clean_task[task]['dataset']]
            count += 1
            print(clean_task[task]['deepline_step1_time'] )
        if clean_task[task]['model_type'] == 'LogisticRegression':
            clean_task[task]['deepline'] = max(dataset2lrscore[clean_task[task]['dataset']],0)
            clean_task[task]['deepline_step1_time'] = dataset2lrtime[clean_task[task]['dataset']]
            count += 1
            print(clean_task[task]['deepline_step1_time'] )
        if clean_task[task]['model_type'] == 'SVC':
            clean_task[task]['deepline'] = max(dataset2svcscore[clean_task[task]['dataset']],0)
            clean_task[task]['deepline_step1_time'] = dataset2svctime[clean_task[task]['dataset']]
            count += 1
            print(clean_task[task]['deepline_step1_time'] )
    # print(count)
    with open("clean_task_rl_200.json",'w') as f:
        json.dump(clean_task, f)


def look_final_planb():
    with open('origin_planB_notebooks.json','r') as f:
        origin_planB_notebooks = json.load(f)
    with open('origin_rl_notebooks.json','r') as f:
        origin_rl_notebooks = json.load(f)
    planB_hai = os.listdir('/home/chensibei/staticfg-master/new_data/final_planB_hai')
    rl_hai = os.listdir('/home/chensibei/staticfg-master/new_data/final_rl_hai')
    hi = os.listdir('/home/chensibei/staticfg-master/new_data/final_hi_res')
    planB_ai = os.listdir('/home/chensibei/staticfg-master/new_data/final_planB_ai')
    rl_ai = os.listdir('/home/chensibei/staticfg-master/final_rl_ai_res')
    planb_hai_count = 0
    for item in planB_hai:
        if len(os.listdir('/home/chensibei/staticfg-master/new_data/final_planB_hai/'+item)) != 0:
            planb_hai_count += 1
    print('planB hai', planb_hai_count)
    rl_hai_count = 0
    for item in rl_hai:
        if len(os.listdir('/home/chensibei/staticfg-master/new_data/final_rl_hai/'+item)) != 0:
            rl_hai_count += 1
    print('rl hai', rl_hai_count)
    count = 0
    for item in planB_ai:
        if len(os.listdir('/home/chensibei/staticfg-master/new_data/final_planB_ai/'+item)) != 0:
            count += 1
    print('planB ai', count)
    count = 0
    for item in rl_ai:
        if os.path.exists('/home/chensibei/staticfg-master/final_rl_ai_res/'+item) and len(os.listdir('final_rl_ai_res/'+item)) != 0:
            count += 1
    print('rl ai', count)
    count = 0
    for item in hi:
        if os.path.exists('/home/chensibei/staticfg-master/new_data/final_hi_res/'):
            count += 1
    print('hi', count)
   
    print(len(origin_planB_notebooks) + planb_hai_count)
    print(len(origin_rl_notebooks) + rl_hai_count)
    # print('planB', count)
    
def look_planB_cross():
    with open('planB_cross.json','r') as f:
        planB_cross = json.load(f)
    print(len(planB_cross))


def generate_new_data():
    with open("clean_task_rl_200.json",'r') as f:
        clean_data = json.load(f)
    with open('/home/chensibei/staticfg-master/origin_planB_notebooks.json','r') as f:
        origin_planB_notebooks = json.load(f)
    with open('/home/chensibei/staticfg-master/origin_rl_notebooks.json','r') as f:
        origin_rl_notebooks = json.load(f)
    new_data = {}
    planB_hai = os.listdir('/home/chensibei/staticfg-master/new_data/final_planB_hai')
    rl_hai = os.listdir('/home/chensibei/staticfg-master/new_data/final_rl_hai')
    hi = os.listdir('/home/chensibei/staticfg-master/new_data/final_hi_res')
    planB_ai = os.listdir('/home/chensibei/staticfg-master/new_data/final_planB_ai')

    old_hi = []
    new_hi = []
    old_rl_ai = []
    new_rl_ai = []
    old_rl_hai = []
    new_rl_hai = []

    for task in clean_data:
        new_data[task] = {}
        if os.path.exists('/home/chensibei/staticfg-master/new_data/final_planB_ai/'+task):
            filename = os.listdir('/home/chensibei/staticfg-master/new_data/final_planB_ai/'+task)[0]
            score = np.load('/home/chensibei/staticfg-master/new_data/final_planB_ai/'+task + '/' + filename, allow_pickle=True).item()['accuracy_score']
            new_data[task]['planB_ai'] = score
        if os.path.exists('/home/chensibei/staticfg-master/final_rl_ai_res/'+task):
            filename = os.listdir('/home/chensibei/staticfg-master/final_rl_ai_res/'+task)[0]
            score = np.load('/home/chensibei/staticfg-master/final_rl_ai_res/'+task + '/' + filename, allow_pickle=True).item()['accuracy_score']
            new_data[task]['rl_ai'] = score
        new_data[task]['notebook_list'] = {}
        for notebook_id in clean_data[task]['notebook_list']:
            print(notebook_id)
            new_rl_ai.append(new_data[task]['rl_ai'])
            old_rl_ai.append(clean_data[task]['rl_ai_seq'])
            new_data[task]['notebook_list'][notebook_id] = {}
            if os.path.exists('/home/chensibei/staticfg-master/new_data/final_hi_res/'+notebook_id +'.npy'):
                score = np.load('/home/chensibei/staticfg-master/new_data/final_hi_res/'+notebook_id +'.npy', allow_pickle=True).item()['accuracy_score']
                new_data[task]['notebook_list'][notebook_id]['hi'] = score
                new_hi.append(new_data[task]['notebook_list'][notebook_id]['hi'])
                old_hi.append(clean_data[task]['notebook_list'][notebook_id]['hi'])
            else:
                continue
            if os.path.exists('/home/chensibei/staticfg-master/new_data/final_planB_hai/'+notebook_id):
                
                if notebook_id in origin_planB_notebooks:
                    score = new_data[task]['notebook_list'][notebook_id]['hi']
                    new_data[task]['notebook_list'][notebook_id]['planB_hai'] = score
                else:
                    filelist = os.listdir('/home/chensibei/staticfg-master/new_data/final_planB_hai/'+notebook_id)
                    if len(filelist) > 0:
                        filename = filelist[0]
                        score = np.load('/home/chensibei/staticfg-master/new_data/final_planB_hai/'+notebook_id+'/'+filename , allow_pickle=True).item()['accuracy_score']
                        new_data[task]['notebook_list'][notebook_id]['planB_hai'] = score
                    else:
                        continue
            if os.path.exists('/home/chensibei/staticfg-master/new_data/final_rl_hai/'+notebook_id):
                
                if notebook_id in origin_rl_notebooks:
                    score = new_data[task]['notebook_list'][notebook_id]['hi']
                    new_data[task]['notebook_list'][notebook_id]['rl_hai'] = score
                    new_rl_hai.append(new_data[task]['notebook_list'][notebook_id]['rl_hai'])
                    old_rl_hai.append(clean_data[task]['notebook_list'][notebook_id]['rl'])
                else:
                    filelist = os.listdir('/home/chensibei/staticfg-master/new_data/final_rl_hai/'+notebook_id)
                    if len(filelist) > 0:
                        filename = filelist[0]
                        score = np.load('/home/chensibei/staticfg-master/new_data/final_rl_hai/'+notebook_id+'/'+filename, allow_pickle=True).item()['accuracy_score']
                        new_data[task]['notebook_list'][notebook_id]['rl_hai'] = score
                        new_rl_hai.append(new_data[task]['notebook_list'][notebook_id]['rl_hai'])
                        old_rl_hai.append(clean_data[task]['notebook_list'][notebook_id]['rl'])
                    else:
                        continue
                # filename = os.listdir('new_data/final_rl_hai/'+notebook_id)[0]
                # score = np.load('new_data/final_rl_hai/'+notebook_id+'/'+filename +'.npy', allow_pickle=True).item()['accuracy_score']
                # new_data[task]['notebook_list'][notebook_id]['rl_hai'] = score
    print('old_rl_ai', np.array(old_rl_ai).mean())
    print('new_rl_ai', np.array(new_rl_ai).mean())
    print('old_rl_hai', np.array(old_rl_hai).mean())
    print('new_rl_hai', np.array(new_rl_hai).mean())
    print('old_hi', np.array(old_hi).mean())
    print('new_hi', np.array(new_hi).mean())
    with open("final_result_215.json",'w') as f:
        json.dump(new_data, f)
    with open("final_result_sibei_215.json",'w') as f:
        json.dump(new_data, f)

def generate_new_data_clean():
    with open("final_result_sibei_215.json",'r') as f:
        new_data =  json.load(f)
    with open("clean_task_rl_200.json",'r') as f:
        clean_data = json.load(f)
    notebook_num = 0

    for task in new_data:
        new_notebook_list = {}
        for notebook_id in new_data[task]['notebook_list']:
            notebook = new_data[task]['notebook_list'][notebook_id]
            if len(notebook) == 3:
                notebook['ai-autosklearn'] = clean_data[task]['notebook_list'][notebook_id]['ai-autosklearn']
                new_notebook_list[notebook_id] = notebook.copy()
                notebook_num += 1
        new_data[task]['notebook_list'] = new_notebook_list
        new_data[task]['deepline'] = clean_data[task]['deepline']
        new_data[task]['model_type'] = clean_data[task]['model_type']
        new_data[task]['dataset'] = clean_data[task]['dataset']
        new_data[task]['label'] = clean_data[task]['label']
   

    new_res = {}
    for task in new_data:
        if len(new_data[task]['notebook_list']) > 0:
            new_res[task] = new_data[task]

    print(len(new_res))
    print(notebook_num)
    with open("final_result_sibei_215.json",'w') as f:
        json.dump(new_res, f)

def step1res():
    with open("final_result_sibei_hole.json",'r') as f:
        new_data =  json.load(f)
    with open("final_result_sibei_215_del_split.json",'r') as f:
        new_data_1 =  json.load(f)
    all_planb_ai, all_rl_ai, all_deepline_ai = [],[],[]
    rf_planb_ai, rf_rl_ai, rf_deepline_ai = [],[],[]
    dt_planb_ai, dt_rl_ai, dt_deepline_ai = [],[],[]
    knn_planb_ai, knn_rl_ai, knn_deepline_ai = [],[],[]
    lr_planb_ai, lr_rl_ai, lr_deepline_ai = [],[],[]
    svc_planb_ai, svc_rl_ai, svc_deepline_ai = [],[],[]
    model = ['RandomForestClassifier', 'DecisionTreeClassifier', 'KNeighborsClassifier', 'LogisticRegression', 'SVC']
    planb_list = [rf_planb_ai, dt_planb_ai, knn_planb_ai, lr_planb_ai, svc_planb_ai]
    rl_list = [rf_rl_ai, dt_rl_ai, knn_rl_ai, lr_rl_ai, svc_rl_ai]
    deepline_list = [rf_deepline_ai, dt_deepline_ai, knn_deepline_ai, lr_deepline_ai, svc_deepline_ai]
    for task in new_data:
        if task not in new_data_1:
            continue
        if len(new_data_1[task]['notebook_list']) == 0:
            continue
        model_index = model.index(new_data[task]['model_type'])
        planb_list[model_index].append(new_data[task]['planB_ai'])
        rl_list[model_index].append(new_data[task]['rl_ai'])
        deepline_list[model_index].append(new_data[task]['deepline'])
        all_planb_ai.append(new_data[task]['planB_ai'])
        all_rl_ai.append(new_data[task]['rl_ai'])
        all_deepline_ai.append(new_data[task]['deepline'])

    all_planb_ai = np.array(all_planb_ai)
    print('all_planb_ai', all_planb_ai.mean())
    all_rl_ai = np.array(all_rl_ai)
    print('all_rl_ai', all_rl_ai.mean())
    all_deepline_ai = np.array(all_deepline_ai)
    print('all_deepline_ai', all_deepline_ai.mean())
    rf_planb_ai = np.array(rf_planb_ai)
    print('all',len(all_deepline_ai))
    print('rf_planb_ai', rf_planb_ai.mean())
    rf_rl_ai = np.array(rf_rl_ai)
    print('rf_rl_ai', rf_rl_ai.mean())
    rf_deepline_ai = np.array(rf_deepline_ai)
    print('rf_deepline_ai', rf_deepline_ai.mean())
    dt_planb_ai = np.array(dt_planb_ai)
    print('rf',len(rf_deepline_ai))
    print('dt_planb_ai', dt_planb_ai.mean())
    dt_rl_ai = np.array(dt_rl_ai)
    print('dt_rl_ai', dt_rl_ai.mean())
    dt_deepline_ai = np.array(dt_deepline_ai)
    print('dt_deepline_ai', dt_deepline_ai.mean())
    print('dt',len(dt_deepline_ai))
    knn_planb_ai = np.array(knn_planb_ai)
    print('knn_planb_ai', knn_planb_ai.mean())
    knn_rl_ai = np.array(knn_rl_ai)
    print('knn_rl_ai', knn_rl_ai.mean())
    knn_deepline_ai = np.array(knn_deepline_ai)
    print('knn_deepline_ai', knn_deepline_ai.mean())
    lr_planb_ai = np.array(lr_planb_ai)
    print('knn',len(knn_deepline_ai))
    print('lr_planb_ai', lr_planb_ai.mean())
    lr_rl_ai = np.array(lr_rl_ai)
    print('lr_rl_ai', lr_rl_ai.mean())
    lr_deepline_ai = np.array(lr_deepline_ai)
    print('lr_deepline_ai', lr_deepline_ai.mean())
    print('lr',len(lr_deepline_ai))
    svc_planb_ai = np.array(svc_planb_ai)
    print('svc_planb_ai', svc_planb_ai.mean())
    svc_rl_ai = np.array(svc_rl_ai)
    print('svc_rl_ai', svc_rl_ai.mean())
    svc_deepline_ai = np.array(svc_deepline_ai)
    print('svc_deepline_ai', svc_deepline_ai.mean())
    print('svc',len(svc_deepline_ai))
    

def step2res():
    with open("final_result_sibei_hole.json",'r') as f:
        new_data =  json.load(f)
    with open("final_result_sibei_215_del_split.json",'r') as f:
        new_data_1 =  json.load(f)
    all_hi_ai, all_rl_ai, all_rl_hai, all_autosklearn = [],[],[],[]
    rf_hi_ai, rf_rl_ai, rf_rl_hai, rf_autosklearn = [],[],[],[]
    dt_hi_ai, dt_rl_ai, dt_rl_hai, dt_autosklearn = [],[],[],[]
    knn_hi_ai, knn_rl_ai, knn_rl_hai, knn_autosklearn = [],[],[],[]
    lr_hi_ai, lr_rl_ai, lr_rl_hai, lr_autosklearn = [],[],[],[]
    svc_hi_ai, svc_rl_ai, svc_rl_hai, svc_autosklearn = [],[],[],[]
    model = ['RandomForestClassifier', 'DecisionTreeClassifier', 'KNeighborsClassifier', 'LogisticRegression', 'SVC']
    hi_list = [rf_hi_ai, dt_hi_ai, knn_hi_ai, lr_hi_ai, svc_hi_ai]
    rl_list = [rf_rl_ai, dt_rl_ai, knn_rl_ai, lr_rl_ai, svc_rl_ai]
    rl_hai_list = [rf_rl_hai, dt_rl_hai, knn_rl_hai, lr_rl_hai, svc_rl_hai]
    autosklearn_list = [rf_autosklearn, dt_autosklearn, knn_autosklearn, lr_autosklearn, svc_autosklearn]
    
    notebooks = [0,0,0,0,0]
    for task in new_data:
        if task not in new_data_1:
            continue
        model_index = model.index(new_data[task]['model_type'])
        himean = 0
        rlmean = 0
        autosklearnmean = 0

        count = 0
        if len(new_data_1[task]['notebook_list']) == 0:
            continue
        for notebook_id in new_data[task]['notebook_list']:
            if notebook_id not in new_data_1[task]['notebook_list']:
                continue
            notebook = new_data[task]['notebook_list'][notebook_id]
            himean += notebook['hi']
            rlmean += notebook['rl_hai']
            autosklearnmean += notebook['ai-autosklearn']
            count += 1
            notebooks[model_index] += 1

        himean /= count
        rlmean /= count
        autosklearnmean /= count

        hi_list[model_index].append(himean)
        rl_list[model_index].append(new_data[task]['rl_ai'])
        rl_hai_list[model_index].append(rlmean)
        autosklearn_list[model_index].append(autosklearnmean)

        all_hi_ai.append(himean)
        all_rl_ai.append(new_data[task]['rl_ai'])
        all_rl_hai.append(rlmean)
        all_autosklearn.append(autosklearnmean)

    all_hi_ai = np.array(all_hi_ai)
    print('all_hi_ai', all_hi_ai.mean())
    all_rl_ai = np.array(all_rl_ai)
    print('all_rl_ai', all_rl_ai.mean())
    all_rl_hai = np.array(all_rl_hai)
    print('all_rl_hai', all_rl_hai.mean())
    all_autosklearn = np.array(all_autosklearn)
    print('all_autosklearn', all_autosklearn.mean())
    print('all',len(all_rl_hai))

    rf_hi_ai = np.array(rf_hi_ai)
    print('rf_hi_ai', rf_hi_ai.mean())
    rf_rl_ai = np.array(rf_rl_ai)
    print('rf_rl_ai', rf_rl_ai.mean())
    rf_rl_hai = np.array(rf_rl_hai)
    print('rf_rl_hai', rf_rl_hai.mean())
    rf_autosklearn = np.array(rf_autosklearn)
    print('rf_autosklearn', rf_autosklearn.mean())
    print('rf',len(rf_autosklearn))


    dt_hi_ai = np.array(dt_hi_ai)
    print('dt_hi_ai', dt_hi_ai.mean())
    dt_rl_ai = np.array(dt_rl_ai)
    print('dt_rl_ai', dt_rl_ai.mean())
    dt_rl_hai = np.array(dt_rl_hai)
    print('dt_rl_hai', dt_rl_hai.mean())
    dt_autosklearn = np.array(dt_autosklearn)
    print('dt_autosklearn', dt_autosklearn.mean())
    print('dt',len(dt_autosklearn))

    knn_hi_ai = np.array(knn_hi_ai)
    print('knn_hi_ai', knn_hi_ai.mean())
    knn_rl_ai = np.array(knn_rl_ai)
    print('knn_rl_ai', knn_rl_ai.mean())
    knn_rl_hai = np.array(knn_rl_hai)
    print('knn_rl_hai', knn_rl_hai.mean())
    knn_autosklearn = np.array(knn_autosklearn)
    print('knn_autosklearn', knn_autosklearn.mean())
    print('knn',len(knn_autosklearn))

    lr_hi_ai = np.array(lr_hi_ai)
    print('lr_hi_ai', lr_hi_ai.mean())
    lr_rl_ai = np.array(lr_rl_ai)
    print('lr_rl_ai', lr_rl_ai.mean())
    lr_rl_hai = np.array(lr_rl_hai)
    print('lr_rl_hai', lr_rl_hai.mean())
    lr_autosklearn = np.array(lr_autosklearn)
    print('lr_autosklearn', lr_autosklearn.mean())
    print('lr',len(lr_autosklearn))

    svc_hi_ai = np.array(svc_hi_ai)
    print('svc_hi_ai', svc_hi_ai.mean())
    svc_rl_ai = np.array(svc_rl_ai)
    print('svc_rl_ai', svc_rl_ai.mean())
    svc_rl_hai = np.array(svc_rl_hai)
    print('svc_rl_hai', svc_rl_hai.mean())
    svc_autosklearn = np.array(svc_autosklearn)
    print('svc_autosklearn', svc_autosklearn.mean())
    print('svc',len(svc_autosklearn))
    print(model)
    print(notebooks)

if __name__ == '__main__':
    # look_final_planb()
    # generate_new_data()
    # generate_new_data_clean()
    step1res()
    step2res()
    # look_planB_cross()
    # table3()
    # look_validaiton_test()
    # stat_same()
    # look_task()
    # look_task()
    # look_svc_not_imp()
    # add_time()
    # add_seq_in_json()
    # look_ope()
    # add_related_score()
    # generate_clean_task_planb()
    # add_abs_score()
    # get_deepline()
    # look_can_remove()
    # overview()
    # look_rf()
    # look_svcrf()
    # look_all_task()
    # fix_dataset()
    # compare_firefly()
    # look_planB_train()
    # look_sequence_between_task()
    # look_test_seq()
    # look_top_3()
    # look_val()
    # last_end()
    # add_related_score()
    # look_task_mean()
    # look_1600()
    # look_score("furkaneris_heart-disease-logistic-regression")
    
    # max_index_1600()
    # duibi()
    # max_index_800("test_error.txt")
    # with open('max_index_new.json','a',encoding='utf8')as f1:
    #     json.dump(max_index,f1,ensure_ascii=False)
    # notebook_list=find_same() 
    #3040个
    # print(len(notebook_list))
    # print(notebook_list)
    # kan()
    # Human_score()
    # deepline_score()
    # modify_hai_1600()
    # hai_score("all_s.txt")
    # max_index_800("test_error.txt")
    # modify_hai("test_error.txt")
    # res_error = {}
    # for notebook_id in score_dic:
    #     if score_dic[notebook_id]['human'] > score_dic[notebook_id]['origin']:
    #         res_error[notebook_id] = {}
    #         res_error[notebook_id] = score_dic[notebook_id]
    # print(len(res_error))
    # with open('score_error.json','a',encoding='utf8')as fp:
    #     json.dump(res_error,fp,ensure_ascii=False)
    # with open('max_index.json','a',encoding='utf8')as f1:
    #     json.dump(max_index,f1,ensure_ascii=False)
    # with open('analyze_score_new_1600.json','a',encoding='utf8')as fp:
    #     json.dump(score_dic,fp,ensure_ascii=False)
    # human = []
    # deepline = []
    # hai = []
    # for notebook_id in score_dic:
    #     human.append(score_dic[notebook_id]['human'])
    #     deepline.append(score_dic[notebook_id]['deepline'])
    #     hai.append(score_dic[notebook_id]['hai'])
    # with open('score_1.json','r',encoding='utf8')as fp:
    #     json_data = json.load(fp)
    #     for notebook_id in json_data:
    #         human.append(json_data[notebook_id]['human'])
    #         deepline.append(json_data[notebook_id]['deepline'])
    #         hai.append(json_data[notebook_id]['hai'])
    
    # print(np.quantile(human, (25, 50, 75), interpolation='midpoint'))
    # print(np.quantile(deepline, (25, 50, 75), interpolation='midpoint'))
    # print(np.quantile(hai, (25, 50, 75), interpolation='midpoint'))
    # print(np.mean(human)," ",np.mean(deepline)," ",np.mean(hai))


