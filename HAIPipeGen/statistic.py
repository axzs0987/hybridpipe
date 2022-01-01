import numpy as np
import os,sys
import json
import math
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
    exist_f = open("/home/yxm/staticfg-master/all_s.txt", "r")
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
    exist_f = open("/home/yxm/staticfg-master/all_s.txt", "r")
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
    exist_f = open("/home/yxm/staticfg-master/all_s.txt", "r")
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
    exist_f = open("/home/yxm/staticfg-master/all_s.txt", "r")
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
    exist_f = open("/home/yxm/staticfg-master/all_s.txt", "r")
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
    exist_f = open("/home/yxm/staticfg-master/all_s.txt", "r")
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
    cnt_validation_low_all = 0
    cnt_ai_higher = 0
    cnt_1  = 0
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
            if score_dic[notebook_id]['hi'] == 1.0:
                cnt_1 +=1
            if max_score == score_dic[notebook_id]['origin']:
                cnt_validation_same +=1
            elif max_score < score_dic[notebook_id]['origin']:
                cnt_validation_low +=1
        elif max_score == score_dic[notebook_id]['origin'] :
            cnt_other_validation +=1
        if max_score == score_dic[notebook_id]['origin']:
            cnt_validation_same_all +=1
            if score_dic[notebook_id]['ai-deepline'] > score_dic[notebook_id]['hi']:
                cnt_ai_higher +=1
        elif max_score < score_dic[notebook_id]['origin']:
            cnt_validation_low_all +=1

    print(cnt)
    print(cnt_1)
    # print(cnt_validation_same)
    # print(cnt_validation_low)
    # print(cnt_other_validation)
    # print(cnt_validation_same_all)
    # print(cnt_validation_low_all)
    # print(cnt_ai_higher)
def stat_base():
    with open('task_add_base.json','r') as f:
        task_dic = json.load(f)
    for task in task_dic:
        if os.path.exists('new_data/base_result/'+task+'.npy'):
            note_score = np.load("new_data/base_result/"+task+'.npy', allow_pickle=True).item()
            task_dic[task]['base'] = note_score['accuracy_score']
        for notebook_id in task_dic[task]['notebook_list']:
            if os.path.exists('new_data/ai_seq_res/'+notebook_id+'.npy'):
                note_score = np.load('new_data/ai_seq_res/'+notebook_id+'.npy', allow_pickle=True).item()
                task_dic[task]['notebook_list'][notebook_id]['add_seq'] = note_score['accuracy_score']
    # task_dic.pop('iris-flower-data_SVC_4')
    # task_dic.pop('habermans-survival-data-set_RandomForestClassifier_3')
    # task_dic.pop('iris-flower-dataset_LogisticRegression_8')
    # task_dic.pop('iris-flower-dataset_DecisionTreeClassifier_8')
    with open('task_add_base.json','w')as f:
        json.dump(task_dic,f,ensure_ascii=False)
    # base_path = os.listdir('new_data/base_result')
    # print(len(base_path))
    # ai_seq_path = os.listdir('new_data/ai_seq_res')
    # print(len(ai_seq_path))
    
def task_add_validation_score():
    with open('merge_del_meitu.json','r') as f:
        score_dic = json.load(f)
    exist_f = open("/home/yxm/staticfg-master/all_s.txt", "r")
    exist = exist_f.readlines()
    exitst_ = [x.strip("\n") for x in exist]
    notebooks = list(set(score_dic) & set(exitst_))
    old_path = os.listdir('validation_merged_result_1')
    # for file_ in old_path:

    old_notebooks = list(set(score_dic) & set(old_path))
    for notebook_id in notebooks:
        max_index = -1
        max_score = -1
        hai_score = -1
        max_root = ""
        origin_score = -1
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
                    if note_validation =='origin.npy':
                        origin_score = note_score['accuracy_score']
                
                else:
                    note_score = np.load("new_data/merge_validation_result_new/"+notebook_id+"/"+note_validation, allow_pickle=True).item()
                    print("new_data/merge_validation_result_new/"+notebook_id+"/"+note_validation+" "+str(note_score['accuracy_score']))
                    if note_score['accuracy_score']>max_score:
                        max_index = note_validation
                        max_root = "new_data/merge_max_result_new/"
                        max_score = note_score['accuracy_score']
                    if note_validation =='origin.npy':
                        origin_score = note_score['accuracy_score']

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
                    if note_validation =='origin.npy':
                        origin_score = note_score['accuracy_score']
                else:
                    note_score = np.load("new_data/merge_validation_result/"+notebook_id+"/"+note_validation, allow_pickle=True).item()
                    print("new_data/merge_validation_result/"+notebook_id+"/"+note_validation+" "+str(note_score['accuracy_score']))
                    if note_score['accuracy_score']>max_score:
                        max_index = note_validation
                        max_root = "new_data/merge_max_result/"
                        max_score = note_score['accuracy_score']
                    if note_validation =='origin.npy':
                        origin_score = note_score['accuracy_score']
        score_dic[notebook_id]['max_index'] = max_index
        score_dic[notebook_id]['max_root'] = max_root
        score_dic[notebook_id]['origin'] = origin_score
        score_dic[notebook_id]['max_validation_score'] = max_score
    for notebook_id in old_notebooks:
        origin_ = np.load("validation_prenotebook_res/"+notebook_id+'.npy', allow_pickle=True).item()
        origin_score = origin_['accuracy_score']
        max_score = origin_score
        max_index = 'orgin.npy'
        max_root = "validation_prenotebook_res"
        note_validation_path = os.listdir("validation_merged_result_1/"+notebook_id)
        for note_validation in note_validation_path:
            note_score = np.load("validation_merged_result_1/"+notebook_id+"/"+note_validation, allow_pickle=True).item()
            if note_score['accuracy_score']>max_score:
                max_index = note_validation
                max_root = "validation_merged_result_1/"
                max_score = note_score['accuracy_score']
        score_dic[notebook_id]['max_index'] = max_index
        score_dic[notebook_id]['max_root'] = max_root
        score_dic[notebook_id]['origin'] = origin_score
        score_dic[notebook_id]['max_validation_score'] = max_score
    with open('merge_del_meitu_add_validation.json','w') as f:
        json.dump(score_dic,f,ensure_ascii=False)
def classify():
    # with open('notebook_location.json','r') as f:
    #     note_dic = json.load(f)
    note_dic = {}
    exist_f = open("/home/yxm/staticfg-master/all_s.txt", "r")
    exist = exist_f.readlines()
    notebooks = [x.strip("\n") for x in exist]
    for notebook_id in notebooks:
        note_dic[notebook_id] = "121.199.64.200"
    with open('notebook_location.json','w',encoding='utf8')as f1:
        json.dump(note_dic,f1,ensure_ascii=False)
def cross():
    with open('note_cross.json','r') as f:
        note_dic = json.load(f)
    path = os.listdir('new_data/cross_val_res')
    for notebook_id in path:
        max_cross = -1
        note_validation_path = os.listdir("new_data/cross_val_res/"+notebook_id)
        for note_validation in note_validation_path:
            note_score = np.load("new_data/cross_val_res/"+notebook_id+"/"+note_validation, allow_pickle=True).item()
            note_mean = np.mean(note_score['accuracy_score'])
            if note_mean > max_cross:
                max_cross = note_mean
            if note_validation == 'origin.npy':
                note_dic[notebook_id]['origin_cross'] = note_mean
        note_dic[notebook_id]['max_cross'] = max_cross
    with open('note_cross.json','w',encoding='utf8')as f1:
        json.dump(note_dic,f1,ensure_ascii=False)
def planB_train():
    res ={}
    path = os.listdir('planB_trainres')
    for task in path:
        res[task]={}
        task_path = os.listdir('planB_trainres/'+task)
        path_ =[int(x.strip(".npy")) for x in task_path]
        path_.sort()
        for item in path_:
            item_npy = str(item)+'.npy'
            task_score = np.load("planB_trainres/"+task+"/"+item_npy, allow_pickle=True).item()
            
            res[task][str(item)] = task_score['accuracy_score']
    with open('planB_train.json','w',encoding='utf8')as f1:
        json.dump(res,f1,ensure_ascii=False)
def planB_train_mean():
    with open('planB_train.json','r',encoding='utf8')as f1:
        tasks = json.load(f1)
    with open('planB_train_tasks.json','r',encoding='utf8')as f1:
        trains = json.load(f1)
    res = {}
    for task in tasks:
        # print(task)
        if task not in trains:
            continue
        res[task]=tasks[task]
        temp =[]
        if not res[task]:
            res[task]['mean']=0
            continue
        for item in res[task]:
            temp.append(tasks[task][item])
        res[task]['mean'] = np.mean(temp)
    res.items()
    list_task = list(res.items())
    print(list_task[0])
    # print(list_task)
    list_task.sort(key=lambda i:i[1]['mean'],reverse=False)
    res = dict(list_task)
    # with open('planB_train_sort.json','w',encoding='utf8')as f1:
    #     json.dump(res,f1,ensure_ascii=False)
    print(len(trains))
    print(len(res))
def find_():
    with open('../firefly_seq/jsons/classification_task_dic_no_dup.json', 'r') as f:
        classification_task_dic = json.load(f)
    k_fold = 3
    distribute_num = 9
    distribute_id = 8
    version = 0
    fold_length = math.ceil(len(classification_task_dic)/k_fold)
    test_index = [str(i) for i in range(version*k_fold, min(version*k_fold + fold_length, len(classification_task_dic)))]
    train_index = list(set(classification_task_dic.keys())-set(test_index))

    pao_ =[]
    for task_id in train_index:
        task_name = classification_task_dic[task_id]['dataset']+'_'+classification_task_dic[task_id]['model']+"_"+classification_task_dic[task_id]['label']
        pao_.append(task_name)
    # print(len(pao_))
    with open('planB_train_tasks.json','r',encoding='utf8')as f1:
        trains = json.load(f1)
    with open('planB_test_tasks.json','r',encoding='utf8')as f1:
        test = json.load(f1)    
    with open('planB_train_sort.json','r',encoding='utf8')as f1:
        yipao = json.load(f1)
    with open('planB_runned.json','r',encoding='utf8')as f1:
        runned = json.load(f1)    
    not_pao = list(set(trains)-set(yipao))
    yaopao = list(set(not_pao)&set(pao_))
    buhui = list(set(not_pao)-set(yaopao))
    test_no = list(set(test)-set(runned))
    print(len(test))
    print(len(test_no))
    # print(buhui)
    # print(len(yaopao))
def topk():
    path = os.listdir('planB_crossvalres')
    res ={}
    for task in path:
        res[task]={}
        task_path = os.listdir('planB_crossvalres/'+task)
        path_ =[int(x.strip(".npy")) for x in task_path]
        path_.sort()
        temp = []
        for item in path_:
            item_npy = str(item)+'.npy'
            task_score = np.load("planB_crossvalres/"+task+"/"+item_npy, allow_pickle=True).item()
            res[task][str(item)] = np.mean(task_score['accuracy_score'])
        # print(res[task].items())

        L=list(res[task].items())
        # print(L[0][1])
        L.sort(key=lambda x:x[1],reverse=True)
        # print(L)
        res[task] = dict(L)
    with open('planB_cross.json','w',encoding='utf8')as f1:
        json.dump(res,f1,ensure_ascii=False)    
    # list_task = list(res.items())
    # print(list_task[0][1])
    # for task in list_task:
    #     print(task[1])
def look_planB():
    with open('note_cross_fix_label_add_planB.json','r',encoding='utf8')as f1:
        note_dic = json.load(f1)
    info_triple = np.load("/home/yxm/staticfg-master/merged_info_triple.npy", allow_pickle=True).item()
    res = {}
    rf_notebooks = []
    knn_notebooks = []
    dt_notebooks = []
    lr_notebooks = []
    svc_notebooks = []
    planB = []
    origin = []
    old = []
    higer = 0
    notebooks = []
    for notebook_id in note_dic:
        if "max_planB" in note_dic[notebook_id] and "origin_cross" in note_dic[notebook_id]:
            if info_triple[notebook_id]['model_type'] == 'RandomForestClassifier':
                rf_notebooks.append(notebook_id)
            if info_triple[notebook_id]['model_type'] == 'KNeighborsClassifier':
                knn_notebooks.append(notebook_id)
            if info_triple[notebook_id]['model_type'] == 'LogisticRegression':
                lr_notebooks.append(notebook_id)
            if info_triple[notebook_id]['model_type'] == 'DecisionTreeClassifier':
                dt_notebooks.append(notebook_id)
            if info_triple[notebook_id]['model_type'] == 'SVC':
                svc_notebooks.append(notebook_id)
            notebooks.append(notebook_id)
    
    for notebook_id in notebooks:
        planB.append(note_dic[notebook_id]['max_planB'])
        origin.append(note_dic[notebook_id]['origin_cross'])
        old.append(note_dic[notebook_id]['max_cross'])
        if note_dic[notebook_id]['max_planB'] > note_dic[notebook_id]['origin_cross']:
            higer +=1
    # print('dt')
    print('higher, ',higer,' sum, ',len(notebooks),' higher/sum ', higer/len(notebooks))
    print('origin              old                planB')
    print(np.mean(origin),np.mean(old),np.mean(planB))
def max_index_planB():
    # global score_dic
    # global max_index
    max_index ={}
    with open('note_cross_fix_label_add_planB.json','r') as f:
        note_dic = json.load(f)
    cnt = 0
    path = os.listdir('new_data/planB_cross_val_res')
    
    for notebook_npy in path:
        notebook_id = notebook_npy.split('.npy')[0]
        try:
            note_validation_path = os.listdir("new_data/planB_cross_val_res/"+notebook_id)
            max_score = -1
            max_index[notebook_id] = {}
            if 'origin_cross' in note_dic[notebook_id] and note_dic[notebook_id]['origin_cross'] == note_dic[notebook_id]['origin_cross']:
                max_index[notebook_id]['origin.npy'] = note_dic[notebook_id]['origin_cross']
                max_score = note_dic[notebook_id]['origin_cross']
            # max_index[notebook_id] = 'origin.npy'
            for note_validation in note_validation_path:
                
                note_score = np.load("new_data/planB_cross_val_res/"+notebook_id+"/"+note_validation, allow_pickle=True).item()
                note_mean = np.mean(note_score['accuracy_score'])
                if note_mean != note_mean:
                    continue
                max_index[notebook_id][note_validation] = note_mean
                # if note_mean>max_score:
                #     max_index[notebook_id] = note_validation
                #     max_score = note_mean 
            L=list(max_index[notebook_id].items())
            L.sort(key=lambda x:x[1],reverse=True)
            max_index[notebook_id] = dict(L)
        except:
            cnt+=1
            print(notebook_id)   
    print(cnt)
    
    with open('max_index_planB.json','w',encoding='utf8')as f1:
        json.dump(max_index,f1,ensure_ascii=False)  
def planB():
    with open('clean_task_add_planB.json','r')as f:
        clean_task = json.load(f)
    with open('max_index_planB.json','r')as f:
        max_index = json.load(f)
    for task in clean_task:
        for notebook_id in clean_task[task]['notebook_list']:
            if notebook_id in max_index:
                # print(notebook_id)
                if os.path.exists("new_data/merge_max_result_planB/"+notebook_id) and len(os.listdir("new_data/merge_max_result_planB/"+notebook_id))>0:
                    path = os.listdir("new_data/merge_max_result_planB/"+notebook_id)
                    # note_score = np.load("new_data/merge_max_result_planB/"+notebook_id+"/"+path[0], allow_pickle=True).item()
                    # print(note_score)
                    # clean_task[task]['notebook_list'][notebook_id]['planB'] = note_score['accuracy_score']
                   
                    # clean_task[task]['notebook_list'][notebook_id]['cross_planB'] = max_index[notebook_id][path[0]]
                    clean_task[task]['notebook_list'][notebook_id]['planB_test_index'] = path[0]
                    
                else:
                    print(notebook_id)
    #                 clean_task[task]['notebook_list'][notebook_id]['planB'] = clean_task[task]['notebook_list'][notebook_id]['hi']
    with open('clean_task_add_planB.json','w',encoding='utf8')as f1:
        json.dump(clean_task,f1,ensure_ascii=False)  
def look_bad_planB(notebook_id):
    path = os.listdir('new_data/planB_cross_val_res/'+notebook_id)
    for item in path:
        note_score = np.load("new_data/planB_cross_val_res/"+notebook_id+"/"+item, allow_pickle=True).item()
        print(item,note_score['accuracy_score'])    
def planB_operation():
    with open('clean_task_rl_200.json','r')as f:
        clean_task = json.load(f)
    add3_path = os.listdir('new_data/merge_max_result_planB_add3')
    path = os.listdir('new_data/merge_max_result_planB')
    merge_add3 = os.listdir('planB_test_merge_code_add_rule3')
    merge = os.listdir('planB_test_merge_code')
    ope_dic = {}
    cnt = 0
    for task in clean_task:
        for notebook_id in clean_task[task]['notebook_list']:
            temp_list =[]
            if notebook_id in add3_path:
                cnt+=1
                note_path = os.listdir('new_data/merge_max_result_planB_add3/'+notebook_id)
                index = note_path[0].split('.npy')[0]
                # print(index)
                if index != 'origin':
                    with open('planB_test_merge_code_add_rule3/'+notebook_id+'/'+index+'.json','r')as f:
                        data = json.load(f)
                    # print(data['seq'])
                    for i in range(len(data['seq'])):
                        # print(data['seq'][i]['operator'])
                        ope = data['seq'][i]['operator']
                        temp_list.append(ope)
                        # if ope in ope_dic:
                        #     ope_dic[ope] +=1
                        # else:
                        #     ope_dic[ope] = 1
                    clean_task[task]['notebook_list'][notebook_id]['planB_sequence'] = temp_list
                else:
                    clean_task[task]['notebook_list'][notebook_id]['planB_sequence'] = []
            elif notebook_id in path:
                cnt+=1
                note_path = os.listdir('new_data/merge_max_result_planB/'+notebook_id)
                if len(note_path)==0:
                    clean_task[task]['notebook_list'][notebook_id]['planB_sequence'] = []
                else:
                    
                    print(notebook_id)
                    index = note_path[0].split('.npy')[0]
                    # print(index)
                    if index != 'origin':
                        with open('planB_test_merge_code/'+notebook_id+'/'+index+'.json','r')as f:
                            data = json.load(f)
                        print(data['seq'])
                        for i in range(len(data['seq'])):
                            # print(data['seq'][i]['operator'])
                            ope = data['seq'][i]['operator']
                            temp_list.append(ope)
                        clean_task[task]['notebook_list'][notebook_id]['planB_sequence'] = temp_list
                    else:
                        clean_task[task]['notebook_list'][notebook_id]['planB_sequence'] = []
    #                     ope = data['seq'][i]['operator']
    #                     if ope in ope_dic:
    #                         ope_dic[ope] +=1
    #                     else:
    #                         ope_dic[ope] = 1
    # L = list(ope_dic.items())
    # L.sort()
    # # print(L)
    # for x in L:
    #     print(x)
    # # print(ope_dic) 
    # print(cnt)
    with open('clean_task_rl_200.json','w',encoding='utf8')as f1:
        json.dump(clean_task,f1,ensure_ascii=False)  

def rl_operation():
    with open('clean_task_rl_200.json','r')as f:
        clean_task = json.load(f)
    notebook_list = os.listdir('new_data/merge_max_result_rl')
    merge = os.listdir('rl_test_merge_code')
    ope_dic = {}
    cnt = 0
    for task in clean_task:
        for notebook_id in clean_task[task]['notebook_list']:
            temp_list =[]
            cnt+=1
            
            if notebook_id in notebook_list:
                note_path = os.listdir('new_data/merge_max_result_rl/'+notebook_id)
                if len(note_path)==0:
                    clean_task[task]['notebook_list'][notebook_id]['rl_sequence'] = []
                else:
                    print(notebook_id)
                    index = note_path[0].split('.npy')[0]
                    # print(index)
                    if index != 'origin':
                        with open('rl_test_merge_code/'+notebook_id+'/'+index+'.json','r')as f:
                            data = json.load(f)
                        print(data['seq'])
                        for i in range(len(data['seq'])):
                            # print(data['seq'][i]['operator'])
                            ope = data['seq'][i]['operator']
                            temp_list.append(ope)
                            if ope in ope_dic:
                                ope_dic[ope] +=1
                            else:
                                ope_dic[ope] = 1
                        clean_task[task]['notebook_list'][notebook_id]['rl_sequence'] = temp_list
                    else:
                        clean_task[task]['notebook_list'][notebook_id]['rl_sequence'] = []
                    
    L = list(ope_dic.items())
    L.sort()
    # print(L)
    for x in L:
        print(x)
    # print(ope_dic) 
    print(cnt)
    with open('clean_task_rl_200.json','w',encoding='utf8')as f1:
        json.dump(clean_task,f1,ensure_ascii=False) 


def look_rl_seq():
    task_dic = np.load('/home/chensibei/firefly_seq/logs/0/63000.npy',allow_pickle=True).item()
    with open("clean_task_rl_200.json",'r') as f:
        clean_task = json.load(f)
    with open('../firefly_seq/jsons/classification_task_dic.json','r')as f:
        test_index=json.load(f)
    print(len(task_dic))
    ope_dic = {}
    cnt = 0
    for index in task_dic:
        task = test_index[str(index)]['dataset']+'_'+test_index[str(index)]['model']+'_'+test_index[str(index)]['label']
        if task not in clean_task:
            continue
        for notebook in clean_task[task]['notebook_list']:
            cnt+= 1
            temp_seq = [i for i in task_dic[index]['seq'][63000] if i != 'blank']
            print(temp_seq)
            set_seq = set()
            for x in temp_seq:
                if x =='InteractionFeatures':
                    set_seq.add('PolynomialFeatures')
                else:
                    set_seq.add(x)
            print(set_seq)
            for x in set_seq:
                if x in ope_dic:
                    ope_dic[x]+=1
                else:
                    ope_dic[x] = 1
        # print(temp_seq)
    L = list(ope_dic.items())
    L.sort()
    # print(L)
    print('25500 model')
    for x in L:
        print(x)
    # print(ope_dic) 
    print(cnt)
def max_index_rl():
    max_index ={}
    with open('note_cross_fix_label_add_planB.json','r') as f:
        note_dic = json.load(f)
    cnt = 0
    path = os.listdir('new_data/rl_cross_val_res')
    
    for notebook_npy in path:
        notebook_id = notebook_npy.split('.npy')[0]
        note_validation_path = os.listdir("new_data/rl_cross_val_res/"+notebook_id)
        max_score = -1
        max_index[notebook_id] = {}
        try:
            if 'origin_cross' in note_dic[notebook_id] and note_dic[notebook_id]['origin_cross'] == note_dic[notebook_id]['origin_cross']:
                max_index[notebook_id]['origin.npy'] = note_dic[notebook_id]['origin_cross']
                max_score = note_dic[notebook_id]['origin_cross']
            
            
        except:
            cnt+=1
            print(notebook_id) 
        
        # max_index[notebook_id] = 'origin.npy'
        for note_validation in note_validation_path:
            
            note_score = np.load("new_data/rl_cross_val_res/"+notebook_id+"/"+note_validation, allow_pickle=True).item()
            note_mean = np.mean(note_score['accuracy_score'])
            if note_mean != note_mean:
                continue
            max_index[notebook_id][note_validation] = note_mean
            # if note_mean>max_score:
            #     max_index[notebook_id] = note_validation
            #     max_score = note_mean 
        L=list(max_index[notebook_id].items())
        L.sort(key=lambda x:x[1],reverse=True)
        max_index[notebook_id] = dict(L)
          
    print(cnt)
    
    with open('max_index_rl.json','w',encoding='utf8')as f1:
        json.dump(max_index,f1,ensure_ascii=False)
def rl():
    with open('clean_task_rl_200.json','r')as f:
        clean_task = json.load(f)
    with open('max_index_rl.json','r')as f:
        max_index = json.load(f)
    for task in clean_task:
        for notebook_id in clean_task[task]['notebook_list']:
            if notebook_id in max_index:
                # print(notebook_id)
                if os.path.exists("new_data/merge_max_result_rl/"+notebook_id) and len(os.listdir("new_data/merge_max_result_rl/"+notebook_id))>0:
                    path = os.listdir("new_data/merge_max_result_rl/"+notebook_id)
                    note_score = np.load("new_data/merge_max_result_rl/"+notebook_id+"/"+path[0], allow_pickle=True).item()
                    # print(note_score)
                    clean_task[task]['notebook_list'][notebook_id]['rl'] = note_score['accuracy_score']
                   
                    # clean_task[task]['notebook_list'][notebook_id]['cross_planB'] = max_index[notebook_id][path[0]]
                    # clean_task[task]['notebook_list'][notebook_id]['planB_test_index'] = path[0]
                    
                else:
                    print(notebook_id)
    #                 clean_task[task]['notebook_list'][notebook_id]['planB'] = clean_task[task]['notebook_list'][notebook_id]['hi']
    with open('clean_task_rl_200.json','w',encoding='utf8')as f1:
        json.dump(clean_task,f1,ensure_ascii=False)    
def rl_ai_operation():
    with open("clean_task_rl_200.json",'r') as f:
        clean_task = json.load(f)
    with open('firefly_seq_56000.json','r')as f:
        firefiy_seq = json.load(f)
    ope_dic = {}
    cnt = 0
    for task in clean_task:
        for notebook in clean_task[task]['notebook_list']:
            cnt+=1
            ope = firefiy_seq[task]
            for x in ope:
                if x in ope_dic:
                    ope_dic[x] +=1
                else:
                    ope_dic[x] = 1
    L = list(ope_dic.items())
    L.sort()
    # print(L)
    for x in L:
        print(x)
    # print(ope_dic) 
    print(cnt)
def add_del_score():
    with open("final_result_sibei_215.json",'r') as f:
        clean_task = json.load(f)
    path = os.listdir('/home/chensibei/staticfg-master/del_data/merge_max_result_rl')
    cnt = 0
    for task in clean_task:
        for notebook_id in clean_task[task]['notebook_list']:
            if notebook_id not in path:
                continue
            note_validation_path = os.listdir('/home/chensibei/staticfg-master/del_data/merge_max_result_rl/'+notebook_id)
            if len(note_validation_path)==0:
                print(notebook_id)
                continue
            # print(note_validation_path)
            note_score = np.load("/home/chensibei/staticfg-master/del_data/merge_max_result_rl/"+notebook_id+"/"+note_validation_path[0], allow_pickle=True).item()
            if note_score['accuracy_score']> clean_task[task]['notebook_list'][notebook_id]['rl_hai']:
                cnt +=1
                clean_task[task]['notebook_list'][notebook_id]['rl_hai'] = note_score['accuracy_score']
    print(cnt)
    with open('final_result_sibei_hole.json','w',encoding='utf8')as f1:
        json.dump(clean_task,f1,ensure_ascii=False) 
if __name__ == '__main__':
    add_del_score()
    # rl_ai_operation()
    # rl()
    # max_index_rl()
    # look_rl_seq()
    # planB_operation()
    # rl_operation()
    # max_index_planB()
    # planB()
    # max_index_planB()
    # look_planB()
    # topk()
    # find_()
    # planB_train_mean()
    # planB_train()
    # cross()
    # classify()
    # task_add_validation_score()
    # stat_base()
    # stat_same()
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
    
    # print(np.percentile(human, (25, 50, 75), interpolation='midpoint'))
    # print(np.percentile(deepline, (25, 50, 75), interpolation='midpoint'))
    # print(np.percentile(hai, (25, 50, 75), interpolation='midpoint'))
    # print(np.mean(human)," ",np.mean(deepline)," ",np.mean(hai))