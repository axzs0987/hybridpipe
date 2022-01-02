import numpy as np
from matplotlib import pyplot as plt
from config import Config
import pprint
import torch
import json
import os
config = Config()
def get_all_model_res():
    res_dic = {}
    filename = '41_test_reward_dic.npy'
    file_dic = np.load(filename, allow_pickle=True).item()
    filename1 = '41_no_model_test_reward_dic.npy'
    file_dic1 = np.load(filename1, allow_pickle=True).item()
    maxindex = np.argmax(file_dic[41])
    maxindex1 = np.argmax(file_dic1[41])
    maxvalue = file_dic[41][maxindex]
    maxvalue1 = file_dic1[41][maxindex1]
    print(config.test_classification_task_dic[41],(file_dic[41][0],file_dic1[41][0],maxindex, maxindex1, maxvalue, maxvalue1))
    for i in range(41,57):
        filename = str(i)+'_model_test_reward_dic.npy'
        file_dic = np.load(filename, allow_pickle=True).item()
        filename1 = str(i)+'_no_model_test_reward_dic.npy'
        file_dic1 = np.load(filename1, allow_pickle=True).item()
        maxindex = np.argmax(file_dic[i])
        maxindex1 = np.argmax(file_dic1[i])
        maxvalue = file_dic[i][maxindex]
        maxvalue1 = file_dic1[i][maxindex1]
        print(i,config.classification_task_dic[i],(file_dic[i][0],file_dic1[i][0],maxindex, maxindex1, maxvalue, maxvalue1))

def plt_all_trained():
    # result_dic = np.load('many_test_reward_dic.npy', allow_pickle=True).item()
    result_dic1 = np.load('no_many_test_reward_dic.npy', allow_pickle=True).item()
    delta = []
    mean_delta = []
    for i in result_dic1:
        mean_delta.append(np.array(result_dic1[i]).mean())
    print(mean_delta)
    
    # for i in result_dic:
    #     delta.append(result_dic[i][0]-result_dic1[i][0])
    # plt.plot(mean_delta)
    # plt.plot(np.zeros(16))
    # ------------------------------------
    # fig = plt.figure(figsize=(10,6), facecolor = 'gray')
    # ax1 = fig.add_subplot(2,2,1)
    # for taskid in result_dic:
    #     plt.plot(result_dic[taskid], label=taskid)
    # ax1 = fig.add_subplot(2,2,2)
    # for taskid in result_dic:
    #     plt.plot(result_dic1[taskid], label=taskid)
    # plt.grid()
    # plt.show()
def plt_plot():
    result_dic = np.load('56_model_test_reward_dic.npy', allow_pickle=True).item()
    plt.plot(result_dic[56])
    plt.show()

def look_reward():
    result_dic = np.load('result_log.npy', allow_pickle=True).item()['reward_dic']
    print(result_dic)
    for i in range(1,2):
        if i in result_dic:
            print(result_dic[i])
            plt.plot(result_dic[i])
    # plt.plot(result_dic[6])
    plt.show()

def look_seq_log():
    result_dic = np.load(config.loss_log_file_name, allow_pickle=True).item()['seq_log']
    print(result_dic.keys())
    
    for i in range(41,57):
        if i in result_dic:
            pprint.pprint(result_dic[i])
            # plt.plot(result_dic[i])
    # plt.show()

def look_all_mean():
    f3 = [0.62,
            1,
            1,
            1,
            0.99,
            0.68,
            0.97,
            0.55,
            0.98,
            0.64,
            1,
            0.77,
            0.51,
            0.21,
            # 0.73,
            # 0.9,
            0.96,
            0.87,
            0.97,
            0.98,
            0.76,
            0.92,
            0.84,
            0.89,
            0.55,
            0.72,
            0.99,
            0.95,
            1,
            0.50,
            0.63,
            0.65,
            0.91,
            0.99,
            0.87,
            0.88,
            0.87,
            0.81,
            0.90,
            0.71,
            0.62,
            0.63,
            # 0.60,
            # 0.92,
            0.78,
            0.54,
            0.94,
            0.68,
            0.61,
            0.66,
            0.98,
            0.85,
            0.74,
            0.75,
            1,
            0.99,
        ]
    f4 = [0.59,
            1,
            1,
            1,
            0.99,
            0.73,
            1,
            0.56,
            0.98,
            0.66,
            1,
            0.78,
            0.28,
            0.21,
            ####
            # 0.82,
            # 0.91,
            0.95,
            0.88,
            0.72,
            1,
            0.72,
            0.92,
            0.98,
            0.87,
            0.53,
            0.81,
            0.99,
            0.95,
            ###
            1,
            0.50,
            0.63,
            0.61,
            0.91,
            0.98,
            0.87,
            0.94,
            0.87,
            0.81,
            0.82,
            0.78,
            0.68,
            0.64,
            ###
            # 0.49,
            # 0.92,
            0.77,
            0.52,
            0.98,
            0.69,
            0.68,
            0.68,
            0.99,
            0.86,
            0.76,
            0.71,
            0.94,
            0.98,
        ]
    print(len(f4))
    print(f4[0:14])
    print(f4[14:28])
    print(f4[28:42])
    print(f4[42:56])
    # print(f4[56])
    mean = np.array(f4).sum()/52
    print(mean)

def look_max_reward():
    result_dic = np.load('result_log.npy', allow_pickle=True).item()['max_reward']
    print(result_dic.keys())
    
    for i in range(0,40):
        if i in result_dic:
            print(result_dic[i])
            plt.plot(result_dic[i])
    plt.show()

def look_result_log():
    result_dic = np.load(config.result_log_file_name, allow_pickle=True).item()
    print(result_dic.keys())
    print(np.array(result_dic['reward_dic'][52]).max())

def look_max_action():
    result_dic = np.load('result_log.npy', allow_pickle=True).item()['max_action']
    print(result_dic.keys())
    print(result_dic[6])
    # plt.plot(result_dic[0])
    # plt.show()

def look_test_q_value():
    result_dic = np.load(config.test_q_value_file_name, allow_pickle=True).item()
    result_dic_t = np.load(config.test_reward_dic_file_name, allow_pickle=True).item()
    print(result_dic['Encoder'])
    
    for i in range(1,57):
        if i not in result_dic['Encoder']:
            continue
        print('-----------------')
        print(i)
        print(result_dic['Encoder'][i])
        print(result_dic_t[i]['reward'][32:])
        print(len(result_dic_t[i]['reward']))

        # engine:41,selection:41,
        # print(len(result_dic_t[i]['reward']))
    

def look_test_reward():
    with open('/home/yxm/staticfg-master/test_reward_24000_56000.json','r') as f:
        res_dic = json.load(f)
    # result_dic = np.load('/home/yxm/staticfg-master/test_reward_24000_37500.json', allow_pickle=True).item()
# result_dic = np.load(config.single_test_reward_dic_file_name, allow_pickle=True).item()
    # print(result_dic)
    print(res_dic.keys())
    max_mean = [0,0]
    all_max = {}
    mean_list = []

    res = {}
    for task in res_dic:
        if -1 in res_dic[task]['reward'].values():
            continue
        for epoch_id in res_dic[task]['reward']:
            if epoch_id not in res:
                res[epoch_id] = []
            res[epoch_id].append(res_dic[task]['reward'][epoch_id])
            # break
        # break
    print(res)
    for epoch_id in res:
        print(epoch_id, np.array(res[epoch_id]).mean())

    # if config.version == 0:
    #     look_range = (1,15)
    # elif config.version == 1:
    #     look_range = (15,29)
    # elif config.version == 2:
    #     look_range = (29,43)
    # elif config.version == 3:
    #     look_range = (43,57)
    
    # for k in range(0,len(result_dic[look_range[1]-1]['reward'])):
    #     mean = 0
    #     # for i in range(43,57):
    
    #     for i in range(look_range[0], look_range[1]):
    #         mean += result_dic[i]['reward'][k]
    #         if i not in all_max:
    #             all_max[i] = 0
    #         if result_dic[i]['reward'][k] > all_max[i]:
    #             all_max[i] = result_dic[i]['reward'][k]
    #     mean /= 14
        # for i in range(1,57):
        #     if i not in range(look_range[0], look_range[1]):
        #         mean += result_dic[i]['reward'][k]
        #         if i not in all_max:
        #             all_max[i] = 0
        #         if result_dic[i]['reward'][k] > all_max[i]:
        #             all_max[i] = result_dic[i]['reward'][k]
            
        #         # plt.plot(result_dic[i]['reward'])
        # mean /= 42
    #     print(k, mean)
    #     if max_mean[1] < mean:
    #         max_mean[1] = mean
    #         max_mean[0] = k
    #     mean_list.append(mean)
    # print(max_mean)
    # plt.plot(mean_list)
    # plt.savefig('test_result_'+str(config.version))
    # print(all_max)
    # for i in range(15,29):
    # for i in range(29,43):
    # for i in range(43,57):
    # for i in range(look_range[0], look_range[1]):
    # # # for i in range(17,29):
    #     print('*******max')
    #     # print(result_dic[i]['seq'][max_mean[0]])
    #     print(result_dic[i].keys())
    #     print(str(i)+'_reward:', '%.3f' % result_dic[i]['reward'][max_mean[0]])
    #     print(str(i)+'_seq:',  result_dic[i]['seq'][max_mean[0]])
    # for i in range(look_range[0], look_range[1]):
    # # # for i in range(17,29):
    #     print('*******last')
    #     print(str(i)+'_reward:',result_dic[i]['reward'][len(result_dic[look_range[1]-1]['reward'])-1])
    #     print(str(i)+'_seq:',result_dic[i]['seq'][len(result_dic[look_range[1]-1]['reward'])-1]])
def look_one_test_reward():
    result_dic = np.load(config.test_reward_dic_file_name, allow_pickle=True).item()
    # print(result_dic.keys())
    # print(result_dic)
    for i in range(42,43):
        if i in result_dic:
            # pprint.pprint(result_dic[i]['pipline'])
            # print(i, str(np.array(result_dic[i]['reward']).max()) + ', ' + str(result_dic[i]['reward'][-1]))
            plt.ylim(0.4,0.7)
            plt.plot(result_dic[i]['reward'])
            plt.savefig('one_test.jpg')

def look_test_loss():
    result_dic = np.load(config.test_loss_log_file_name, allow_pickle=True).item()
    for i in range(1,57):
        print(result_dic[5][i][-1].detach().numpy())
        plt.plot(result_dic[5][i][-1].detach().numpy())
    plt.savefig('test_loss.jpg')


def pg_look_loss():
    result_dic = np.load(config.loss_log_file_name, allow_pickle=True).item()
    print(len(result_dic[0]))
    
    # plt.plot(result_dic[0])
    # one_file = open("loss0.txt",'w+')
    # count = 0
    # for i in range(len(result_dic[0])):
    #     if count%100 == 0:
    #         one_file.write(str(count+1))
    #         one_file.write(' ')
    #         one_file.write(str(result_dic[0][i]))
    #         one_file.write('\n')
    #     count += 1
    # one_file.close()
    # plt.savefig('loss0.jpg')
    # plt.cla()

    # #######
    # plt.plot(result_dic[2])
    # one_file = open("loss2.txt",'w+')
    # count = 0
    # for i in range(len(result_dic[2])):
    #     if count%100 == 0:
    #         one_file.write(str(count+1))
    #         one_file.write(' ')
    #         one_file.write(str(result_dic[2][i]))
    #         one_file.write('\n')
    #     count += 1
    # one_file.close()
    # plt.savefig('loss2.jpg')
    # plt.cla()

    # #######
    # one_file = open("loss3.txt",'w+')
    # count = 0
    # for i in range(len(result_dic[3])):
    #     if count%100 == 0:
    #         one_file.write(str(count+1))
    #         one_file.write(' ')
    #         one_file.write(str(result_dic[3][i]))
    #         one_file.write('\n')
    #     count += 1
    # one_file.close()
    # plt.plot(result_dic[3])
    # plt.savefig('loss3.jpg')
    # plt.cla()


    # #######
    # one_file = open("loss4.txt",'w+')
    # count = 0
    # for i in range(len(result_dic[4])):
    #     if count%100 == 0:
    #         one_file.write(str(count+1))
    #         one_file.write(' ')
    #         one_file.write(str(result_dic[4][i]))
    #         one_file.write('\n')
    #     count += 1

    # plt.plot(result_dic[4])
    # plt.savefig('loss4.jpg')
    # plt.cla()

    # ####
    # one_file = open("loss5.txt",'w+')
    # count = 0
    # for i in range(len(result_dic[5])):
    #     if count%100 == 0:
    #         one_file.write(str(count+1))
    #         one_file.write(' ')
    #         one_file.write(str(result_dic[5][i]))
    #         one_file.write('\n')
    #     count += 1
    # plt.plot(result_dic[5])
    # plt.savefig('loss5.jpg')
    # plt.cla()

    # result_dic_1 = np.load(config.lp_loss_log_file_name, allow_pickle=True)
    # one_file = open("loss_lp.txt",'w+')
    # count = 0
    # for i in range(len(result_dic_1)):
    #     if count%100 == 0:
    #         one_file.write(str(count+1))
    #         one_file.write(' ')
    #         one_file.write(str(result_dic_1[i]))
    #         one_file.write('\n')
    #     count += 1
    # plt.plot(result_dic_1)
    # plt.savefig('loss_lp.jpg')
    # plt.cla()

    # plt.show()

def look_loss():
    result_dic = np.load(config.loss_log_file_name, allow_pickle=True).item()
    # print(result_dic)
    result_dic_1 = np.load(config.lp_loss_log_file_name, allow_pickle=True)
    x = []
    x0 = []
    x2 = []
    x3 = []
    x4 = []
    x5 = []
    count = 0
    for k in result_dic_1:
        x.append(count*6)
        count += 1
    count = 0
    for k in result_dic[0]:
        x0.append(count*6)
        count += 1
    count = 0
    for k in result_dic[2]:
        x2.append(count*6)
        count += 1
    count = 0
    for k in result_dic[3]:
        x3.append(count*6)
        count += 1
    count = 0
    for k in result_dic[4]:
        x4.append(count*6)
        count += 1
    count = 0
    for k in result_dic[5]:
        x5.append(count*6)
        count += 1

    plt.figure(figsize=(15, 6.5))
    plt.plot(x0, result_dic[0], linewidth = 2, color = 'orange')
    plt.title('Loss Trend of Numerical Imputer agent')
    plt.xlabel('Train Step')
    plt.ylabel('Loss')
    print(len(result_dic[0]))
    plt.savefig('loss_2_0.jpg')
    plt.cla()


    plt.plot(x2, result_dic[2], linewidth = 2, color = 'orange')
    plt.title('Loss Trend of Encoder agent')
    plt.xlabel('Train Step')
    plt.ylabel('Loss')
    plt.savefig('loss_2_2.jpg')
    plt.cla()

    plt.plot(x3, result_dic[3], linewidth = 2, color = 'orange')
    plt.title('Loss Trend of Scaler agent')
    plt.xlabel('Train Step')
    plt.ylabel('Loss')
    plt.savefig('loss_2_3.jpg')
    plt.cla()

    plt.plot(x4, result_dic[4], linewidth = 2, color = 'orange')
    plt.title('Loss Trend of Feature Engine agent')
    plt.xlabel('Train Step')
    plt.ylabel('Loss')
    plt.savefig('loss_2_4.jpg')
    plt.cla()

    plt.plot(x5, result_dic[5], linewidth = 2, color = 'orange')
    plt.title('Loss Trend of Feature Selection agent')
    plt.xlabel('Train Step')
    plt.ylabel('Loss')
    plt.savefig('loss_2_5.jpg')
    plt.cla()

    plt.plot(x, result_dic_1, linewidth = 2, color = 'orange')
    plt.title('Loss Trend of Logical Pipeline agent')
    plt.xlabel('Train Step')
    plt.ylabel('Loss')
    plt.savefig('loss_2_6.jpg')
    plt.cla()
    # plt.show()
    
def look_single_loss():
    res = np.load(config.single_loss_log_file_name)
    plt.plot(res)
    plt.savefig('singleloss.jpg')
    # plt.cla()
def get_all_mdel_best():
    res = np.load("1_test.npy")
    print(res)
    # start_number = 25000
    # score_version = {}
    # while(start_number <= 30500):
    #     for version in range(0,4):
    #         if version not in score_version:
    #             score_version[version] = {}
    #         if start_number not in score_version[version]:
    #             score_version[version][start_number] = []
    #         mean_score = 0
    #         res = np.load("rf_test/" + str(version) + "_" + str(start_number) + "_test.npy").item()
    #         for key in res:
    #             mean_score += res[key]
    #         mean_score /= 14
    #         score_version[version][start_number].append(mean_score)
    #     # print(score_version)
    #     start_number += 500
    # with open('rf_score_version.json', 'w') as result_file:
    #     json.dump(score_version, relook_test_rewardsult_file)

def look_improve():
    with open('/home/yxm/staticfg-master/clean_task_no_1_fix_label.json', 'r') as f:
        clean_data = json.load(f)
    notebooks = 0
    res = np.load('logs/0/test_reward.npy', allow_pickle =True).item()
    # print(res.keys())
    tasks = []
    for taskid in res:
        tasks.append(config.classification_task_dic[taskid]['dataset'] + '_RandomForestClassifier_' + config.classification_task_dic[taskid]['label'])
    # print('tasks', tasks)
    ai_better = 0
    hi_better = 0
    equal = 0
    for task in clean_data:
        # print(task)
        if task in tasks:
            print("######")
            # print(clean_data[task]['hi_mean'])
            hi_score = clean_data[task]['base']
            for i in res:
                if config.classification_task_dic[i]['dataset'] in task and config.classification_task_dic[i]['label'] in task:
                    ai_score = res[i]['reward'][-1]

            print(ai_score-hi_score)
            if ai_score > hi_score:
                ai_better += 1
            elif ai_score == hi_score:
                equal += 1
            else:
                hi_better += 1
    print('ai_better', ai_better)
    print('equal', equal)
    print('hi_better', hi_better)
                
        
def look_all_task():
    datasets = []
    csv_file = []
    label = []

    classification_task_dic = {}
    task_id = 0
    # for task in os.listdir(dataset_path):
    #     csv_name = os.listdir(dataset_path + task)[0]
    #     classification_task_dic[task_id]  = {}
    #     classification_task_dic[task_id]['data_path'] = task + '/' + csv_name
    #     classification_task_dic[task_id]['label'] = 
    #     task_id += 1

    with open('/home/yxm/staticfg-master/clean_task_no_1_fix_label_s.json', 'r') as f:
        clean_data = json.load(f)
    notebooks = 0
    for task in clean_data:
        if len(clean_data[task]['notebook_list']) > 0:
            # if clean_data[task]['dataset'] not in datasets:
            classification_task_dic[task_id] = {}
            classification_task_dic[task_id]['dataset'] = clean_data[task]['dataset']
            classification_task_dic[task_id]['csv_file'] = clean_data[task]['dataset_file']
            classification_task_dic[task_id]['label'] = clean_data[task]['label']
            classification_task_dic[task_id]['model'] = clean_data[task]['model_type']
            task_id += 1
            datasets.append(clean_data[task]['dataset'])
            csv_file.append(clean_data[task]['dataset_file'])
            label.append(clean_data[task]['label'])
        notebooks += len(clean_data[task]['notebook_list'])
    with open('jsons/classification_task_dic_no_dup.json', 'w') as f:
        json.dump(classification_task_dic, f)
    print(len(datasets))
    print(len(set(csv_file)))
    print(notebooks)

    # for index,dataset in enumerate(datasets):
    # # for index,dataset in enumerate(os.listdir('datasets')):
    #     if not os.path.exists('datasets/'+dataset):
    #         os.mkdir('datasets/'+dataset)
    #     if len(os.listdir('datasets/'+dataset)) == 0:
    #         print(dataset, csv_file[index])
    #         # os.system('rm -rf datasets/'+dataset)
    #     # print(dataset, csv_file[index])
    #     if os.path.exists('/home/yxm/KGTorrent/dataset/'+dataset):
    #         os.system('cp /home/yxm/KGTorrent/dataset/'+dataset+'/' + csv_file[index].replace('(','\\(').replace(')','\\)').replace(' ', '\\ ') +' ' +'datasets/'+dataset)
    #     else:
    #         os.system('cp /home/datamanager/dataset/statsklearn/dataset/'+dataset+'/' + csv_file[index].replace('(','\\(').replace(')','\\)').replace(' ', '\\ ') +' ' +'datasets/'+dataset)
def generate_test_classification_task_dict():
    with open('/home/yxm/staticfg-master/clean_task_planB.json','r') as f:
        clean_data = json.load(f)
    classification_task_dic = config.classification_task_dic
    test_index = config.test_index
    
    test_names = [classification_task_dic[item]['dataset']+'_'+classification_task_dic[item]['model']+"_"+classification_task_dic[item]['label'] for item in classification_task_dic if item in test_index]
    task_id = 372
    res = {}

    for task in clean_data:
        if task in test_names:
            continue
        res[task_id] = {}
        res[task_id]['dataset'] = clean_data[task]['dataset']
        res[task_id]['csv_file'] = clean_data[task]['dataset_file']
        res[task_id]['label'] = clean_data[task]['label']
        res[task_id]['model'] = clean_data[task]['model_type']
        test_index.append(str(task_id))
        classification_task_dic[task_id] = res[task_id]
        task_id += 1
        

    with open('jsons/classification_task_dic.json','w') as f:
        json.dump(classification_task_dic, f)
    with open('jsons/test_index.json','w') as f:
        json.dump(test_index,f)


def generate_train_classification_task_dic():
    with open("jsons/classification_task_dic.json",'r') as f:
        classification_task_dic = json.load(f)
    with open("jsons/test_classification_task_dict.json",'r') as f:
        test_classification_task_dic = json.load(f)
    res = {}
    for taskid in classification_task_dic:
        if taskid in test_classification_task_dic:
            continue
        res[taskid] = classification_task_dic[taskid]
    with open("jsons/train_classification_task_dic.json",'w') as f:
        json.dump(res, f)
if __name__ == '__main__':
    # pg_look_loss()
    # look_test_reward()
    # generate_test_classification_task_dict()
    generate_train_classification_task_dic()
    # look_all_task()
    # look_improve()
    # look_seq_log()
    # look_single_loss()
    # look_loss()
    # get_all_mdel_best()
    # look_test_q_value()
    # look_all_mean()
    # look_result_log()
    # look_one_test_reward()
    # look_test_loss()
    # pg_look_loss()