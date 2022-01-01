import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import json
import pprint
import math
import os
# plt.style.use('seaborn-dark-palette')
plt.rc('font', family='STIXGeneral', weight='bold')

# color_list = sns.color_palette()
color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
color_list1 = ['tab:brown','tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
hatches = ['/', '\\', '-', '.', '+']
def look_font():
    a = sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])
    for i in a:
        print(i)


def bad_case():
    with open('clean_task.json', 'r') as f:
        clean_data = json.load(f)
    add_seq_worse = 0
    add_seq_worse1 = 0
    bad_all_ = 0
    bad_all_same = 0
    res = []
    all_ = 0

    all_imp = 0
    some_imp = 0
    all_not_imp = 0
    high_base = 0
    high_human = 0
    hibetteraddseq = 0

    origin = 0
    for task in clean_data:
        not_imp = 0
        imp = 0
        # if clean_data[task]['model'] != 'RandomForestClassifier':
        # if clean_data[task]['model'] != 'DecisionTreeClassifier':
        # if clean_data[task]['model'] != 'KNeighborsClassifier':
        # if clean_data[task]['model'] != 'LogisticRegression':
        # if clean_data[task]['model'] != 'SVC':
            # continue
        
        for notebook in clean_data[task]['notebook_list']:
            # print(notebook)
            notebook =  clean_data[task]['notebook_list'][notebook]
            all_ += 1
           
            if notebook['hai'] <= notebook['hi']:
                if notebook['max_validation_score'] > notebook['origin']:
                    origin += 1
                if clean_data[task]['base'] > 0.98:
                    high_base += 1
                if notebook['hi'] - notebook['add_seq'] > 0.5:
                    high_human += 1
                if notebook['hi'] >= notebook['add_seq']:
                    hibetteraddseq += 1
                # if notebook['hi'] - notebook['add_seq'] > 0.5:
                #     high_human += 1
    
                not_imp += 1
                if notebook['add_seq'] < clean_data[task]['base']:
                    add_seq_worse += 1
                if notebook['add_seq'] < notebook['hi']:
                    add_seq_worse1 += 1

                
                if notebook['hai'] == notebook['hi']:
                    bad_all_same += 1
                notebook['base'] = clean_data[task]['base']
                notebook['task'] = task
    
                res.append(notebook)
                bad_all_ += 1
            else:
                imp += 1
        if not_imp == 0:
            all_imp += 1
        elif not_imp < len(clean_data[task]['notebook_list']):
            some_imp += 1
        else:
            # print('imp', imp)
            all_not_imp += 1
    print('add_seq_worse', add_seq_worse)
    print('add_seq_worse1', add_seq_worse1)
    print("all_", all_)
    print('bad_all_same', bad_all_same)
    print('bad_all_', bad_all_)
    print('all_imp', all_imp)
    print('some_imp', some_imp)
    print('all_not_imp', all_not_imp)
    print('high_base', high_base)
    print('high_human', high_human)
    print('hibetteraddseq', hibetteraddseq)
    print('origin', origin)
    res = sorted(res, key=lambda x: x['hai']-x['hi'], reverse=True)
    # pprint.pprint(res)
def look_bad_case():
    with open('clean_task.json', 'r') as f:
        clean_data = json.load(f)
    x = []
    for task in clean_data:
        if clean_data[task]['model'] != 'RandomForestClassifier':
        # if clean_data[task]['model'] != 'DecisionTreeClassifier':
        # if clean_data[task]['model'] != 'KNeighborsClassifier':
        # if clean_data[task]['model'] != 'LogisticRegression':
        # if clean_data[task]['model'] != 'SVC':
            continue
        for notebook in clean_data[task]['notebook_list']:
            
            # print(notebook)
            if notebook['hai'] <= notebook['hi']:
                x.append(clean_data[task]['base'])
                line_x = [clean_data[task]['base'], clean_data[task]['base']]
                line_connect = [notebook['hi'],notebook['hai']]
                plt.scatter([clean_data[task]['base']], [notebook['add_seq']], c='b',s=0.5)
                plt.plot(line_x, line_connect, c='r',marker='3')
    plt.plot(x,x,c='y')
    plt.savefig('look_bad_case_rf.png')

def look_bad_case_how_bad():
    with open('clean_task.json', 'r') as f:
        clean_data = json.load(f)
    x = []
    data_set_list = []
    for task in clean_data:
        for notebook in clean_data[task]['notebook_list']:
            # print(notebook)
            notebook = clean_data[task]['notebook_list'][notebook]
            if notebook['hai'] > notebook['hi']:
                # x_point = notebook['hi']-notebook['base']
                # x_point = clean_data[task]['data_index']
                x_point = clean_data[task]['base']
                # x_point = task
                notebook['base'] = clean_data[task]['base']
                x.append(clean_data[task]['base'])
                line_x = [x_point, x_point]
                line_connect = [notebook['hi'],notebook['hai']]
                # line_connect = [notebook['hi'],notebook['add_seq']]
                # line_connect = [notebook['ai-deepline'],notebook['add_seq']]
                plt.scatter([x_point], [clean_data[task]['base']], c='y',s=0.5)
                plt.scatter([x_point], [notebook['add_seq']], c='b',s=0.5)
                if notebook['hai'] < notebook['hi']:
                # if notebook['ai-deepline'] < notebook['add_seq']:
                    plt.plot(line_x, line_connect, c='pink',marker='3')
                else:
                    plt.plot(line_x, line_connect, c='orange',marker='3')
                # plt.plot(x,x,c='y')
    # plt.savefig('look_bad_case_hi_addseq.png')
    plt.savefig('look_good_case.png')

def look_scatter_line():
    with open('clean_task.json', 'r') as f:
        clean_data = json.load(f)

    x = []
    fig = plt.figure()
    num = 0
    for task in clean_data:
        if clean_data[task]['hi_mean'] >= clean_data[task]['hai_mean']:
            print('########')
            # pprint.pprint(clean_data[task])
        # print(clean_data[task]['model'])
        # if clean_data[task]['model'] == 'RandomForestClassifier':
        # if clean_data[task]['model'] == 'DecisionTreeClassifier':
        # if clean_data[task]['model'] == 'KNeighborsClassifier':
        # if clean_data[task]['model'] == 'LogisticRegression':
        if clean_data[task]['model'] == 'SVC':
            x.append(clean_data[task]['base'])
            line_x = [clean_data[task]['base'], clean_data[task]['base']]
            line_hai = [clean_data[task]['hai_mean']-clean_data[task]['hai_std'],clean_data[task]['hai_mean']+clean_data[task]['hai_std']]
            line_hi = [clean_data[task]['hi_mean']-clean_data[task]['hi_std'],clean_data[task]['hi_mean']+clean_data[task]['hi_std']]
            line_connect = [clean_data[task]['hi_mean'],clean_data[task]['hai_mean']]

            # plt.scatter([clean_data[task]['base']], [clean_data[task]['ai_seq_mean']], c='gray')
            # plt.plot(line_x, line_hi, c='b',marker='3')
            # plt.plot(line_x, line_hai, c='g',marker='3')
            if clean_data[task]['hi_mean'] < clean_data[task]['hai_mean']:
                # plt.plot(line_x, line_connect, c='green')
                continue
            else:
                num += 1
                plt.plot(line_x, line_connect, c='red')

        # plt.scatter([clean_data[task]['base']], clean_data[task]['hi_mean'], c='pink')
        # plt.scatter([clean_data[task]['base']], clean_data[task]['hai_mean'], c='orange')
    plt.plot(x,x,c='y')
    plt.savefig('look_scatter_line_svm.png')
    print('num', num)
def hist_randomadd():
    tick_size = 30
    legend_size = 20
    title_size = 30
    ylabel_size = 30
    title_pad = 40
    label_pad = 15
    lagend_ncol = 5

    hatches = ['\\','-', '/']
    labels = ['Imputer', 'Encoder', 'Scaler', 'FeatureEngine', 'FeatureSelect']
    
    improved = [205, 170, 229, 225, 283]
    # not_improved = [357, 312, 256, 523, 474]
    not_changed = [215, 187, 114, 236, 253]
    reduced = [129, 110, 121, 262, 197]
   
    fig,ax = plt.subplots(figsize=(12,10))
    # ax = plt.figure(figsize=(10,50))
    
    ax.bar(labels, improved,label='Improved', hatch=hatches[0])
    ax.bar(labels,not_changed,bottom=improved,label='Not-Changed', hatch=hatches[1])
    ax.bar(labels,reduced,bottom=[improved[i]+not_changed[i] for i in range(len(improved))],label='Reduced', hatch=hatches[2])

    ax.set_ylabel('# Notebooks', fontsize=title_size, labelpad = label_pad, fontweight='bold')
    # ax.set_ylabel('# Notebooks')
    # ax.set_title('Number Of Noteooks With Operation Add Randomly')
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    ax.legend(prop={'weight':'bold', 'size': legend_size})
    plt.tick_params(labelsize=tick_size)
    plt.subplots_adjust(bottom=0.28, left=0.15, right=0.9, top=0.95)
    plt.savefig('hist_randomadd.eps')


def boxplot():
    tick_size = 45
    legend_size = 38
    title_size = 50
    ylabel_size = 50
    title_pad = 40
    label_pad = 35
    lagend_ncol = 5

    hatches = ['/', '\\', '-', '.', '+']

    fig = plt.figure(figsize=(20,17))
    plt.ylim(0, 1)
    plt.tick_params(labelsize=tick_size)
    plt.title("Accuracy of Pipelines", fontsize=title_size, pad = title_pad,  fontweight='bold')
    plt.ylabel('Accuracy', fontsize=ylabel_size, labelpad=label_pad,  fontweight='bold')

    merge = []
    rule_base = []
    notebook_only = []
    deepline_only = []
    autosklearn_only = []
    with open('boxplot.txt', 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
                pass

            res = line.split(' ')
            merge.append(float(res[0]))
            rule_base.append(float(res[1]))
            notebook_only.append(float(res[2]))
            deepline_only.append(float(res[3]))
            autosklearn_only.append(float(res[4]))

    labels = ['HAIPipe-opt', 'HumanPipe', 'HAIPipe-hue', 'Deepline', 'Autosklearn']
    
    bp = plt.boxplot([merge, notebook_only, rule_base, deepline_only, autosklearn_only], labels=labels, patch_artist=True, )
    [bp['boxes'][i].set(facecolor=color_list[i], hatch=hatches[i], linewidth=2) for i in range(len(labels))]
    for whisker in bp['whiskers']:
        whisker.set(color='black', linewidth=2)

    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='black', linewidth=2)

    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='white', linewidth=3)

    print('merge mean', np.array(merge).mean())
    print('merge 25%', np.quantile(merge, 0.25, interpolation='lower'))
    print('merge median', np.median(merge))
    print('merge 75%', np.quantile(merge, 0.75, interpolation='higher'))

    print('rule mean', np.array(rule_base).mean())
    print('rule 25%', np.quantile(rule_base, 0.25, interpolation='lower'))
    print('rule median', np.median(rule_base))
    print('rule 75%', np.quantile(rule_base, 0.75, interpolation='higher'))

    print('human mean', np.array(notebook_only).mean())
    print('human 25%', np.quantile(notebook_only, 0.25, interpolation='lower'))
    print('human median', np.median(notebook_only))
    print('human 75%', np.quantile(notebook_only, 0.75, interpolation='higher'))

    print('deepline mean', np.array(deepline_only).mean())
    print('deepline 25%', np.quantile(deepline_only, 0.25, interpolation='lower'))
    print('deepline median', np.median(deepline_only))
    print('deepline 75%', np.quantile(deepline_only, 0.75, interpolation='higher'))

    print('autosklearn mean', np.array(autosklearn_only).mean())
    print('autosklearn 25%', np.quantile(autosklearn_only, 0.25, interpolation='lower'))
    print('autosklearn median', np.median(autosklearn_only))
    print('autosklearn 75%', np.quantile(autosklearn_only, 0.75, interpolation='higher'))
    plt.savefig('boxplot.eps')

def hist_operation_frequence_new_data():
    tick_size = 45
    legend_size = 38
    title_size = 50
    ylabel_size = 50
    title_pad = 40
    label_pad = 35
    lagend_ncol = 5


    # rec1 = [0, 0, 0.09, 1]
    # rec2 = [0.11, 0, 0.09, 1]
    # rec3 = [0.22, 0, 0.26, 1]
    # rec4 = [0.50, 0, 0.23, 1]
    # rec5 = [0.75, 0, 0.20, 1]
    rec1 = [0,0.87,1,0.09]
    rec2 = [0,0.76,1,0.09]
    rec3 = [0,0.47,1,0.27]
    rec4 = [0,0.22,1,0.23]
    rec5 = [0,0,1,0.20]

    labels = ['Imputer', 'Encoder', 'Scaler', 'FeatureEngine', 'FeatureSelect']

    imputer = {
        'dropna': 120, #dpn
        'fillna': 103,#fln
        'SimpleImputer': 9,#spi
        }
    encoder = {
        'get_dummies': 176, # gdm
        'LabelEncoder': 138, # LEC
        'OneHotEncoder': 9, # OHE
        }
    scaler = {
        'StandardScaler': 246, #SdS
        'MinMaxScaler': 62, # MMS
        'RobustScaler': 3, # rbs
        'QuantileTransformer': 2, # qtt
        'Normalizer': 1, # NML
        'MaxAbsScaler': 0, # MAS
        'PowerTransformer': 0, # PTF
        'KBinsDiscretizer': 0, # KBD
        }
    fengine = {
        'PCA': 31, # PcA
        'PolynomialFeatures': 15, # PLF
        'IncrementalPCA': 0, 
        'TruncatedSVD': 0,
        'KernelPCA': 0,
        'RandomTreesEmbedding': 0,
        }
    fselection = {
        'SelectKBest': 19,
        'RFE': 19,
        'SelectPercentile': 2,
        'VarianceThreshold': 1,
        'SelectFpr': 0,
        'SelectFwe': 0,
        }
    # imputer = {
    #     'DPN': 120, #dpn
    #     'FLN': 103,#fln
    #     'SPI': 9,#spi
    #     }
    # encoder = {
    #     'GDM': 176, # gdm
    #     'LEC': 138, # LEC
    #     'OHE': 9, # OHE
    #     }
    # scaler = {
    #     'SDS': 246, #SdS
    #     'MMS': 62, # MMS
    #     'RBS': 3, # rbs
    #     'QTT': 2, # qtt
    #     'NML': 1, # NML
    #     'MAS': 0, # MAS
    #     'PTF': 0, # PTF
    #     'KBS': 0, # KBD
    #     }
    # fengine = {
    #     'PCA': 31, # PcA
    #     'PLF': 15, # PLF
    #     'ICA': 0, 
    #     'TVD': 0,
    #     'KCA': 0,
    #     'RTE': 0,
    #     }
    # fselection = {
    #     'SKB': 19,
    #     'RFE': 19,
    #     'SPT': 2,
    #     'VTS': 1,
    #     'SFP': 0,
    #     'SFW': 0,
    #     }

    fig = plt.figure(figsize=(15,25))
    plt.subplots_adjust(wspace=1.2)
    # ax1 = fig.add_subplot(5,1,1)
    ax1 = plt.axes(rec1)
    plt.xlim(0, 0.35)
    plt.tick_params(labelsize=tick_size)
    # ax1.set_title("Ratio of Usage", fontsize=title_size, pad = title_pad,  fontweight='bold')
    ax1.set_xticks([])
    # plt.ylabel('Ratio of Usage', fontsize=ylabel_size, labelpad=label_pad,  fontweight='bold')

    # ax2 = fig.add_subplot(5,1,2)
    ax2 = plt.axes(rec2)
    plt.xlim(0, 0.35)
    plt.tick_params(labelsize=tick_size)
    ax2.set_xticks([])
    # ax2.set_title("Encoder", fontsize=title_size, pad = title_pad,  fontweight='bold')

    # ax3 = fig.add_subplot(5,1,3)
    ax3 = plt.axes(rec3)
    plt.xlim(0, 0.35)
    plt.tick_params(labelsize=tick_size)
    ax3.set_xticks([])
    # ax3.set_title("Scaler", fontsize=title_size, pad = title_pad,  fontweight='bold')

    # ax4 = fig.add_subplot(5,1,4)
    ax4 = plt.axes(rec4)
    plt.xlim(0, 0.35)
    plt.tick_params(labelsize=tick_size)
    ax4.set_xticks([])
    # ax4.set_title("Feature Engine", fontsize=title_size, pad = title_pad,  fontweight='bold')

    # ax5 = fig.add_subplot(5,1,5)
    ax5 = plt.axes(rec5)
    plt.xlim(0, 0.35)
    plt.tick_params(labelsize=tick_size)
    
    # ax5.set_title("Feature Selection", fontsize=title_size, pad = title_pad,  fontweight='bold')
    ax5.set_xlabel("Ratio of Usage", fontsize=title_size, labelpad = title_pad,  fontweight='bold')
    axes = [ax1,ax2,ax3,ax4,ax5]
    dicts = [imputer, encoder, scaler, fengine, fselection]
    
    for index,ax in enumerate(axes):
        x = []
        y = []
        for index1 in range(len(list(dicts[index]))):
            operation = list(dicts[index])[len(list(dicts[index]))-index1-1]
            x.append(operation)
            y.append(dicts[index][operation]/794)
        
        # axes[index].barh(range(len(x)),y, width=0.3,color='tab:brown')
        x_input = [ i+0.1 for i in range(len(x))]
        axes[index].barh(x_input,y, height=0.8,color = color_list1[index], label=labels[index])
        axes[index].set_yticks(range(len(x)))
        axes[index].set_yticklabels(x)
        # for tick in axes[index].get_xticklabels():
            # tick.set_rotation(45)
        axes[index].legend(loc = 4,  prop={'weight':'bold', 'size': legend_size}, borderaxespad= 0.)
    plt.savefig("operation_usage_new_data.eps", bbox_inches='tight')

def hist_operation_frequence():
    tick_size = 45
    legend_size = 38
    title_size = 50
    ylabel_size = 50
    title_pad = 40
    label_pad = 35
    lagend_ncol = 5


    # rec1 = [0, 0, 0.09, 1]
    # rec2 = [0.11, 0, 0.09, 1]
    # rec3 = [0.22, 0, 0.26, 1]
    # rec4 = [0.50, 0, 0.23, 1]
    # rec5 = [0.75, 0, 0.20, 1]
    rec1 = [0,0.87,1,0.09]
    rec2 = [0,0.76,1,0.09]
    rec3 = [0,0.47,1,0.27]
    rec4 = [0,0.22,1,0.23]
    rec5 = [0,0,1,0.20]

    labels = ['Imputer', 'Encoder', 'Scaler', 'FeatureEngine', 'FeatureSelect']

    imputer = {
        'dropna': 120, #dpn
        'fillna': 103,#fln
        'SimpleImputer': 9,#spi
        }
    encoder = {
        'get_dummies': 176, # gdm
        'LabelEncoder': 138, # LEC
        'OneHotEncoder': 9, # OHE
        }
    scaler = {
        'StandardScaler': 246, #SdS
        'MinMaxScaler': 62, # MMS
        'RobustScaler': 3, # rbs
        'QuantileTransformer': 2, # qtt
        'Normalizer': 1, # NML
        'MaxAbsScaler': 0, # MAS
        'PowerTransformer': 0, # PTF
        'KBinsDiscretizer': 0, # KBD
        }
    fengine = {
        'PCA': 31, # PcA
        'PolynomialFeatures': 15, # PLF
        'IncrementalPCA': 0, 
        'TruncatedSVD': 0,
        'KernelPCA': 0,
        'RandomTreesEmbedding': 0,
        }
    fselection = {
        'SelectKBest': 19,
        'RFE': 19,
        'SelectPercentile': 2,
        'VarianceThreshold': 1,
        'SelectFpr': 0,
        'SelectFwe': 0,
        }
    # imputer = {
    #     'DPN': 120, #dpn
    #     'FLN': 103,#fln
    #     'SPI': 9,#spi
    #     }
    # encoder = {
    #     'GDM': 176, # gdm
    #     'LEC': 138, # LEC
    #     'OHE': 9, # OHE
    #     }
    # scaler = {
    #     'SDS': 246, #SdS
    #     'MMS': 62, # MMS
    #     'RBS': 3, # rbs
    #     'QTT': 2, # qtt
    #     'NML': 1, # NML
    #     'MAS': 0, # MAS
    #     'PTF': 0, # PTF
    #     'KBS': 0, # KBD
    #     }
    # fengine = {
    #     'PCA': 31, # PcA
    #     'PLF': 15, # PLF
    #     'ICA': 0, 
    #     'TVD': 0,
    #     'KCA': 0,
    #     'RTE': 0,
    #     }
    # fselection = {
    #     'SKB': 19,
    #     'RFE': 19,
    #     'SPT': 2,
    #     'VTS': 1,
    #     'SFP': 0,
    #     'SFW': 0,
    #     }

    fig = plt.figure(figsize=(15,25))
    plt.subplots_adjust(wspace=1.2)
    # ax1 = fig.add_subplot(5,1,1)
    ax1 = plt.axes(rec1)
    plt.xlim(0, 0.35)
    plt.tick_params(labelsize=tick_size)
    # ax1.set_title("Ratio of Usage", fontsize=title_size, pad = title_pad,  fontweight='bold')
    ax1.set_xticks([])
    # plt.ylabel('Ratio of Usage', fontsize=ylabel_size, labelpad=label_pad,  fontweight='bold')

    # ax2 = fig.add_subplot(5,1,2)
    ax2 = plt.axes(rec2)
    plt.xlim(0, 0.35)
    plt.tick_params(labelsize=tick_size)
    ax2.set_xticks([])
    # ax2.set_title("Encoder", fontsize=title_size, pad = title_pad,  fontweight='bold')

    # ax3 = fig.add_subplot(5,1,3)
    ax3 = plt.axes(rec3)
    plt.xlim(0, 0.35)
    plt.tick_params(labelsize=tick_size)
    ax3.set_xticks([])
    # ax3.set_title("Scaler", fontsize=title_size, pad = title_pad,  fontweight='bold')

    # ax4 = fig.add_subplot(5,1,4)
    ax4 = plt.axes(rec4)
    plt.xlim(0, 0.35)
    plt.tick_params(labelsize=tick_size)
    ax4.set_xticks([])
    # ax4.set_title("Feature Engine", fontsize=title_size, pad = title_pad,  fontweight='bold')

    # ax5 = fig.add_subplot(5,1,5)
    ax5 = plt.axes(rec5)
    plt.xlim(0, 0.35)
    plt.tick_params(labelsize=tick_size)
    
    # ax5.set_title("Feature Selection", fontsize=title_size, pad = title_pad,  fontweight='bold')
    ax5.set_xlabel("Ratio of Usage", fontsize=title_size, labelpad = title_pad,  fontweight='bold')
    axes = [ax1,ax2,ax3,ax4,ax5]
    dicts = [imputer, encoder, scaler, fengine, fselection]
    
    for index,ax in enumerate(axes):
        x = []
        y = []
        for index1 in range(len(list(dicts[index]))):
            operation = list(dicts[index])[len(list(dicts[index]))-index1-1]
            x.append(operation)
            y.append(dicts[index][operation]/794)
        
        # axes[index].barh(range(len(x)),y, width=0.3,color='tab:brown')
        x_input = [ i+0.1 for i in range(len(x))]
        axes[index].barh(x_input,y, height=0.8,color = color_list1[index], label=labels[index])
        axes[index].set_yticks(range(len(x)))
        axes[index].set_yticklabels(x)
        # for tick in axes[index].get_xticklabels():
            # tick.set_rotation(45)
        axes[index].legend(loc = 4,  prop={'weight':'bold', 'size': legend_size}, borderaxespad= 0.)
    plt.savefig("operation_usage.eps", bbox_inches='tight')

def hist_operation_frequence_hol():
    tick_size = 50
    legend_size = 35
    title_size = 50
    ylabel_size = 50
    title_pad = 40
    label_pad = 35
    lagend_ncol = 5

    fig = plt.figure(figsize=(30,10))
    rec1 = [0, 0, 0.09, 1]
    rec2 = [0.11, 0, 0.09, 1]
    rec3 = [0.22, 0, 0.26, 1]
    rec4 = [0.50, 0, 0.23, 1]
    rec5 = [0.75, 0, 0.20, 1]
    # rec1 = [0,0.87,1,0.09]
    # rec2 = [0,0.76,1,0.09]
    # rec3 = [0,0.47,1,0.27]
    # rec4 = [0,0.22,1,0.23]
    # rec5 = [0,0,1,0.20]

    labels = ['Imputer', 'Encoder', 'Scaler', 'FeatureEngine', 'FeatureSelect']

    imputer = {
        'dropna': 120, #dpn
        'fillna': 103,#fln
        'SimpleImp': 9,#spi
        }
    encoder = {
        'get_dummies': 176, # gdm
        'LabelEnc': 138, # LEC
        'OneHotEnc': 9, # OHE
        }
    scaler = {
        'StandardSc': 246, #SdS
        'MinMaxSc': 62, # MMS
        'RobustSc': 3, # rbs
        'QuanTrans': 2, # qtt
        'Normalizer': 1, # NML
        'MaxAbsSc': 0, # MAS
        'PowerTrans': 0, # PTF
        'KBinsDisc': 0, # KBD
        }
    fengine = {
        'PCA': 31, # PcA
        'PolFeatures': 15, # PLF
        'IPCA': 0, 
        'TruncSVD': 0,
        'KPCA': 0,
        'RTEmb': 0,
        }
    fselection = {
        'SelectKB': 19,
        'RFE': 19,
        'SelectPer': 2,
        'VarThresh': 1,
        'SelectFpr': 0,
        'SelectFwe': 0,
        }
    # imputer = {
    #     'DPN': 120, #dpn
    #     'FLN': 103,#fln
    #     'SPI': 9,#spi
    #     }
    # encoder = {
    #     'GDM': 176, # gdm
    #     'LEC': 138, # LEC
    #     'OHE': 9, # OHE
    #     }
    # scaler = {
    #     'SDS': 246, #SdS
    #     'MMS': 62, # MMS
    #     'RBS': 3, # rbs
    #     'QTT': 2, # qtt
    #     'NML': 1, # NML
    #     'MAS': 0, # MAS
    #     'PTF': 0, # PTF
    #     'KBS': 0, # KBD
    #     }
    # fengine = {
    #     'PCA': 31, # PcA
    #     'PLF': 15, # PLF
    #     'ICA': 0, 
    #     'TVD': 0,
    #     'KCA': 0,
    #     'RTE': 0,
    #     }
    # fselection = {
    #     'SKB': 19,
    #     'RFE': 19,
    #     'SPT': 2,
    #     'VTS': 1,
    #     'SFP': 0,
    #     'SFW': 0,
    #     }

    fig = plt.figure(figsize=(40,10))
    plt.subplots_adjust(wspace=1.2)
    # ax1 = fig.add_subplot(5,1,1)
    ax1 = plt.axes(rec1)
    plt.ylim(0, 0.35)
    plt.tick_params(labelsize=tick_size)
    # ax1.set_title("Ratio of Usage", fontsize=title_size, pad = title_pad,  fontweight='bold')
    # ax1.set_yticks([])
    # plt.ylabel('Ratio of Usage', fontsize=ylabel_size, labelpad=label_pad,  fontweight='bold')

    # ax2 = fig.add_subplot(5,1,2)
    ax2 = plt.axes(rec2)
    plt.ylim(0, 0.35)
    plt.tick_params(labelsize=tick_size)
    ax2.set_yticks([])
    # ax2.set_title("Encoder", fontsize=title_size, pad = title_pad,  fontweight='bold')

    # ax3 = fig.add_subplot(5,1,3)
    ax3 = plt.axes(rec3)
    plt.ylim(0, 0.35)
    plt.tick_params(labelsize=tick_size)
    ax3.set_yticks([])
    # ax3.set_title("Scaler", fontsize=title_size, pad = title_pad,  fontweight='bold')

    # ax4 = fig.add_subplot(5,1,4)
    ax4 = plt.axes(rec4)
    plt.ylim(0, 0.35)
    plt.tick_params(labelsize=tick_size)
    ax4.set_yticks([])
    # ax4.set_title("Feature Engine", fontsize=title_size, pad = title_pad,  fontweight='bold')

    # ax5 = fig.add_subplot(5,1,5)
    ax5 = plt.axes(rec5)
    plt.ylim(0, 0.35)
    plt.tick_params(labelsize=tick_size)
    ax5.set_yticks([])
    # ax5.set_title("Feature Selection", fontsize=title_size, pad = title_pad,  fontweight='bold')
    ax1.set_ylabel("% Usage", fontsize=title_size, labelpad = title_pad,  fontweight='bold')
    axes = [ax1,ax2,ax3,ax4,ax5]
    dicts = [imputer, encoder, scaler, fengine, fselection]
    
    for index,ax in enumerate(axes):
        x = []
        y = []
        for index1 in range(len(list(dicts[index]))):
            operation = list(dicts[index])[index1]
            x.append(operation)
            y.append(dicts[index][operation]/794)
        
        # axes[index].barh(range(len(x)),y, width=0.3,color='tab:brown')
        x_input = [ i+0.1 for i in range(len(x))]
        axes[index].bar(x,y, width=0.8,color = color_list1[index], label=labels[index])
        # axes[index].set_xticks(range(len(x)))
        axes[index].set_xticks(range(len(x)))
        axes[index].set_xticklabels(x)
        for tick in axes[index].get_xticklabels():
            tick.set_rotation(90)
        axes[index].legend(loc = 0,  prop={'weight':'bold', 'size': legend_size}, borderaxespad= 0.,bbox_to_anchor=(0.99, 0.99))
    plt.savefig("operation_usage.eps", bbox_inches='tight')

def hist_operation_frequence_hol_new_data():
    tick_size = 50
    legend_size = 35
    title_size = 50
    ylabel_size = 50
    title_pad = 40
    label_pad = 35
    lagend_ncol = 5

    fig = plt.figure(figsize=(30,10))
    rec1 = [0, 0, 0.09, 1]
    rec2 = [0.11, 0, 0.09, 1]
    rec3 = [0.22, 0, 0.26, 1]
    rec4 = [0.50, 0, 0.23, 1]
    rec5 = [0.75, 0, 0.20, 1]

    labels = ['Imputer', 'Encoder', 'Scaler', 'FeatureEngine', 'FeatureSelect']

    imputer = {
        'dropna': 396, #dpn
        'fillna': 412,#fln
        'SimpleImputer': 59,#spi
        }
    encoder = {
        'get_dummies': 946, # gdm
        'LabelEncoder': 593, # LEC
        'OneHotEncoder': 112, # OHE
        }
    scaler = {
        'StandardScaler': 1026, #SdS
        'MinMaxScaler': 268, # MMS
        'RobustScaler': 30, # rbs
        'QuantileTransformer': 3, # qtt
        'Normalizer': 4, # NML
        'MaxAbsScaler': 1, # MAS
        'PowerTransformer': 5, # PTF
        'KBinsDiscretizer': 4, # KBD
        }
    fengine = {
        'PCA': 163, # PcA
        'PolynomialFeatures': 27, # PLF
        'IncrementalPCA': 1, 
        'TruncatedSVD': 8,
        'KernelPCA': 7,
        'RandomTreesEmbedding': 0,
        }
    fselection = {
        'SelectKBest': 90,
        'RFE': 88,
        'SelectPercentile': 4,
        'VarianceThreshold': 9,
        'SelectFpr': 0,
        'SelectFwe': 0,
        }

    fig = plt.figure(figsize=(40,10))
    plt.subplots_adjust(wspace=1.2)
    # ax1 = fig.add_subplot(5,1,1)
    ax1 = plt.axes(rec1)
    plt.ylim(0, 0.4)
    plt.tick_params(labelsize=tick_size)
    # ax1.set_title("Ratio of Usage", fontsize=title_size, pad = title_pad,  fontweight='bold')
    # ax1.set_yticks([])
    # plt.ylabel('Ratio of Usage', fontsize=ylabel_size, labelpad=label_pad,  fontweight='bold')

    # ax2 = fig.add_subplot(5,1,2)
    ax2 = plt.axes(rec2)
    plt.ylim(0, 0.4)
    plt.tick_params(labelsize=tick_size)
    ax2.set_yticks([])
    # ax2.set_title("Encoder", fontsize=title_size, pad = title_pad,  fontweight='bold')

    # ax3 = fig.add_subplot(5,1,3)
    ax3 = plt.axes(rec3)
    plt.ylim(0, 0.4)
    plt.tick_params(labelsize=tick_size)
    ax3.set_yticks([])
    # ax3.set_title("Scaler", fontsize=title_size, pad = title_pad,  fontweight='bold')

    # ax4 = fig.add_subplot(5,1,4)
    ax4 = plt.axes(rec4)
    plt.ylim(0, 0.4)
    plt.tick_params(labelsize=tick_size)
    ax4.set_yticks([])
    # ax4.set_title("Feature Engine", fontsize=title_size, pad = title_pad,  fontweight='bold')

    # ax5 = fig.add_subplot(5,1,5)
    ax5 = plt.axes(rec5)
    plt.ylim(0, 0.4)
    plt.tick_params(labelsize=tick_size)
    ax5.set_yticks([])
    # ax5.set_title("Feature Selection", fontsize=title_size, pad = title_pad,  fontweight='bold')
    ax1.set_ylabel("% Usage", fontsize=title_size, labelpad = title_pad,  fontweight='bold')
    axes = [ax1,ax2,ax3,ax4,ax5]
    dicts = [imputer, encoder, scaler, fengine, fselection]
    
    for index,ax in enumerate(axes):
        x = []
        y = []
        for index1 in range(len(list(dicts[index]))):
            operation = list(dicts[index])[index1]
            x.append(operation)
            y.append(dicts[index][operation]/3175)
        
        # axes[index].barh(range(len(x)),y, width=0.3,color='tab:brown')
        x_input = [ i+0.1 for i in range(len(x))]
        axes[index].bar(x,y, width=0.8,color = color_list1[index], label=labels[index])
        # axes[index].set_xticks(range(len(x)))
        axes[index].set_xticks(range(len(x)))
        axes[index].set_xticklabels(x)
        for tick in axes[index].get_xticklabels():
            tick.set_rotation(90)
        axes[index].legend(loc = 0,  prop={'weight':'bold', 'size': legend_size}, borderaxespad= 0.,bbox_to_anchor=(0.99, 0.99))
    plt.savefig("new_operation_usage.eps", bbox_inches='tight')

def hist_operation_frequence_hol_new_data():
    tick_size = 50
    legend_size = 35
    title_size = 50
    ylabel_size = 50
    title_pad = 40
    label_pad = 35
    lagend_ncol = 5

    fig = plt.figure(figsize=(30,10))
    rec1 = [0, 0, 0.09, 1]
    rec2 = [0.11, 0, 0.09, 1]
    rec3 = [0.22, 0, 0.26, 1]
    rec4 = [0.50, 0, 0.23, 1]
    rec5 = [0.75, 0, 0.20, 1]

    labels = ['Imputer', 'Encoder', 'Scaler', 'FeatureEngine', 'FeatureSelect']

    imputer = {
        'dropna': 396, #dpn
        'fillna': 412,#fln
        'SimpleImputer': 59,#spi
        }
    encoder = {
        'get_dummies': 946, # gdm
        'LabelEncoder': 593, # LEC
        'OneHotEncoder': 112, # OHE
        }
    scaler = {
        'StandardScaler': 1026, #SdS
        'MinMaxScaler': 268, # MMS
        'RobustScaler': 30, # rbs
        'QuantileTransformer': 3, # qtt
        'Normalizer': 4, # NML
        'MaxAbsScaler': 1, # MAS
        'PowerTransformer': 5, # PTF
        'KBinsDiscretizer': 4, # KBD
        }
    fengine = {
        'PCA': 163, # PcA
        'PolynomialFeatures': 27, # PLF
        'IncrementalPCA': 1, 
        'TruncatedSVD': 8,
        'KernelPCA': 7,
        'RandomTreesEmbedding': 0,
        }
    fselection = {
        'SelectKBest': 90,
        'RFE': 88,
        'SelectPercentile': 4,
        'VarianceThreshold': 9,
        'SelectFpr': 0,
        'SelectFwe': 0,
        }

    fig = plt.figure(figsize=(40,10))
    plt.subplots_adjust(wspace=1.2)
    # ax1 = fig.add_subplot(5,1,1)
    ax1 = plt.axes(rec1)
    plt.ylim(0, 0.4)
    plt.tick_params(labelsize=tick_size)
    # ax1.set_title("Ratio of Usage", fontsize=title_size, pad = title_pad,  fontweight='bold')
    # ax1.set_yticks([])
    # plt.ylabel('Ratio of Usage', fontsize=ylabel_size, labelpad=label_pad,  fontweight='bold')

    # ax2 = fig.add_subplot(5,1,2)
    ax2 = plt.axes(rec2)
    plt.ylim(0, 0.4)
    plt.tick_params(labelsize=tick_size)
    ax2.set_yticks([])
    # ax2.set_title("Encoder", fontsize=title_size, pad = title_pad,  fontweight='bold')

    # ax3 = fig.add_subplot(5,1,3)
    ax3 = plt.axes(rec3)
    plt.ylim(0, 0.4)
    plt.tick_params(labelsize=tick_size)
    ax3.set_yticks([])
    # ax3.set_title("Scaler", fontsize=title_size, pad = title_pad,  fontweight='bold')

    # ax4 = fig.add_subplot(5,1,4)
    ax4 = plt.axes(rec4)
    plt.ylim(0, 0.4)
    plt.tick_params(labelsize=tick_size)
    ax4.set_yticks([])
    # ax4.set_title("Feature Engine", fontsize=title_size, pad = title_pad,  fontweight='bold')

    # ax5 = fig.add_subplot(5,1,5)
    ax5 = plt.axes(rec5)
    plt.ylim(0, 0.4)
    plt.tick_params(labelsize=tick_size)
    ax5.set_yticks([])
    # ax5.set_title("Feature Selection", fontsize=title_size, pad = title_pad,  fontweight='bold')
    ax1.set_ylabel("% Usage", fontsize=title_size, labelpad = title_pad,  fontweight='bold')
    axes = [ax1,ax2,ax3,ax4,ax5]
    dicts = [imputer, encoder, scaler, fengine, fselection]
    
    for index,ax in enumerate(axes):
        x = []
        y = []
        for index1 in range(len(list(dicts[index]))):
            operation = list(dicts[index])[index1]
            x.append(operation)
            y.append(dicts[index][operation]/3175)
        
        # axes[index].barh(range(len(x)),y, width=0.3,color='tab:brown')
        x_input = [ i+0.1 for i in range(len(x))]
        axes[index].bar(x,y, width=0.8,color = color_list1[index], label=labels[index])
        # axes[index].set_xticks(range(len(x)))
        axes[index].set_xticks(range(len(x)))
        axes[index].set_xticklabels(x)
        for tick in axes[index].get_xticklabels():
            tick.set_rotation(90)
        axes[index].legend(loc = 0,  prop={'weight':'bold', 'size': legend_size}, borderaxespad= 0.,bbox_to_anchor=(0.99, 0.99))
    plt.savefig("new_operation_usage.eps", bbox_inches='tight')

def hist_popdata():
    tick_size = 45
    legend_size = 38
    title_size = 50
    ylabel_size = 50
    title_pad = 40
    label_pad = 35
    lagend_ncol = 5

    pop1_y = [0.8101689777979845, 0.7987471110858528, 0.7596634178222323, 0.579622641509434, 0.4225235849056604]
    pop2_y = [0.963224306823842, 0.9242902208201892, 0.8235513863523163, 0.914298245614035, 0.7602108583762245]
    pop3_y = [0.7897469934582436, 0.7713805302844836, 0.7309115258890516, 0.7889814814814815, 0.7760231842914598]
    pop4_y = [0.9023148148148148, 0.8425925925925924, 0.8782407407407405, 0.7240740740740741, 0.864814814814815]
    pop5_y = [0.8889191452865925, 0.8801922849667362, 0.7224891789098215, 0.912, 0.9784141355470982]
    labels = ['HAIPipe-opt', 'HumanPipe', 'HAIPipe-hue', 'Deepline', 'Autosklearn']
    xticks = ['']
    hatches = ['/', '\\', '-', '.', '+']

    fig = plt.figure(figsize=(30,10))
    plt.subplots_adjust(wspace=0.4)
    ax1 = fig.add_subplot(1,5,1)
    # plt.ylim(0.6,0.9)
    plt.tick_params(labelsize=tick_size)
    ax1.set_xticks([])
    ax1.set_title("red-wine-quality", fontsize=title_size, pad = title_pad,  fontweight='bold')
    plt.ylabel('Accuracy', fontsize=ylabel_size, labelpad=label_pad,  fontweight='bold')

    ax2 = fig.add_subplot(1,5,2)
    # plt.ylim(0.6,0.9)
    ax2.set_xticks([])
    plt.tick_params(labelsize=tick_size)
    ax2.set_title("voicegender", fontsize=title_size, pad = title_pad,  fontweight='bold')

    ax3 = fig.add_subplot(1,5,3)
    # plt.ylim(0.6,0.9)
    plt.tick_params(labelsize=tick_size)
    ax3.set_xticks([])
    ax3.set_title("telco-customer-churn", fontsize=title_size, pad = title_pad,  fontweight='bold')

    ax4 = fig.add_subplot(1,5,4)
    # plt.ylim(0.6,0.9)
    plt.tick_params(labelsize=tick_size)
    ax4.set_xticks([])
    ax4.set_title("social-network-ads", fontsize=title_size, pad = title_pad,  fontweight='bold')

    ax5 = fig.add_subplot(1,5,5)
    # plt.ylim(0.6,0.9)
    plt.tick_params(labelsize=tick_size)
    ax5.set_xticks([])
    ax5.set_title("weather-dataset-rattle-package", fontsize=title_size, pad = title_pad,  fontweight='bold')
    
    for index in range(5):
        ax1.bar(index*0.3, height=[pop1_y[index]], width = 0.25, label = labels[index], hatch = hatches[index])
    for index in range(5):
        ax2.bar(index*0.3, height=[pop2_y[index]], width = 0.25, label = labels[index], hatch = hatches[index])
    for index in range(5):
        ax3.bar(index*0.3, height=[pop3_y[index]], width = 0.25, label = labels[index], hatch = hatches[index])
    for index in range(5):
        ax4.bar(index*0.3, height=[pop4_y[index]], width = 0.25, label = labels[index], hatch = hatches[index])
    for index in range(5):
        ax5.bar(index*0.3, height=[pop5_y[index]], width = 0.25, label = labels[index], hatch = hatches[index])
    
    ax3.legend(loc = 8, ncol=lagend_ncol, prop={'weight':'bold', 'size': legend_size}, columnspacing = 0.7, bbox_to_anchor=(0.4, -0.19), borderaxespad= 0.)

    plt.savefig("hist_popdata.png", bbox_inches='tight')
    
def hist_difficulty():
    tick_size = 50
    legend_size = 50
    title_size = 50
    ylabel_size = 50
    title_pad = 40
    label_pad = 35
    lagend_ncol = 5

    # easy_y = [0.9600256549948778, 0.9545451530066872, 0.856733936182817, 0.8565658169130512, 0.7893827076852823]
    # labels = ['HAIPipe-opt', 'HumanPipe', 'HAIPipe-hue', 'Deepline', 'Autosklearn']
    xticks = ['']
    hatches = ['/', '\\', '-', '.', '+']
    # normal_y = [0.8611453174436311, 0.8370359791432249, 0.7831786554175983, 0.7764866714590238, 0.8147874932877085]
    # hard_y = [0.7191086357430101, 0.679158969379256, 0.6532545914072609, 0.695704155324664, 0.6603818165206216]

    easy_y = [0.9600256549948778, 0.9545451530066872, 0.8565658169130512, 0.7893827076852823]
    labels = ['HAIPipe', 'HumanPipe', 'Deepline', 'Autosklearn']
    normal_y = [0.8611453174436311, 0.8370359791432249, 0.7764866714590238, 0.8147874932877085]
    hard_y = [0.7191086357430101, 0.679158969379256, 0.695704155324664, 0.6603818165206216]

    fig = plt.figure(figsize=(30,10))
    plt.subplots_adjust(wspace=0.2)
    ax1 = fig.add_subplot(1,3,1)
    plt.ylim(0.6,1)
    plt.tick_params(labelsize=tick_size)
    ax1.set_xticks([])
    ax1.set_title("Easy", fontsize=title_size, pad = title_pad, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=ylabel_size, labelpad=label_pad,fontweight='bold')

    ax2 = fig.add_subplot(1,3,2)
    plt.ylim(0.6,1)
    ax2.set_xticks([])
    plt.tick_params(labelsize=tick_size)
    ax2.set_title("Normal", fontsize=title_size, pad = title_pad, fontweight='bold')

    ax3 = fig.add_subplot(1,3,3)
    plt.ylim(0.6,1)
    plt.tick_params(labelsize=tick_size)
    ax3.set_xticks([])
    ax3.set_title("Hard", fontsize=title_size, pad = title_pad, fontweight='bold')
    
    for index in range(4):
        ax1.bar(index*0.3, height=[easy_y[index]], width = 0.25, label = labels[index], hatch = hatches[index], color = color_list[index])
    for index in range(4):
        ax2.bar(index*0.3, height=[normal_y[index]], width = 0.25, label = labels[index], hatch = hatches[index], color = color_list[index])
    for index in range(4):
        ax3.bar(index*0.3, height=[hard_y[index]], width = 0.25, label = labels[index], hatch = hatches[index], color = color_list[index])
    
    ax3.legend(loc = 8, ncol=lagend_ncol, fontsize = legend_size, columnspacing = 0.7, bbox_to_anchor=(-0.7, -0.20), borderaxespad= 0.)

    plt.savefig("hist_difficulty.eps", bbox_inches='tight')

def hist_model():
    tick_size = 25
    legend_size = 50
    title_size = 50
    ylabel_size = 50
    title_pad = 40
    label_pad = 35
    lagend_ncol = 5

    # planBhai, rlhai, planbseq, rlseq, hi
    # svc_y = [0.849, 0.834, 0.840, 0.777, 0.774]
    # knn_y = [0.837, 0.832, 0.781, 0.758, 0.811]
    # lr_y = [0.858,0.841, 0.822, 0.804, 0.824]
    # dtc_y = [0.809, 0.808, 0.783, 0.791, 0.799]
    # rf_y = [0.856, 0.857, 0.848, 0.851, 0.855]
    # labels = ['ES_HAI', 'Firefly_HAI', 'ES', 'Firefly','HI']

    # task planB, Firefly, Deepline
    # all_y = [0.819, 0.803, 0.726]
    # svc_y = [0.840, 0.777, 0.775]
    # knn_y = [0.781, 0.758, 0.756]
    # lr_y = [0.822,0.804, 0.723]
    # dtc_y = [0.783, 0.791, 0.710]
    # rf_y = [0.848, 0.851, 0.629]
    labels = ['ES', '${HAIPipe^{AI}}$', 'Deepline']
    ### not remove split, only add
    # all_y = [0.819, 0.803, 0.726]
    # svc_y = [0.840, 0.777, 0.629]
    # knn_y = [0.782, 0.758, 0.723]
    # lr_y = [0.822, 0.802, 0.710]
    # dtc_y = [0.783, 0.791, 0.756]
    # rf_y = [0.848, 0.851, 0.775]
    ### remove split, delete
    all_y = [0.820, 0.804, 0.725]
    svc_y = [0.840, 0.777, 0.629]
    knn_y = [0.795, 0.771, 0.732]
    lr_y = [0.819, 0.799, 0.706]
    dtc_y = [0.783, 0.791, 0.756]
    rf_y = [0.846, 0.849, 0.771]




    # svc_y = [0.8406198564907007, 0.7737640017707708, 0.6374183920957707, 0.7023276539906499]
    # knn_y = [0.787580186146183, 0.7484315414192625, 0.7246655283280244, 0.7398417741979275]
    # lr_y = [0.8649953611012575,0.8445719824325975, 0.8219298187920329, 0.7662266692880485]
    # dtc_y = [0.7929275610542302, 0.78270195032151, 0.7331317994303858, 0.6813924958537203]
    # rf_y = [0.85080018803126, 0.8424448292755009, 0.8205110135537793, 0.7586789083826476]

    # svc_y = [0.8406198564907007, 0.7737640017707708, 0.7243589748918322, 0.6374183920957707, 0.7023276539906499]
    # knn_y = [0.787580186146183, 0.7484315414192625, 0.7034144062128632, 0.7246655283280244, 0.7398417741979275]
    # lr_y = [0.8649953611012575,0.8445719824325975, 0.7977601738003619, 0.8219298187920329, 0.7662266692880485]
    # dtc_y = [0.7929275610542302, 0.78270195032151, 0.7259982419952078, 0.7331317994303858, 0.6813924958537203]
    # rf_y = [0.85080018803126, 0.8424448292755009, 0.7619242151295385, 0.8205110135537793, 0.7586789083826476]

    # labels = ['HAIPipe-opt', 'HumanPipe', 'HAIPipe-hue', 'Deepline', 'Autosklearn']
    
    xticks = ['']
    hatches = ['/', '\\', '-', '.', '+']

    fig = plt.figure(figsize=(30,10))
    plt.subplots_adjust(wspace=0.4)
    ax1 = fig.add_subplot(1,6,1)
    plt.ylim(0.6,0.9)
    plt.tick_params(labelsize=tick_size)
    ax1.set_xticks([])
    ax1.set_title("ALL", fontsize=title_size, pad = title_pad,  fontweight='bold')
    plt.ylabel('Accuracy', fontsize=ylabel_size, labelpad=label_pad,  fontweight='bold')

    plt.subplots_adjust(wspace=0.4)
    ax2 = fig.add_subplot(1,6,2)
    plt.ylim(0.6,0.9)
    plt.tick_params(labelsize=tick_size)
    ax2.set_xticks([])
    ax2.set_title("SVM", fontsize=title_size, pad = title_pad,  fontweight='bold')
    # plt.ylabel('Accuracy', fontsize=ylabel_size, labelpad=label_pad,  fontweight='bold')

    ax3 = fig.add_subplot(1,6,3)
    plt.ylim(0.6,0.9)
    ax3.set_xticks([])
    plt.tick_params(labelsize=tick_size)
    ax3.set_title("KNN", fontsize=title_size, pad = title_pad,  fontweight='bold')

    ax4 = fig.add_subplot(1,6,4)
    plt.ylim(0.6,0.9)
    plt.tick_params(labelsize=tick_size)
    ax4.set_xticks([])
    ax4.set_title("LR", fontsize=title_size, pad = title_pad,  fontweight='bold')

    ax5 = fig.add_subplot(1,6,5)
    plt.ylim(0.6,0.9)
    plt.tick_params(labelsize=tick_size)
    ax5.set_xticks([])
    ax5.set_title("DT", fontsize=title_size, pad = title_pad,  fontweight='bold')

    ax6 = fig.add_subplot(1,6,6)
    plt.ylim(0.6,0.9)
    plt.tick_params(labelsize=tick_size)
    ax6.set_xticks([])
    ax6.set_title("RF", fontsize=title_size, pad = title_pad,  fontweight='bold')
    
    for index in range(3):
        ax1.bar(index*0.3, height=[all_y[index]], width = 0.25, label = labels[index], hatch = hatches[index], color = color_list[index])
    for index in range(3):
        ax2.bar(index*0.3, height=[svc_y[index]], width = 0.25, label = labels[index], hatch = hatches[index], color = color_list[index])
    for index in range(3):
        ax3.bar(index*0.3, height=[knn_y[index]], width = 0.25, label = labels[index], hatch = hatches[index], color = color_list[index])
    for index in range(3):
        ax4.bar(index*0.3, height=[lr_y[index]], width = 0.25, label = labels[index], hatch = hatches[index], color = color_list[index])
    for index in range(3):
        ax5.bar(index*0.3, height=[dtc_y[index]], width = 0.25, label = labels[index], hatch = hatches[index], color = color_list[index])
    for index in range(3):
        ax6.bar(index*0.3, height=[rf_y[index]], width = 0.25, label = labels[index], hatch = hatches[index], color = color_list[index])
    
    # ax3.legend(loc = 8, ncol=lagend_ncol, prop={'weight':'bold', 'size': legend_size}, columnspacing = 0.7, bbox_to_anchor=(0.5, -0.20), borderaxespad= 0.)
    ax3.legend(loc = 8, ncol=lagend_ncol, prop={'weight':'bold', 'size': legend_size}, columnspacing = 0.7, bbox_to_anchor=(1.4, -0.20), borderaxespad= 0.)

    plt.savefig("hist_model.pdf", bbox_inches='tight')

def look_ope():
    with open('clean_task.json','r') as f:
        clean_task = json.load(f)

    better = {}
    equal = {}
    worse = {}

    better_all = 0
    equal_all = 0
    worse_all = 0

    ope_num = {}
    for task in clean_task:
        for notebook_id in clean_task[task]['notebook_list']:
            # if clean_task[task]['model'] != 'RandomForestClassifier':
            # if clean_task[task]['model'] != 'DecisionTreeClassifier':
                # continue
            notebook = clean_task[task]['notebook_list'][notebook_id]
            best_index = notebook['max_index'].split('.')[0]
            if not os.path.exists('merge_code_new/'+notebook_id+'/'+best_index+'.json'):
                continue
            with open('merge_code_new/'+notebook_id+'/'+best_index+'.json','r') as f:
                merge_code_new_data = json.load(f)
            seq = merge_code_new_data['seq']

            add_seq = []
            for i in seq:
                add_seq.append(i['operator'])
            if notebook['hai'] > notebook['hi']:
                for ope in add_seq:
                    if ope not in better:
                        better[ope] = 0
                    better[ope] += 1
                better_all += 1
            elif notebook['hai'] == notebook['hi']:
                for ope in add_seq:
                    if ope not in equal:
                        equal[ope] = 0
                    equal[ope] += 1
                equal_all += 1
            else:
                for ope in add_seq:
                    if ope not in worse:
                        worse[ope] = 0
                    worse[ope] += 1
                worse_all += 1
    print('better', better)
    print('equal', equal)
    print('worse', worse)
    print('better_all', better_all)
    print('equal_all', equal_all)
    print('worse_all', worse_all)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    fig,ax = plt.subplots()
    ope_list = []
    groups = [better, equal, worse]
    group_names = ['hai better', 'equal', 'hai worse']
    group_num = [better_all, equal_all, worse_all]
    for group in groups:
        for key in group:
            if key not in ope_list:
                ope_list.append(key)

    x = np.arange(len(groups)) 
    print(x)
    x = np.array([0,10,20])
    # x_input = [ i+0.1 for i in range(len(groups))]
    opex = []
    for ope in ope_list:
        score_list = []
        for index,group in enumerate(groups):
            if ope in group:
                score_list.append(round(group[ope]/(better[ope]+equal[ope]+worse[ope]),3))
        opex.append(score_list)
    print('opex', opex)
    width = 0.35
    fig, ax = plt.subplots()
    for index,ope in enumerate(ope_list):
        print(opex[index])
        # print(x)
        rects = ax.bar(x+index*width, opex[index], width, label=ope)
        autolabel(rects)
    # rects2 = ax.bar(x + width/2, women_means, width, label='Women')

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Scores')
# ax.set_title('Scores by group and gender')
    # ax.set_xticks(group_names)
# ax.set_xticklabels(labels)
    ax.legend()
    plt.savefig('look_ope_dt_operatio.png')

def look_related_score_planb():
    with open('clean_task_planB_200.json','r') as f:
        clean_task = json.load(f)
    mean = 0
    all_ = 0
    rf_all_ = 0
    dt_all_ = 0
    knn_all_ = 0
    lr_all_ = 0
    svc_all_ = 0

    rf_mean = 0
    dt_mean = 0
    knn_mean = 0
    lr_mean = 0
    svc_mean = 0

    s_09_all_ = 0
    s_08_all_ = 0
    s_07_all_ = 0
    s_06_all_ = 0

    s_09 = 0
    s_08 = 0
    s_07 = 0
    s_06 = 0

    hi_mean_list = [[[],[],[],[],[],[],[],[],[],[]],
                    [[],[],[],[],[],[],[],[],[],[]],
                    [[],[],[],[],[],[],[],[],[],[]],
                    [[],[],[],[],[],[],[],[],[],[]],
                    [[],[],[],[],[],[],[],[],[],[]]]
    matrix = [[[],[],[],[],[],[],[],[],[],[]],
                    [[],[],[],[],[],[],[],[],[],[]],
                    [[],[],[],[],[],[],[],[],[],[]],
                    [[],[],[],[],[],[],[],[],[],[]],
                    [[],[],[],[],[],[],[],[],[],[]]]
    mean_score = [[[],[],[],[],[],[],[],[],[],[]],
                    [[],[],[],[],[],[],[],[],[],[]],
                    [[],[],[],[],[],[],[],[],[],[]],
                    [[],[],[],[],[],[],[],[],[],[]],
                    [[],[],[],[],[],[],[],[],[],[]]]
    num_matrix = [[[],[],[],[],[],[],[],[],[],[]],
                    [[],[],[],[],[],[],[],[],[],[]],
                    [[],[],[],[],[],[],[],[],[],[]],
                    [[],[],[],[],[],[],[],[],[],[]],
                    [[],[],[],[],[],[],[],[],[],[]]]

    # merged_matrix = [[[],[],[],[]],
    #                 [[],[],[],[]],
    #                 [[],[],[],[]],
    #                 [[],[],[],[]],
    #                 [[],[],[],[]]]
    # merged_mean_score = [[[],[],[],[]],
    #                 [[],[],[],[]],
    #                 [[],[],[],[]],
    #                 [[],[],[],[]],
    #                 [[],[],[],[]]]
    # merged_num = [[[],[],[],[]],
    #                 [[],[],[],[]],
    #                 [[],[],[],[]],
    #                 [[],[],[],[]],
    #                 [[],[],[],[]]]
    merged_matrix = [[[],[]],
                    [[],[]],
                    [[],[]],
                    [[],[]],
                    [[],[]]]
    merged_mean_score = [[[],[]],
                    [[],[]],
                    [[],[]],
                    [[],[]],
                    [[],[]]]
    merged_num = [[[],[]],
                    [[],[]],
                    [[],[]],
                    [[],[]],
                    [[],[]]]
    classifiers = ['RandomForestClassifier', 'DecisionTreeClassifier', 'KNeighborsClassifier', 'LogisticRegression', 'SVC']

    # pprint.pprint(matrix)
    for task in clean_task:
        # print(task)
        
        task_obj = clean_task[task]
        
        if not 'abs_planB_improve' in task_obj:
            continue
        # if abs(task_obj['abs_planB_improve']) > 0.2:
        #     continue
        mean += task_obj['abs_planB_improve']
        all_ +=1
        if task_obj['model_type'] == 'RandomForestClassifier':
            rf_mean += task_obj['abs_planB_improve']
            rf_all_ += 1
        if task_obj['model_type'] == 'DecisionTreeClassifier':
            dt_mean += task_obj['abs_planB_improve']
            dt_all_ += 1
        if task_obj['model_type'] == 'LogisticRegression':
            lr_mean += task_obj['abs_planB_improve']
            lr_all_ += 1
        if task_obj['model_type'] == 'KNeighborsClassifier':
            knn_mean += task_obj['abs_planB_improve']
            knn_all_ += 1
        if task_obj['model_type'] == 'SVC':
            svc_mean += task_obj['abs_planB_improve']
            svc_all_ += 1
        if task_obj['hi_mean'] < 1 and task_obj['hi_mean'] >= 0.9:
            s_09 += task_obj['abs_planB_improve']
            s_09_all_ += 1
        if task_obj['hi_mean'] < 0.9 and task_obj['hi_mean'] >= 0.8:
            s_08 += task_obj['abs_planB_improve']
            s_08_all_ += 1
        if task_obj['hi_mean'] < 0.8 and task_obj['hi_mean'] >= 0.7:
            s_07 += task_obj['abs_planB_improve']
            s_07_all_ += 1
        if task_obj['hi_mean'] < 0.7:
            s_06 += task_obj['abs_planB_improve']
            s_06_all_ += 1

        # if task_obj['hi_mean'] < 1 and task_obj['hi_mean'] >= 0.9:
        #     mean += task_obj['abs_planB_improve']
        #     all_ += 1
        hi_mean_index = math.floor(task_obj['hi_mean'] / 0.1)
        if hi_mean_index == 10:
            hi_mean_index = 9
        classifier_index = classifiers.index(task_obj['model_type'])
        # print(hi_mean_index)
        # print(classifier_index, hi_mean_index)

        matrix[classifier_index][hi_mean_index].append(round(task_obj['abs_planB_improve'],3))
        # matrix[classifier_index][hi_mean_index].append(round(task_obj['abs_planB_improve'],3))
        # if classifier_index == 1 and hi_mean_index == 0:
            # print(classifier_index, hi_mean_index)
            # print(len(matrix[classifier_index][hi_mean_index]))
    # pprint.pprint(matrix)
    # print(matrix[1][0])
    
    for classifier_index in range(len(matrix)):
        for human_score_index in range(len(matrix[classifier_index])):
            # print(classifier_index, human_score_index)
            for item in matrix[classifier_index][human_score_index]:
                if human_score_index in [0,1,2,3,4,5,6,7]:
                    merged_matrix[classifier_index][0].append(item)
                else:
                    merged_matrix[classifier_index][1].append(item)
            num_matrix[classifier_index][human_score_index] = len(matrix[classifier_index][human_score_index])
            mean_score[classifier_index][human_score_index] = np.array(matrix[classifier_index][human_score_index]).mean()

    for classifier_index in range(len(merged_matrix)):
        for human_score_index in range(len(merged_matrix[classifier_index])):
            # print(classifier_index, human_score_index, np.array(merged_matrix[classifier_index][human_score_index]))
            merged_num[classifier_index][human_score_index] = len(merged_matrix[classifier_index][human_score_index])
            merged_mean_score[classifier_index][human_score_index] = np.array(merged_matrix[classifier_index][human_score_index]).mean()
                  
    # 
    # pprint.pprint(num_matrix)
    mdat = np.around(np.ma.masked_array(mean_score,np.isnan(mean_score)),3)
    # pprint.pprint(mdat)
    # pprint.pprint(num_matrix)
    merged_mean_score = np.around(np.array(merged_mean_score),3)
    pprint.pprint(merged_mean_score)
    pprint.pprint(merged_num)
    classifer_mean = np.around(np.average(mdat,axis=0),3)
    human_index_mean = np.around(np.average(mdat,axis=1),3)
  
    print(classifiers)
    print(human_index_mean)
    # print(mdat[:,1])
    # print(mdat[0])
    

    print('0+, 0.1+, 0.2+, 0.3+, 0.4+ ,0.5+, 0.6+, 0.7+, 0.8+, 0.9+')
    print(classifer_mean)

    print('s06',np.array(s_06).mean()/s_06_all_)
    print('s07',np.array(s_07).mean()/s_07_all_)
    print('s08',np.array(s_08).mean()/s_08_all_)
    print('s09',np.array(s_06).mean()/s_09_all_)

    print('rf_mean',np.array(rf_mean).mean()/rf_all_)
    print('dt_mean',np.array(dt_mean).mean()/dt_all_)
    print('knn_mean',np.array(knn_mean).mean()/knn_all_)
    print('lr_mean',np.array(lr_mean).mean()/lr_all_)
    print('svc_mean',np.array(svc_mean).mean()/svc_all_)

    num_matrix = np.array(num_matrix)[:,6:]
    mean_score = np.array(mean_score)[:,6:]


    fig, ax = plt.subplots()
    plt.imshow(mean_score, cmap=plt.cm.hot, vmin=-0.1, vmax=0.2)
    plt.colorbar()
    ax.set_xticklabels(['0','0.6+','0.7+','0.8+','0.9+'])
    ax.set_yticklabels(['0','RF','DT','KN','LR','SVC'])
    ax.set_xlabel('human score mean')
    ax.set_ylabel('classifier')
    plt.savefig('look_abs_score.png')
    print('mean/all_', mean/all_)


def look_location():
    with open('rl_location.json','r') as f:
        location = json.load(f)
    with open('clean_task_rl_200.json','r') as f:
        clean_task = json.load(f)


    res = {}
    for task in clean_task:
        res[task] = {}
        for notebook_id in clean_task[task]['notebook_list']:
            notebook = clean_task[task]['notebook_list'][notebook_id]
            nb_location = location[notebook_id]
            if nb_location not in res[task]:
                res[task][nb_location] = []
            res[task][nb_location].append(notebook['rl_abs'])
        for nb_location in res[task]:
            res[task][nb_location] = np.array(res[task][nb_location]).mean()
    pprint.pprint(res)


def look_related_score_planbseq():
    with open('clean_task_rl_200.json','r') as f:
        clean_task = json.load(f)


    mean = 0
    all_ = 0
    rf_all_ = 0
    dt_all_ = 0
    knn_all_ = 0
    lr_all_ = 0
    svc_all_ = 0

    rf_mean = 0
    dt_mean = 0
    knn_mean = 0
    lr_mean = 0
    svc_mean = 0

    s_09_all_ = 0
    s_08_all_ = 0
    s_07_all_ = 0
    s_06_all_ = 0

    s_09 = 0
    s_08 = 0
    s_07 = 0
    s_06 = 0

    hi_mean_list = [[[],[],[],[],[],[],[],[],[],[]],
                    [[],[],[],[],[],[],[],[],[],[]],
                    [[],[],[],[],[],[],[],[],[],[]],
                    [[],[],[],[],[],[],[],[],[],[]],
                    [[],[],[],[],[],[],[],[],[],[]]]
    matrix = [[[],[],[],[],[],[],[],[],[],[]],
                    [[],[],[],[],[],[],[],[],[],[]],
                    [[],[],[],[],[],[],[],[],[],[]],
                    [[],[],[],[],[],[],[],[],[],[]],
                    [[],[],[],[],[],[],[],[],[],[]]]
    mean_score = [[[],[],[],[],[],[],[],[],[],[]],
                    [[],[],[],[],[],[],[],[],[],[]],
                    [[],[],[],[],[],[],[],[],[],[]],
                    [[],[],[],[],[],[],[],[],[],[]],
                    [[],[],[],[],[],[],[],[],[],[]]]
    num_matrix = [[[],[],[],[],[],[],[],[],[],[]],
                    [[],[],[],[],[],[],[],[],[],[]],
                    [[],[],[],[],[],[],[],[],[],[]],
                    [[],[],[],[],[],[],[],[],[],[]],
                    [[],[],[],[],[],[],[],[],[],[]]]

    # merged_matrix = [[[],[],[],[]],
    #                 [[],[],[],[]],
    #                 [[],[],[],[]],
    #                 [[],[],[],[]],
    #                 [[],[],[],[]]]
    # merged_mean_score = [[[],[],[],[]],
    #                 [[],[],[],[]],
    #                 [[],[],[],[]],
    #                 [[],[],[],[]],
    #                 [[],[],[],[]]]
    # merged_num = [[[],[],[],[]],
    #                 [[],[],[],[]],
    #                 [[],[],[],[]],
    #                 [[],[],[],[]],
    #                 [[],[],[],[]]]
    merged_matrix = [[[],[]],
                    [[],[]],
                    [[],[]],
                    [[],[]],
                    [[],[]]]
    merged_mean_score = [[[],[]],
                    [[],[]],
                    [[],[]],
                    [[],[]],
                    [[],[]]]
    merged_num = [[[],[]],
                    [[],[]],
                    [[],[]],
                    [[],[]],
                    [[],[]]]
    classifiers = ['RandomForestClassifier', 'DecisionTreeClassifier', 'KNeighborsClassifier', 'LogisticRegression', 'SVC']

    # pprint.pprint(matrix)
    all_score = []
    for task in clean_task:
        # print(task)
        
        task_obj = clean_task[task]
        
        if not 'abs_planB_improve' in task_obj:
            continue

        # abs_rl_imp_add_himean, abs_autosklearn_imp_add_himean,deepline
        print_name = 'abs_planB_imp_add_himean'
        print(task_obj[print_name])

        all_score.append(task_obj[print_name])
        # if abs(task_obj['abs_planB_seq_improve']) > 0.2:
            # continue
        # abs_deepline_improve, abs_planB_seq_improve, abs_autosklearn_improve, abs_rl_seq_improve, abs_rl_improve
        mean += task_obj[print_name]
        all_ +=1
        if task_obj['model_type'] == 'RandomForestClassifier':
            rf_mean += task_obj[print_name]
            rf_all_ += 1
        if task_obj['model_type'] == 'DecisionTreeClassifier':
            dt_mean += task_obj[print_name]
            dt_all_ += 1
        if task_obj['model_type'] == 'LogisticRegression':
            lr_mean += task_obj[print_name]
            lr_all_ += 1
        if task_obj['model_type'] == 'KNeighborsClassifier':
            knn_mean += task_obj[print_name]
            knn_all_ += 1
        if task_obj['model_type'] == 'SVC':
            svc_mean += task_obj[print_name]
            svc_all_ += 1
        # if task_obj['hi_mean'] < 1 and task_obj['hi_mean'] >= 0.9:
        #     s_09 += task_obj['abs_planB_seq_improve']
        #     s_09_all_ += 1
        # if task_obj['hi_mean'] < 0.9 and task_obj['hi_mean'] >= 0.8:
        #     s_08 += task_obj['abs_planB_seq_improve']
        #     s_08_all_ += 1
        # if task_obj['hi_mean'] < 0.8 and task_obj['hi_mean'] >= 0.7:
        #     s_07 += task_obj['abs_planB_seq_improve']
        #     s_07_all_ += 1
        # if task_obj['hi_mean'] < 0.7:
        #     s_06 += task_obj['abs_planB_seq_improve']
        #     s_06_all_ += 1
        if task_obj['hi_mean'] < 1 and task_obj['hi_mean'] >= 0.8:
            s_08 += task_obj[print_name]
            s_08_all_ += 1
        if task_obj['hi_mean'] < 0.8:
            s_07 += task_obj[print_name]
            s_07_all_ += 1

        # if task_obj['hi_mean'] < 1 and task_obj['hi_mean'] >= 0.9:
        #     mean += task_obj['abs_planB_improve']
        #     all_ += 1
        hi_mean_index = math.floor(task_obj['hi_mean'] / 0.1)
        if hi_mean_index == 10:
            hi_mean_index = 9
        classifier_index = classifiers.index(task_obj['model_type'])
        # print(hi_mean_index)
        # print(classifier_index, hi_mean_index)

        matrix[classifier_index][hi_mean_index].append(round(task_obj[print_name],3))
        # matrix[classifier_index][hi_mean_index].append(round(task_obj['abs_planB_improve'],3))
        # if classifier_index == 1 and hi_mean_index == 0:
            # print(classifier_index, hi_mean_index)
            # print(len(matrix[classifier_index][hi_mean_index]))
    # pprint.pprint(matrix)
    # print(matrix[1][0])
    
    all_score = np.array(all_score)
    avg = all_score.mean()
    qua25 = np.quantile(all_score, 0.25)
    qua5 = np.quantile(all_score, 0.5)
    qua75 = np.quantile(all_score, 0.75)
    print('avg', avg)
    print('qua25', qua25)
    print('qua5', qua5)
    print('qua75', qua75)
    for classifier_index in range(len(matrix)):
        for human_score_index in range(len(matrix[classifier_index])):
            # print(classifier_index, human_score_index)
            for item in matrix[classifier_index][human_score_index]:
                if human_score_index in [0,1,2,3,4,5,6,7]:
                    merged_matrix[classifier_index][0].append(item)
                else:
                    merged_matrix[classifier_index][1].append(item)
            num_matrix[classifier_index][human_score_index] = len(matrix[classifier_index][human_score_index])
            mean_score[classifier_index][human_score_index] = np.array(matrix[classifier_index][human_score_index]).mean()

    for classifier_index in range(len(merged_matrix)):
        for human_score_index in range(len(merged_matrix[classifier_index])):
            # print(classifier_index, human_score_index, np.array(merged_matrix[classifier_index][human_score_index]))
            merged_num[classifier_index][human_score_index] = len(merged_matrix[classifier_index][human_score_index])
            merged_mean_score[classifier_index][human_score_index] = np.array(merged_matrix[classifier_index][human_score_index]).mean()
                  
    # 
    # pprint.pprint(num_matrix)
    mdat = np.around(np.ma.masked_array(mean_score,np.isnan(mean_score)),3)
    # pprint.pprint(mdat)
    pprint.pprint(num_matrix)
    merged_mean_score = np.around(np.array(merged_mean_score),3)
    pprint.pprint(merged_mean_score)
    # pprint.pprint(merged_num)
    classifer_mean = np.around(np.average(mdat,axis=0),3)
    human_index_mean = np.around(np.average(mdat,axis=1),3)
  
    print(classifiers)
    print(human_index_mean)
    # print(mdat[:,1])
    # print(mdat[0])
    

    print('0+, 0.1+, 0.2+, 0.3+, 0.4+ ,0.5+, 0.6+, 0.7+, 0.8+, 0.9+')
    print(classifer_mean)

    print('s06',np.array(s_06).mean()/s_06_all_)
    print('s07',np.array(s_07).mean()/s_07_all_)
    print('s08',np.array(s_08).mean()/s_08_all_)
    print('s09',np.array(s_06).mean()/s_09_all_)

    print('rf_mean',np.array(rf_mean).mean()/rf_all_)
    print('dt_mean',np.array(dt_mean).mean()/dt_all_)
    print('knn_mean',np.array(knn_mean).mean()/knn_all_)
    print('lr_mean',np.array(lr_mean).mean()/lr_all_)
    print('svc_mean',np.array(svc_mean).mean()/svc_all_)

    num_matrix = np.array(num_matrix)[:,6:]
    mean_score = np.array(mean_score)[:,6:]


    fig, ax = plt.subplots()
    plt.imshow(mean_score, cmap=plt.cm.hot, vmin=-0.1, vmax=0.2)
    plt.colorbar()
    ax.set_xticklabels(['0','0.6+','0.7+','0.8+','0.9+'])
    ax.set_yticklabels(['0','RF','DT','KN','LR','SVC'])
    ax.set_xlabel('human score mean')
    ax.set_ylabel('classifier')
    plt.savefig('look_abs_score.png')
    print('mean/all_', mean/all_)

def plot_compare_task():
    str_ = ''
    # with open('validation_analyze_score_add_autosklearn_only_1.json', 'r') as f:
        # analyze_score = json.load(f)
    # with open('final_result_sibei_215.json', 'r') as f:
    with open('final_result_sibei_hole.json', 'r') as f:
        clean_data = json.load(f)
    with open('final_result_sibei_215_del_split.json', 'r') as f:
        del_split = json.load(f)
    better = []
    worse = []
    for task in clean_data:
        # rl_hai, rl_ai, ai-autosklearn, hi
        if task not in del_split:
            continue
        if len(del_split[task]['notebook_list']) == 0:
            continue

        for notebook_id in clean_data[task]['notebook_list']:
            notebook = clean_data[task]['notebook_list'][notebook_id]
            print(notebook['rl_hai'], notebook['hi'])
            # if clean_data[task]['abs_rl_improve'] >= 0:
            if notebook_id not in del_split[task]['notebook_list']:
                continue
            if notebook['rl_hai'] > notebook['hi']: # abs_rl_seq_improve, abs_autosklearn_improve
                # better.append((clean_data[task]['abs_rl_improve'], 0)) 
                better.append((notebook['rl_hai'], notebook['hi']))
            else:
                # worse.append((clean_data[task]['abs_rl_improve'], 0))
                worse.append((notebook['rl_hai'], notebook['hi']))
    
    print(len(better))
    print(len(worse))
    if len(better) > len(worse):
        for index in range(max(len(better), len(worse))):
            str_ += str(better[index][1])
            str_ += ' '
            str_ += str(better[index][0])
            str_ += ' '
            if index >= len(worse):
                str_ += '   \n'
            else:
                str_ += str(worse[index][1])
                str_ += ' '
                str_ += str(worse[index][0])
                str_ += '\n'
    else:
        for index in range(max(len(better), len(worse))):
            if index < len(better):
                str_ += str(better[index][1])
                str_ += ' '
                str_ += str(better[index][0])
                str_ += ' '
            else:
                str_ += '    '
            if index >= len(worse):
                str_ += '   \n'
            else:
                str_ += str(worse[index][1])
                str_ += ' '
                str_ += str(worse[index][0])
                str_ += '\n'
    print(len(better)/ (len(better)+len(worse)))
    print(len(worse)/ (len(better)+len(worse)))
    # with open('scatter_task_rl_hi.txt', 'w') as f:
    # with open('scatter_pip_rl_autosklearn.txt', 'w') as f:
        # f.write(str_)
    
def plot_compare_pipeline():
    str_ = ''
    # with open('validation_analyze_score_add_autosklearn_only_1.json', 'r') as f:
        # analyze_score = json.load(f)
    # with open('final_result_sibei_215.json', 'r') as f:
    with open('final_result_sibei_hole.json', 'r') as f:
        clean_data = json.load(f)
    with open('final_result_sibei_215_del_split.json', 'r') as f:
        del_split = json.load(f)
    better = []
    worse = []
    for task in clean_data:
        if task not in del_split:
            continue
        if len(del_split[task]['notebook_list']) == 0:
            continue
        for notebook_id in clean_data[task]['notebook_list']:
            if notebook_id not in del_split[task]['notebook_list']:
                continue
            notebook = clean_data[task]['notebook_list'][notebook_id]
            print(notebook['rl_hai'], clean_data[task]['rl_ai'])
            # if clean_data[task]['abs_rl_improve'] >= 0:
            if notebook['rl_hai'] >= clean_data[task]['rl_ai']: # abs_rl_seq_improve, abs_autosklearn_improve
                # better.append((clean_data[task]['abs_rl_improve'], 0)) 
                better.append((notebook['rl_hai'], clean_data[task]['rl_ai']))
            else:
                # worse.append((clean_data[task]['abs_rl_improve'], 0))
                worse.append((notebook['rl_hai'], clean_data[task]['rl_ai']))
    
    print(len(better))
    print(len(worse))
    if len(better) > len(worse):
        for index in range(max(len(better), len(worse))):
            str_ += str(better[index][1])
            str_ += ' '
            str_ += str(better[index][0])
            str_ += ' '
            if index >= len(worse):
                str_ += '   \n'
            else:
                str_ += str(worse[index][1])
                str_ += ' '
                str_ += str(worse[index][0])
                str_ += '\n'
    
    print(len(better)/ (len(better)+len(worse)))
    print(len(worse)/ (len(better)+len(worse)))
    # with open('scatter_task_rl_hi.txt', 'w') as f:
    with open('scatter_pip_rl_rlseq.txt', 'w') as f:
        f.write(str_)

def plot_time():
    tick_size = 30
    legend_size = 30
    title_size = 30
    ylabel_size = 30
    title_pad = 20
    label_pad = 18
    lagend_ncol = 5

    with open('clean_task_rl_200.json','r') as f:
        clean_data = json.load(f)
    with open('final_result_sibei_hole.json','r') as f:
        hole_clean_data = json.load(f)

    rl_all, rl_rf, rl_dt, rl_knn, rl_lr, rl_svc = [],[],[],[],[],[]
    planB_all, planB_rf, planB_dt, planB_knn, planB_lr, planB_svc = [],[],[],[],[],[]
    deepline_all, deepline_rf, deepline_dt, deepline_knn, deepline_lr, deepline_svc = [],[],[],[],[],[]

    rl_lists = [rl_rf, rl_dt, rl_knn, rl_lr, rl_svc]
    planB_lists = [planB_rf, planB_dt, planB_knn, planB_lr, planB_svc]
    deepline_lists = [deepline_rf, deepline_dt, deepline_knn, deepline_lr, deepline_svc]
    models = ['RandomForestClassifier', 'DecisionTreeClassifier', 'KNeighborsClassifier','LogisticRegression','SVC']
    for task in clean_data:
        if len(hole_clean_data[task]['notebook_list']) == 0:
            continue
        found_index = models.index(clean_data[task]['model_type'])
        rl_lists[found_index].append(clean_data[task]['rl_step1_time'])
        planB_lists[found_index].append(clean_data[task]['planB_step1_time'])
        deepline_lists[found_index].append(clean_data[task]['deepline_step1_time'])
        rl_all.append(clean_data[task]['rl_step1_time'])
        planB_all.append(clean_data[task]['planB_step1_time'])
        deepline_all.append(clean_data[task]['deepline_step1_time'])
        

    rl_shows = [np.array(rl_all).mean(), np.array(rl_rf).mean(), np.array(rl_dt).mean(), np.array(rl_knn).mean(), np.array(rl_lr).mean(), np.array(rl_svc).mean()]
    planB_shows = [np.array(planB_all).mean(), np.array(planB_rf).mean(), np.array(planB_dt).mean(), np.array(planB_knn).mean(), np.array(planB_lr).mean(), np.array(planB_svc).mean()]
    deepline_shows = [np.array(deepline_all).mean(), np.array(deepline_rf).mean(), np.array(deepline_dt).mean(), np.array(deepline_knn).mean(), np.array(deepline_lr).mean(), np.array(deepline_svc).mean()]
    xticks = np.arange(6)
    fig, ax = plt.subplots(figsize=(12,10))
    ax.bar(xticks, rl_shows, width=0.25, label='${HAIPipe^{AI}}$', hatch = hatches[0], color = color_list[0])
    ax.bar(xticks+0.25, planB_shows, width=0.25, label='Exhaustive Search', hatch = hatches[1], color = color_list[1])
    ax.bar(xticks+0.5, deepline_shows, width=0.25, label='Deepline', hatch = hatches[2], color = color_list[2])

    # ax.set_title("Comparison of Firefly and Exhaustive Search On Running Time", fontsize=title_size, fontweight='bold')
    ax.set_ylabel("Running Time(s)", fontsize=title_size, fontweight='bold')
    ax.legend(prop={'weight':'bold', 'size': legend_size})
    plt.tick_params(labelsize=tick_size)
    plt.yscale('log') 
    # ax.set_ylim(0.001,1000)
    ax.set_yticks([10,100,1000])
    ax.set_xticks(xticks + 0.125)
    ax.set_xticklabels(['All', 'RF', 'DT', 'KNN', 'LR', 'SVC'])

    plt.savefig("runningtime.pdf")


def ranking():
    x = list(range(11))
    classes = ['No.'+str(i+1) for i in x]
    old = [0.973, 0.964, 0.964, 0.938, 0.919, 0.916, 0.915, 0.915, 0.915, 0.915, 0.915]
    improved = [0.006, 0.011, 0.011, 0.019, 0.067, 0.041, 0.058, 0.058, 0.058, 0.028, 0.028]
    old.reverse()
    improved.reverse()
    classes.reverse()


    plt.barh(classes, old, label = 'original', color = 'gray')
    plt.barh(classes, improved, left=old, label = 'improved', color = 'green')

    plt.xlim(0.9,1)
    plt.legend()
    plt.yticks(classes)
    plt.savefig('rank.png')
    
if __name__ == "__main__":
    # hist_difficulty()
    # hist_model()
    # hist_popdata()
    # hist_operation_frequence()
    # hist_operation_frequence_hol()
    # hist_operation_frequence_new_AI()
    # hist_operation_frequence_hol_new_data()
    # hist_randomadd()
    # boxplot()
    # look_scatter_line()
    # look_bad_case()
    # look_bad_case_how_bad()
    # look_ope()
    # look_related_score_planb()
    # look_related_score_planbseq()
    # look_location()
    # plot_compare_pipeline()
    plot_compare_task()
    # bad_case()
    # plot_time()
    # ranking()