from HybridPipeGen.core.al.config import Config
from HybridPipeGen.core.al.utils.runner import Runner
from HybridPipeGen.core.al.sample import Sampler
import numpy as np
import os
import torch
from sklearn.utils import resample
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable

import random
config = Config()

class Finetue:
    def __init__(self, notebook_id, k, T, V, model):
        self.T = T
        self.k = k
        self.V = V
        self.notebook_id = notebook_id
        self.load_data()
        self.run_fail = []
        
        self.sampler = Sampler(self.k, self.dataset_features, self.pipeline_features, model)
        self.notebook_id = notebook_id
        
    def load_data(self):
        features = np.load(config.test_features + self.notebook_id + '.npy', allow_pickle=True).item()
        self.dataset_features = features['dataset_features']
        self.pipeline_features = features['pipeline_features']
        self.seq_ids = features['seq_ids']

    def get_labels(self):
        origin_scores = {}
        res = {}
        info_triple = np.load(config.info_triple_path, allow_pickle=True).item()
        avg = 0
        count = 0
        for sample_id in self.sampler.samples:
            if sample_id in self.sampler.samples_label:
                continue
            seq_id = self.seq_ids[sample_id]

            code_path = 'HybridPipeGen/core/tmpdata/rl_cross_validation_code_py/'+ self.notebook_id + '/' + seq_id + '.py'
            if not os.path.exists('HybridPipeGen/core/tmpdata/rl_cross_val_res/'+ self.notebook_id + '/' + seq_id + '.npy'):
                runner = Runner()
                # print(type(info_triple))
                dataset_path = config.dataset_path + info_triple[self.notebook_id]['dataset_name']
                try:
                    runner.run_one_case(code_path, dataset_path)
                except:
                    pass
   
            if not os.path.exists('HybridPipeGen/core/tmpdata/rl_cross_val_res/' + self.notebook_id + '/' + seq_id + '.npy'):
                self.run_fail.append(seq_id)
                origin_score = {'accuracy_score': np.array([np.array([-1])])}
                # continue
            else:
                origin_score = np.load('HybridPipeGen/core/tmpdata/rl_cross_val_res/' + self.notebook_id + '/' + seq_id + '.npy', allow_pickle=True).item()

            origin_scores[sample_id] = origin_score['accuracy_score'].mean()
        
            if origin_scores[sample_id] != origin_scores[sample_id]:
                origin_scores[sample_id] = -1
            if origin_scores[sample_id] != -1:
                avg += origin_scores[sample_id]
                count += 1
        
        if count != 0:
            avg /= count

        for sample_id in origin_scores:
            if origin_scores[sample_id] == -1:
                continue
            res[sample_id] = origin_scores[sample_id] - avg

        for id_ in res:
            if id_ not in self.sampler.samples_label:
                self.sampler.samples_label[id_] = res[id_]

        temp = []
        for item in self.sampler.samples:
            if item in self.sampler.samples_label:
                temp.append(item)
        self.sampler.samples = temp

        return res

    def finetune(self, i=True, r=True, d=True):
        self.i = i
        self.r = r
        self.d = d

        if self.i:
            self.i_str = 't'
        else:
            self.i_str = 'f'
        if self.r:
            self.r_str = 't'
        else:
            self.r_str = 'f'
        if self.d:
            self.d_str = 't'
        else:
            self.d_str = 'f'
        for iter in range(self.T):
            sample_ids = self.sampler.select_topk(self.k, i ,r ,d)
            self.get_labels()
            # while len(self.sampler.samples_label) < (iter+1)*self.k and len(self.sampler.tryed) < len(self.sampler.dataset_features):

            #     self.sampler.select_topk((iter+1)*self.k - len(self.sampler.samples_label), i, r, d)
            #     self.get_labels()

            if len(self.sampler.samples_label.items()) == 0:
                if not os.path.exists('HybridPipeGen/core/tmpdata/almodel'):
                    os.mkdir('HybridPipeGen/core/tmpdata/almodel')
                torch.save(self.sampler.model, 'HybridPipeGen/core/tmpdata/almodel/'+self.i_str + self.r_str + self.d_str + '_'+ str(self.k) + '_' + str(self.notebook_id))
                return
            train_dataset_features = self.dataset_features[self.sampler.samples,:]
            train_pipeline_features = self.pipeline_features[self.sampler.samples,:]
            label =  torch.FloatTensor(list(self.sampler.samples_label.items()))[:, 1]
        
            
            self.sampler.model = self.train(self.sampler.model, train_dataset_features, train_pipeline_features, label, save=True)
            if i == True:
                for ind in range(self.sampler.emcm.num_committee):
                    bootstrap_labeled_pos_list = resample(
                        self.sampler.samples,
                        random_state=random.randrange(1000000)
                    )

                    data_dataset_feature_train = self.dataset_features[bootstrap_labeled_pos_list, :]
                    data_pipeline_feature_train = self.pipeline_features[bootstrap_labeled_pos_list, :]
                    data_y_train = []

                    for id_ in bootstrap_labeled_pos_list:
                        data_y_train.append(self.sampler.samples_label[id_])
                    data_y_train = torch.FloatTensor(data_y_train)

                    self.sampler.emcm.qbc_models[ind] = self.train(self.sampler.emcm.qbc_models[ind], data_dataset_feature_train, data_pipeline_feature_train, data_y_train, save=False)

    def get_result(self):
        score_dict = self.eval_all()
        score_dict = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)

        res = []

        found = False
        index = 0
        info_triple = np.load(config.info_triple_path, allow_pickle=True).item()

        # best_val = score_dict[0:self.V]

        val_score = {}
        while len(val_score) < min(self.V, len(score_dict)) and index < len(score_dict):
        # for tuple in best_val:
           
            tuple = score_dict[index]
            seq_id, one_score = tuple

            code_path = 'HybridPipeGen/core/tmpdata/rl_cross_validation_code_py/' +  self.notebook_id + '/' + seq_id + '.py'
            if not os.path.exists('HybridPipeGen/core/tmpdata/rl_cross_val_res/'+ self.notebook_id + '/' + seq_id + '.npy'):

                runner = Runner()
                dataset_path = config.dataset_path + info_triple[self.notebook_id]['dataset_name']
                # try:
                runner.run_one_case(code_path, dataset_path)
                # except:
                    # pass

                # print(config.param_hybrid_test_result_save_root_path + self.notebook_id + '/' + best_seq_id + '.npy')
            if not os.path.exists('HybridPipeGen/core/tmpdata/rl_cross_val_res/'+ self.notebook_id + '/' + seq_id + '.npy'):
                self.run_fail.append(seq_id)
                index += 1
                continue
            sc =  np.load('HybridPipeGen/core/tmpdata/rl_cross_val_res/' + self.notebook_id + '/' + seq_id + '.npy', allow_pickle=True).item()['accuracy_score'].mean()
            val_score[seq_id] = sc
            index += 1
        val_score = sorted(val_score.items(), key=lambda x: x[1], reverse=True)
        index = 0
        while found == False and index < len(score_dict):
            if index < len(val_score):
                best_seq_id, best_out_score = val_score[index]
            else:
                best_seq_id, best_out_score = score_dict[index]
            if best_seq_id == 'origin':
                origin_score = 'origin'
            else:
                code_path = 'HybridPipeGen/core/tmpdata/rl_test_merge_code_py/' +  self.notebook_id + '/' + best_seq_id + '.py'
                # if not os.path.exists(config.param_hybrid_test_result_save_root_path + self.notebook_id + '/' + best_seq_id + '.npy'):
                runner = Runner()
                dataset_path = config.dataset_path + info_triple[self.notebook_id]['dataset_name']
                
                runner.run_one_case(code_path, dataset_path)
                # print(config.param_hybrid_test_result_save_root_path + self.notebook_id + '/' + best_seq_id + '.npy')
                if not os.path.exists('HybridPipeGen/core/tmpdata/merge_max_result_rl/' + self.notebook_id + '/' + best_seq_id + '.npy'):
                    self.run_fail.append(best_seq_id)
                    index += 1
                    continue
                
                origin_score = np.load('HybridPipeGen/core/tmpdata/merge_max_result_rl/'+ self.notebook_id + '/' + best_seq_id + '.npy', allow_pickle=True).item()
            found = True
        if not found:
            origin_score = 'origin'
            best_seq_id = 'origin'
        return origin_score, val_score, best_seq_id
    def eval_all(self):
        model = torch.load('HybridPipeGen/core/tmpdata/almodel/'+self.i_str + self.r_str + self.d_str + '_' + str(self.k) +'_' +  self.notebook_id)
        if len(self.pipeline_features) == 1:
            return {self.seq_ids[0]: 0}
        # model.eval()
        model.seq_lstm.dropout=0
        out = model(torch.FloatTensor(self.dataset_features), torch.LongTensor(self.pipeline_features))

        score = {}
        for index, seq_id in enumerate(self.seq_ids):
            if seq_id not in self.run_fail:
                score[seq_id] = out[index].detach()

    
        return score
    def train(self, model, train_dataset_features, train_pipeline_features, label, save):
        learning_rate = 0.001
        opt = torch.optim.Adam(model.parameters(),lr=learning_rate)
        
        train_dataset_features = torch.FloatTensor(train_dataset_features)
        train_pipeline_features = torch.LongTensor(train_pipeline_features)

        label = torch.FloatTensor(label)
        train_dataset = TensorDataset(train_dataset_features, train_pipeline_features, label)
        train_loader = DataLoader(train_dataset,batch_size=len(train_dataset_features),shuffle=True)

        loss_func = torch.nn.L1Loss(reduction='mean')
        epoch_num = 5
        if len(train_dataset_features) != 1:
            for epoach in range(epoch_num):
                sum = 0
                for (dataset_feature,pipeline_feature,y) in train_loader:
                    dataset_feature = Variable(dataset_feature)
                    pipeline_feature = Variable(pipeline_feature)
                    y = Variable(y)
                    # print('epoach', epoach)
                    out = model(dataset_feature, pipeline_feature)
                    out = out.flatten()
                    # print('y', y)
                    # print('out', out)
                    loss = loss_func(out, y)
                    # print('loss', loss)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    for name, parms in model.named_parameters():	
                        # print('-->name:', name)
                        # # print('-->para:', parms)
                        # # print('-->grad_requirs:',parms.requires_grad)
                        # print('-->grad_value:',parms.grad)
                        # print('-->grad_value:',parms.grad.mean())
                        sum += parms.grad.mean()

                    # break


        if save == True:
            if not os.path.exists('HybridPipeGen/core/tmpdata/almodel'):
                os.mkdir('HybridPipeGen/core/tmpdata/almodel')
            torch.save(model, 'HybridPipeGen/core/tmpdata/almodel/'+self.i_str + self.r_str + self.d_str  + '_' + str(self.k) + '_'+str(self.notebook_id))
        return model