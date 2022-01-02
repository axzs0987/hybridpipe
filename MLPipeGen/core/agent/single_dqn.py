import argparse
import random
# import gym
import torch
from torch.optim import Adam
# from tester import Tester
from .buffer import ReplayBuffer
from MLPipeGen.core.config import Config
from MLPipeGen.core..util import get_class_attr_val
from .model import DQN, RnnDQN
from MLPipeGen.core.trainer import Trainer
import warnings
import numpy as np
import os
warnings.filterwarnings('ignore')
class DQNAgent:
    def __init__(self, process_id, config: Config):
        self.config = config
        self.is_training = True
        self.process_id = process_id
        self.buffer = ReplayBuffer(self.config.max_buff, self.process_id)
        if self.config.use_cuda:
            self.model = RnnDQN(self.config.single_action_dim).cuda()
            self.lpipeline_model = DQN(self.config.lpip_state_dim, self.config.lpipeline_action_dim).cuda()

        else:
            self.model = RnnDQN(self.config.single_action_dim)
            self.lpipeline_model = DQN(self.config.lpip_state_dim, self.config.lpipeline_action_dim)
        
        # self.imputercat_model_optim = Adam(self.imputercat_model.parameters(), lr=self.config.learning_rate)
        self.model_optim = Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.lpipeline_model_optim = Adam(self.lpipeline_model.parameters(), lr=self.config.learning_rate)

        if self.config.use_cuda:
            self.cuda()

    def act(self, state, index, tryed_list=None, epsilon=None, not_random=False, taskid=None):
        if tryed_list is None:
            tryed_list = []
        if epsilon is None: epsilon = self.config.epsilon_min
    
        if index == 'LogicPipeline':
            model = self.lpipeline_model
            action_dim = self.config.lpipeline_action_dim
        else:
            model = self.model
            action_dim = self.config.single_action_dim
        randnum = random.random()
        if ((randnum > epsilon or not self.is_training)) or not_random:
            print(state)
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            if self.config.use_cuda:
                state = state.cuda()
            q_value = model.forward(state)
            print("\033[1;36m" + str(q_value) + "\033[0m")
            print('trylist', tryed_list)
            action_index_list = [i for i in range(action_dim)]
            action_index_list = np.array([i for i in list(action_index_list) if i not in tryed_list])
            q_value_temp = np.array([i for index, i in enumerate(list(q_value[0])) if index not in tryed_list])
            print('q_value', q_value_temp)
  
            action = action_index_list[q_value_temp.argmax()]
            if not not_random:
                print("\033[1;35mmodelchosen\033[0m")
        else:
            q_value_temp = np.array([index for index, i in enumerate(np.zeros(action_dim)) if index not in tryed_list])
            action_index_list = [i for i in range(action_dim)]
            action_index_list = list(np.array([i for i in list(action_index_list) if i not in tryed_list]))
            action = random.sample(action_index_list, 1)[0]
            print("\033[1;35mrandomchosen\033[0m")
        print('\033[1;35maction '+ str(action) + "\033[0m")
        print('\033[1;35mindex ' + str(index) + "\033[0m")
        return action, randnum > epsilon 

    def learn_lp(self):
        s0, a, r = self.buffer.lp_sample(self.config.logic_batch_size)
        a = torch.tensor(np.array(a), dtype=torch.long)
        r = torch.tensor(np.array(r), dtype=torch.float)
        s0 = torch.tensor(s0, dtype=torch.float)
        if self.config.use_cuda: # need modify
            s0 = s0.cuda()
            s1 = s1.cuda()
            a = a.cuda()
            r = r.cuda()
            done = done.cuda()

            q_values = self.lpipeline_model(s0).cuda()
        else:
            result = []
            q_values = self.lpipeline_model(s0)

        q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
            # expected_q_value_0 = r_imputernum + self.config.gamma * next_q_value_0 * (1 - done_imputernum)
        loss = (q_value - r.detach()).pow(2).mean()
        self.lpipeline_model_optim.zero_grad()
        loss.backward()
        self.lpipeline_model_optim.step()
        return loss.item()

    def learning(self, test=False):
        if not test:
            s0, a, r, s1, done, index, logic_pipeline_id = self.buffer.sample(self.config.batch_size)
        else:
            s0, a, r, s1, done, index, logic_pipeline_id = self.buffer.sample(5)
        
        # print('a', a)
        # print('index', index)
        if os.path.exists(self.config.single_loss_log_file_name):
            loss_log = list(np.load(self.config.single_loss_log_file_name, allow_pickle=True))
        else:
            loss_log = []


        a = torch.tensor(np.array(a), dtype=torch.long)
       
        r = torch.tensor(np.array(r), dtype=torch.float)
       
        done = torch.tensor(np.array(done), dtype=torch.float)

        s0 = torch.tensor(s0, dtype=torch.float)

        s1 = torch.tensor(s0, dtype=torch.float)


        result = []

        q_values = self.model(s0)
        next_q_values = self.model(s1)
        next_q_value = next_q_values.max(1)[0]
        q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
        expected_q_value = r + self.config.gamma * next_q_value * (1 - done)
        # expected_q_value_0 = r_imputernum + next_q_value_0 * (1 - done_imputernum)
        loss = (q_value - expected_q_value.detach()).pow(2).mean()
        self.model_optim.zero_grad()
        loss.backward()
        self.model_optim.step()
        result.append(loss.item())
        loss_log.append(loss.item())

        np.save(self.config.single_loss_log_file_name, loss_log)
        return result

    def cuda(self):
        self.model.cuda()

    def test_load_weights(self, output, output1, number=None):
        if number is None:
            if output is None: return
            if os.path.exists(output1+'/model_imputernum_model.pkl'):
                print('load_imputernum.....')
                self.imputernum_model.load_state_dict(torch.load(output1+'/model_imputernum_model.pkl'))
                print(self.imputernum_model.named_parameters())
            if os.path.exists(output1+'/model_encoder_model.pkl'):
                print('load_encoder.....')
                self.encoder_model.load_state_dict(torch.load(output1+'/model_encoder_model.pkl'))
            if os.path.exists(output+'/model_fpreprocessing_model.pkl'):
                print('load_fprecessing.....')
                self.fpreprocessing_model.load_state_dict(torch.load(output+'/model_fpreprocessing_model.pkl'))
            if os.path.exists(output+'/model_fengine_model.pkl'):
                print('load_fegine.....')
                self.fengine_model.load_state_dict(torch.load(output+'/model_fengine_model.pkl'))
            if os.path.exists(output+'/model_fselection_model.pkl'):
                print('load_fselection.....')
                self.fselection_model.load_state_dict(torch.load(output+'/model_fselection_model.pkl'))
            if os.path.exists(output+'/model_logical_pipeline.pkl'):
                print('load_logical_pipeline.....')
                self.lpipeline_model.load_state_dict(torch.load(output+'/model_logical_pipeline.pkl'))
        else:
            if output is None: return
            if os.path.exists(output1+'/model_imputernum_model.pkl'):
                print('load_imputernum.....')
                self.imputernum_model.load_state_dict(torch.load(output1+'/model_imputernum_model.pkl'))
                print(self.imputernum_model.named_parameters())
            if os.path.exists(output1+'/model_encoder_model.pkl'):
                print('load_encoder.....')
                self.encoder_model.load_state_dict(torch.load(output1+'/model_encoder_model.pkl'))
            if os.path.exists(output+'/' + str(number) + '_model_fpreprocessing_model.pkl'):
                print('load_fprecessing.....')
                self.fpreprocessing_model.load_state_dict(torch.load(output+'/' + str(number) + '_model_fpreprocessing_model.pkl'))
            if os.path.exists(output+'/' + str(number) + '_model_fengine_model.pkl'):
                print('load_fegine.....')
                self.fengine_model.load_state_dict(torch.load(output+'/' + str(number) + '_model_fengine_model.pkl'))
            if os.path.exists(output+'/' + str(number) + '_model_fselection_model.pkl'):
                print('load_fselection.....')
                self.fselection_model.load_state_dict(torch.load(output+'/' + str(number) + '_model_fselection_model.pkl'))
            if os.path.exists(output+'/' + str(number) + '_model_logical_pipeline.pkl'):
                print('load_logical_pipeline.....')
                self.lpipeline_model.load_state_dict(torch.load(output+'/' + str(number) + '_model_logical_pipeline.pkl'))

    def load_weights(self, output, number=None):
        if number is None:
            if output is None: return
            if os.path.exists(output+'/model_single.pkl'):
                print('load_single_model.....')
                self.model.load_state_dict(torch.load(output+'/model_single.pkl'))
                print(self.model.named_parameters())
            if os.path.exists(output+'/model_logical_pipeline.pkl'):
                print('load_logical_pipeline.....')
                self.lpipeline_model.load_state_dict(torch.load(output+'/model_logical_pipeline.pkl'))
        else:
            if output is None: return
            if os.path.exists(output+'/' + str(number) + '_model_single.pkl'):
                print('load_single_model.....')
                self.model.load_state_dict(torch.load(output+'/' + str(number) + '_model_single.pkl'))
                print(self.model.named_parameters())
            if os.path.exists(output+'/' + str(number) + '_model_logical_pipeline.pkl'):
                print('load_logical_pipeline.....')
                self.lpipeline_model.load_state_dict(torch.load(output+'/' + str(number) + '_model_logical_pipeline.pkl'))

    def save_model(self, output, tag=''):
        torch.save(self.model.state_dict(), '%s/model_%s.pkl' % (output, 'single'))
        torch.save(self.lpipeline_model.state_dict(), '%s/model_%s.pkl' % (output, 'logical_pipeline'))
    

    def test_save_model(self, output, tag=''):
        torch.save(self.model.state_dict(), '%s/%s_model_%s.pkl' % (output, tag, 'single'))
        torch.save(self.lpipeline_model.state_dict(), '%s/model_%s.pkl' % (output, 'logical_pipeline'))
   
    def save_config(self, output):
        with open(output + '/config.txt', 'w') as f:
            attr_val = get_class_attr_val(self.config)
            for k, v in attr_val.items():
                f.write(str(k) + " = " + str(v) + "\n")


