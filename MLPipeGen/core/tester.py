import math
from copy import deepcopy
import numpy as np
from MLPipeGen.core.config import Config
from MLPipeGen.core.logger import TensorBoardLogger
from MLPipeGen.core.util import get_output_folder
import os
from MLPipeGen.core.env.primitives.primitive import Primitive
from MLPipeGen.core.env.primitives.imputercat import ImputerCatPrim
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class Tester:
    def __init__(self, agent, env, test_pred, config: Config):
        self.agent = agent
        self.env = env
        self.config = config
        self.test_pred = test_pred

        # non-Linear epsilon decay
        epsilon_final = self.config.epsilon_min
        epsilon_start = self.config.epsilon
        epsilon_decay = self.config.eps_decay
        self.epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
            -1. * frame_idx / epsilon_decay)

        self.outputdir = self.config.model_dir
        # self.dpgoutputdir = self.config.dpgoutputdir
        # self.agent.save_config(self.outputdir)
        # self.board_logger = TensorBoardLogger(self.outputdir)

        
    def get_five_items_from_pipeline(self, fr, state, reward_dic, seq, taskid, need_save=True):
        tryed_list = []
        print("\033[1;34m"+ str(fr) +"\033[0m")
        epsilon = self.epsilon_by_frame(fr)
        pipeline_index = self.env.pipeline.get_index()
        has_num_nan, has_cat_nan = self.env.has_nan()
        if self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index]== 'ImputerNum': # imputernum   ----pipeline_index == 0:
            if has_num_nan:
                # try:
                action, isModel = self.agent.act(state, self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index], tryed_list, epsilon, not_random=True, taskid=taskid)
                # except:
                #     print('error state:', state)
                #     return
                temp = self.config.imputernums[action]
                step = deepcopy(temp)
            else:
                action = len(self.config.imputernums)
                step = Primitive()
        elif self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index]== 'ImputerCat': #pipeline_index == 1: # imputercat
            action = -1
            if has_cat_nan:
                step = ImputerCatPrim()
            else:
                step = Primitive()
        elif self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index]== 'Encoder': #pipeline_index == 2: # encoder
            if self.env.has_cat_cols():
                action, isModel = self.agent.act(state, self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index], tryed_list, epsilon, not_random=True, taskid=taskid)
                temp = self.config.encoders[action]
                step = deepcopy(temp)
            else:
                action = len(self.config.encoders)
                step = Primitive()
        elif self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index] in ['FeaturePreprocessing', 'FeatureEngine', 'FeatureSelection']: # elif pipeline_index in [3,4,5]: # fpreprocessin
            action, isModel = self.agent.act(state, self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index], tryed_list, epsilon, not_random=True, taskid=taskid)
            if self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index] == 'FeaturePreprocessing':
                temp = self.config.fpreprocessings[action]
                step = deepcopy(temp)
            elif self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index] == 'FeatureEngine':
                temp = self.config.fengines[action]
                step = deepcopy(temp)
            elif self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index] == 'FeatureSelection':
                temp = self.config.fselections[action]
                step = deepcopy(temp)

        # state, reward, next_state, done = self.env.step(step)
        step_result = self.env.step(step)
        tryed_list.append(action)
        repeat_time = 0
        while step_result==0 or step_result == 1:
            if self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index]== 'ImputerNum': 
                if has_num_nan:
                    try:
                        action, isModel = self.agent.act(state, self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index], tryed_list, epsilon, not_random=True, taskid=taskid)
                    except:
                        print('error state:', state)
                        return
                    temp = self.config.imputernums[action]
                    step = deepcopy(temp)
                else:
                    action = len(self.config.imputernums)
                    step = Primitive()
            elif self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index]== 'ImputerCat':  # imputercat
                action = -1
                if has_cat_nan:
                    step = ImputerCatPrim()
                else:
                    step = Primitive()
            elif self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index]== 'Encoder': # encoder
                if self.env.has_cat_cols():
                    action, isModel = self.agent.act(state, self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index], tryed_list, epsilon, not_random=True, taskid=taskid)
                    temp = self.config.encoders[action]
                    step = deepcopy(temp)
                else:
                    action = len(self.config.encoders)
                    step = Primitive()
            elif self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index] in ['FeaturePreprocessing', 'FeatureEngine', 'FeatureSelection']:  # fpreprocessin
                action, isModel = self.agent.act(state, self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index], tryed_list, epsilon, not_random=True, taskid=taskid)
                if self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index] == 'FeaturePreprocessing':
                    temp = self.config.fpreprocessings[action]
                    step = deepcopy(temp)
                elif self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index] == 'FeatureEngine':
                    temp = self.config.fengines[action]
                    step = deepcopy(temp)
                elif self.config.lpipelines[self.env.pipeline.logic_pipeline_id][pipeline_index] == 'FeatureSelection':
                    temp = self.config.fselections[action]
                    step = deepcopy(temp)
            if action in tryed_list:
                repeat_time += 1
                continue
            tryed_list.append(action)
            step_result = self.env.step(step)

        state, reward, next_state, done = step_result
        seq.append(step.name)
        state = next_state
        loss = 0

        # if self.config.checkpoint and fr % self.config.checkpoint_interval == 0:
        #     self.agent.save_checkpoint(fr, self.outputdir)
        if done:
            self.end_time = self.env.end_time
            self.env.reset(taskid=taskid, default=False, metric=self.config.metric_list[0], predictor=self.config.classifier_predictor_list[self.test_pred])
            self.env.pipeline.logic_pipeline_id, _ = self.agent.act(self.env.lpip_state, 'LogicPipeline', epsilon = self.epsilon_by_frame(0), not_random=True)
            if self.env.pipeline.taskid not in reward_dic:
                reward_dic[self.env.pipeline.taskid] = {'reward':{}, 'seq': {}, 'time': {}}
            reward_dic[self.env.pipeline.taskid]['reward'][self.pre_fr] = reward
            reward_dic[self.env.pipeline.taskid]['seq'][self.pre_fr] = seq
            reward_dic[self.env.pipeline.taskid]['time'][self.pre_fr] = self.end_time-self.start_time
            if need_save:
                np.save(self.config.test_reward_dic_file_name, reward_dic)
        return state, reward_dic, seq, reward, done

    # def look_res():

    def test_one_dataset(self, taskid, pre_fr=0, number = None):
        if os.path.exists(self.config.test_reward_dic_file_name):
            reward_dic = np.load(self.config.test_reward_dic_file_name,allow_pickle=True).item()
        else:
            reward_dic = {}
        if number == None:
            self.agent.load_weights(self.outputdir)
        else:
            self.agent.load_weights(self.outputdir, tag=number)
            # self.agent.test_load_weights(self.outputdir, self.dpgoutputdir, number=31000)
        score =0
        if self.config.version == 0:
            look_range = (1,15)
        elif self.config.version == 1:
            look_range = (15,29)
        elif self.config.version == 2:
            look_range = (29,43)
        elif self.config.version == 3:
            look_range = (43,57)
        for i in range(taskid,taskid+1):
            seq = []
            self.env.reset(taskid=i, default=False, metric=self.config.metric_list[0], predictor=self.config.classifier_predictor_list[self.test_pred])
            self.env.pipeline.logic_pipeline_id, _ = self.agent.act(self.env.lpip_state, 'LogicPipeline', epsilon = self.epsilon_by_frame(0), not_random=True)
            self.env.pipeline.logic_pipeline_id = 5
            state = self.env.get_state()
            for fr in range(pre_fr + 1, self.config.test_frames + 1):
                state, reward_dic, seq, reward, done = self.get_five_items_from_pipeline(fr, state, reward_dic, seq, taskid=i, need_save=False)
            score += reward
        return score
            # print('self.env.pipeline.logic_pipeline_id', self.env.pipeline.logic_pipeline_id)
            # print('reward_dict', reward_dic)
            # print('seq', seq)


    # def test(self, pre_fr=0, number = None):
    #     if os.path.exists(self.config.test_reward_dic_file_name):
    #         reward_dic = np.load(self.config.test_reward_dic_file_name,allow_pickle=True).item()
    #     else:
    #         reward_dic = {}
    #     if number == None:
    #         self.agent.load_weights(self.outputdir)
    #     else:
    #         self.agent.load_weights(self.outputdir, tag=number)
    #     score = 0
    #         # self.agent.test_load_weights(self.outputdir, self.dpgoutputdir, number=31000)
    #     if self.config.version == 0:
    #         look_range = (1,15)
    #     elif self.config.version == 1:
    #         look_range = (15,29)
    #     elif self.config.version == 2:
    #         look_range = (29,43)
    #     elif self.config.version == 3:
    #         look_range = (43,57)
    #     for i in range(look_range[0],look_range[1]):
    #         seq = []
    #         self.env.reset(taskid=i, default=False, metric=self.config.metric_list[0], predictor=self.config.classifier_predictor_list[self.test_pred])
    #         self.env.pipeline.logic_pipeline_id, _ = self.agent.act(self.env.lpip_state, 'LogicPipeline', epsilon = self.epsilon_by_frame(0), not_random=True)
    #         state = self.env.get_state()
    #         for fr in range(pre_fr + 1, self.config.test_frames + 1):
    #             state, reward_dic, seq, reward, done = self.get_five_items_from_pipeline(fr, state, reward_dic, seq, taskid=i)
    #         score += reward
    #     return score/14
    def test(self, pre_fr=0, number = None):
        if os.path.exists(self.config.test_reward_dic_file_name):
            reward_dic = np.load(self.config.test_reward_dic_file_name,allow_pickle=True).item()
        else:
            reward_dic = {}
        if pre_fr == None:
            self.agent.load_weights(self.outputdir)
        else:
            self.agent.load_weights(self.outputdir, tag=pre_fr)
        self.pre_fr = pre_fr
        score = 0
        # with open('')
        # lost_test_classification_task_dict = 
        # print(self.config.classification_task_dic['0'])
            # self.agent.test_load_weights(self.outputdir, self.dpgoutputdir, number=31000)
        for i in self.config.test_index:
            # if int(i) < 200:
            #     continue
            seq = []
            select_cl = 0
            for cid, cl in enumerate(self.config.classifier_predictor_list):
                if cl.name == self.config.classification_task_dic[i]['model']:
                    select_cl = cl
            print('taskid', i)
            self.start_time = time.time()
            self.env.reset(taskid=i, default=False, metric=self.config.metric_list[0], predictor=select_cl)
            self.env.pipeline.logic_pipeline_id, _ = self.agent.act(self.env.lpip_state, 'LogicPipeline', epsilon = self.epsilon_by_frame(0), not_random=True)
            state = self.env.get_state()
            for fr in range(pre_fr + 1, pre_fr + 7):
                state, reward_dic, seq, reward, done = self.get_five_items_from_pipeline(fr, state, reward_dic, seq, taskid=i)
            score += reward
        return score/14
    def test_one_task(self,task_id):
        self.agent.load_weights(self.outputdir, tag='56000')
        self.pre_fr = 0
        score = 0
        reward_dic = {}
        # with open('')
        # lost_test_classification_task_dict = 
        # print(self.config.classification_task_dic['0'])
            # self.agent.test_load_weights(self.outputdir, self.dpgoutputdir, number=31000)
        for i in self.config.test_index:
            # if int(i) < 200:
            #     continue
            if i != task_id:
                continue
            seq = []
            select_cl = 0
            for cid, cl in enumerate(self.config.classifier_predictor_list):
                if cl.name == self.config.classification_task_dic[i]['model']:
                    select_cl = cl
            print('taskid', i)
            self.start_time = time.time()
            self.env.reset(taskid=i, default=False, metric=self.config.metric_list[0], predictor=select_cl)
            self.env.pipeline.logic_pipeline_id, _ = self.agent.act(self.env.lpip_state, 'LogicPipeline', epsilon = self.epsilon_by_frame(0), not_random=True)
            state = self.env.get_state()
            for fr in range(pre_fr + 1, pre_fr + 7):
                state, reward_dic, seq, reward, done = self.get_five_items_from_pipeline(fr, state, reward_dic, seq, taskid=i,need_save=False)
            score += reward

    def inference(self, data_path):
        self.agent.load_weights(self.outputdir, tag='56000')
        self.pre_fr = 0
        score = 0
        reward_dic = {}
        # with open('')
        # lost_test_classification_task_dict = 
        # print(self.config.classification_task_dic['0'])
            # self.agent.test_load_weights(self.outputdir, self.dpgoutputdir, number=31000)
        datasetname = data_path.split("/")[-2]
        for taskid in self.config.classification_task_dic:
            if datasetname == self.config.classification_task_dic[taskid]['dataset']:
                i = taskid
        seq = []
        select_cl = 0
        for cid, cl in enumerate(self.config.classifier_predictor_list):
            if cl.name == self.config.classification_task_dic[i]['model']:
                select_cl = cl
        print('taskid', i)
        self.start_time = time.time()
        self.env.reset(taskid=i, default=False, metric=self.config.metric_list[0], predictor=select_cl)
        self.env.pipeline.logic_pipeline_id, _ = self.agent.act(self.env.lpip_state, 'LogicPipeline', epsilon = self.epsilon_by_frame(0), not_random=True)
        state = self.env.get_state()
        for fr in range(self.pre_fr + 1, self.pre_fr + 7):
            state, reward_dic, seq, reward, done = self.get_five_items_from_pipeline(fr, state, reward_dic, seq, taskid=i,need_save=False)
        score += reward

        return seq
        # return score/14