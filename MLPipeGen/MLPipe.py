import json
import numpy as np
import os
import shutil
from MLPipeGen.core.tester import Tester
from MLPipeGen.core.config import Config
from MLPipeGen.core.agent.dqn import DQNAgent
from MLPipeGen.core.env.enviroment import Environment
class MLPipe:
    def __init__(self, data_path, label_index):
        self.config = Config()
        self.agent = DQNAgent(self.config.version, self.config)
        self.env = Environment(self.config,train=False)
        self.tester = Tester(self.agent, self.env, 0, self.config)
        self.data_path = data_path
        self.label_index =label_index

    def inference(self):
        return self.tester.inference(self.data_path)