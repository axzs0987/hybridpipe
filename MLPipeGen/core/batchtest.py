from trainer import Trainer
from tester import Tester
from agent.dqn import DQNAgent
from env.enviroment import Environment
from config import Config

if __name__ == '__main__':
    # dqn.py --train --env CartPole-v0
    config = Config()
    agent = DQNAgent(config.version, config)

    # start_number = 25000
    # while(start_number <= 30500):
    # config.outputdir
    test_env = Environment(config,train=False)
    tester = Tester(agent, test_env, 0, config)
    tester.test(56000)