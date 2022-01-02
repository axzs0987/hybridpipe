from trainer import Trainer
from tester import Tester
from agent.dqn import DQNAgent
from env.enviroment import Environment
from config import Config
import json

if __name__ == '__main__':
    # dqn.py --train --env CartPole-v0
    config = Config()

    # env = Environment(config)
    agent = DQNAgent(config.version, config)
    is_train = True
    if is_train==True:
        env = Environment(config)
        trainer = Trainer(agent, env, 0, config)
        # trainer.train(pre_fr=0) # 2.v1
        # trainer.train(pre_fr=7000)
        trainer.train(pre_fr=67000)
        # trainer.train(pre_fr=15302) # 2.v1
        # trainer.train(pre_fr=29265) # 2.v2
        # trainer.train(pre_fr=52421) # 2.v0
        # trainer.train(pre_fr=540)
        # trainer.train(pre_fr=0) # loss v2
    if is_train == False:
        # num = 28500
        # while num <= 34500:
        num = 0
        max_num={}
        max_score={}
        
        max_add_all = 0
        max_add_num = 0
        max_num[0] = 0
        max_num[4] = 0
        max_num[9] = 0
        max_num[12] = 0

        max_score[0]=0
        max_score[4]=0
        max_score[9]=0
        max_score[12]=0
        while num <= 99500:
            try:
                all_score = 0
        
                for pred in [0,4,9,12]:
                    test_env = Environment(config,train=False)
                    tester = Tester(agent, test_env, config=config, test_pred=pred)
                    score = tester.test(number=num)
                    num += 500
                    if score > max_score[pred]:
                        max_score[pred] = score
                        max_num[pred] = num
                    all_score += score
                if all_score > max_add_all:
                    max_add_all = all_score
                    max_add_num = num
            except:
                break
            
            
            
        # res = {}
        # res['model_best_score'] = max_score
        # res['model_best_num'] = max_num
        # res['all_best_score'] = max_add_all
        # res['all_best_num'] = max_add_num
        # with open(config.outputdir + '_testresult.json', 'w') as f:
        #     json.dump(res, f)

    if is_train==4:
        # try:
            num=3500 #v3 1
            # num=57500 #v1 4
            # num=33500 #v2 2
            # num=41500 #v0 4
            res_score = {}
            time=0
            while time < 5:
                for pred in [0]:
                    mean=0
                    test_env = Environment(config,train=False)
                    tester = Tester(agent, test_env, config=config, test_pred=pred)
                    score = tester.test(number=num)
                    res_score[pred] = score
                with open('res_'+str(config.version)+"_"+str(time)+".json", 'w') as f:
                    json.dump(res_score,f)
        # except:
        #     print('error')

    if is_train == 3:
        test_env = Environment(config,train=False)
        tester = Tester(agent, test_env, config=config,test_pred=12)
    
        tester.test_one_dataset(taskid=9,pre_fr=0,number=10500)
    
    if is_train == 5:
        test_env = Environment(config,train=False)

        best_num = 0
        res_score = {}
        time=0

        best_score = 0
        tester = Tester(agent, test_env, config=config,test_pred=0)

        num=500
        while num <= 100000:
            try:
                mean=0
                for taskid in range(1,57):
                    
                    score = tester.test_one_dataset(taskid=taskid,pre_fr=0,number=num)
                    mean += score
                    with open('res_'+str(config.version)+"_"+str(time)+".json", 'w') as f:
                        json.dump(res_score,f)
                
                if mean > best_score:
                    best_score = mean
                    best_num = num

                num+=500
            except:
                break   
        
    
        tester.test_one_dataset(taskid=9,pre_fr=0,number=10500)
    # trainer.train()

    # elif args.test:
    #     if args.model_path is None:
    #         print('please add the model path:', '--model_path xxxx')
    #         exit(0)
    #     tester = Tester(agent, env, args.model_path)
    #     tester.test()
