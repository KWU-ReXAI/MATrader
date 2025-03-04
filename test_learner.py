import os
import logging
import numpy as np
import math
import tensorflow as tf
import keras
from tqdm import tqdm
from collections import deque
import threading
import random
from trading import Trader
from parameters import Agent_Memory, parameters
# from feature_network import SDAE
from environment import Environment
import wandb

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
lock = threading.Lock()
random.seed(42)
np.random.seed(42)

from test_network import TD3_network
from utils.memory import TD3_MemoryBuffer
class TD3_Agent:
    def __init__(self, stock_code,chart_data,training_data, delayed_reward_threshold=.05,
                balance =10000, lr = 0.001,output_path='',
                load_value_network_path = None, load_policy_network_path = None, window_size = 5):
        #Initialization
        # 환경 설정
        self.chart_data = chart_data
        self.stock_code = stock_code
        self.window_size = window_size
        
        #environment setting
        self.environment = Environment(chart_data, training_data)
        self.trader = Trader(self.environment, balance, delayed_reward_threshold=delayed_reward_threshold)
        self.training_data = training_data
        self.sample = None
        self.training_data_idx = -1

        # Environment and parameters
        self.act_dim = parameters.NUM_ACTIONS
        self.inp_dim = self.training_data.shape[2] #학습 데이터 크기
        self.lr = lr
        self.delayed_reward_threshold = delayed_reward_threshold
        # Create networks
        self.network = TD3_network(self.inp_dim, self.act_dim, lr, parameters.TAU, self.window_size)
        self.network.load_weights(load_policy_network_path,load_value_network_path)
        
        # 로그 등 출력 경로
        self.output_path = output_path

    def run(self,max_episode=1, reward_n_step = 1,noise=0.001):
        #db_name = "{}-TD3".format(self.stock_code)
        csv_path = os.path.join(self.output_path, "td3_test_action_history_"+ str(reward_n_step)+".csv")
        plt_path = os.path.join(self.output_path,"td3_test_plt_" + str(reward_n_step))
        #self.wandb = wandb.init(project="TD3",name=db_name)
        f = open(csv_path, "w"); f.write("date,price,action,num_stock,portfolio_value\n")

        info = "[{code}] LR:{lr} " \
            "DRT:{delayed_reward_threshold}".format(
            code=self.stock_code, lr=self.lr,
            delayed_reward_threshold=self.delayed_reward_threshold
        )
        logging.info(info)
        done =  False
        self.reset(); memory_reward = []
        state,done = self.environment.build_state()
        while True:
            if done: break
            # Actor picks an action (following the deterministic policy) and retrieve reward
            policy = self.network.actor_predict(np.array(state))
            #noise_action = self.plus_noise(np.array(policy),noise,policy.shape[1])
            action, confidence = self.network.select_action(policy)
			print("action: ", action)
			print("confidence", confidence)
			print()
            reward,_ = self.trader.act(action,confidence, f, 1)
            memory_reward.append(reward)
            state,done = self.environment.build_state()
        pv = self.trader.balance + self.trader.prev_price * self.trader.num_stocks * (1- parameters.TRADING_TAX)
        sr =  np.mean(memory_reward) / (np.std(memory_reward) + 1e-10)
        sr *= np.sqrt(len(self.chart_data))
        logging.info("[{}]-[{}]"
            "#Buy:{} #Sell:{} #Hold:{} "
            "#Stocks:{} PV:{:,.0f} SR {:.5f}".format(threading.currentThread().getName(),
                self.stock_code, self.trader.num_buy,
                self.trader.num_sell, self.trader.num_hold, self.trader.num_stocks,pv,sr))
        #wandb.log({"reward":self.trader.portfolio_value, "sharpe_ratio":sharpe_ratio})
        self.environment.plt_result(plt_path)

    def reset(self):
        self.environment.reset()
        self.trader.reset()
        