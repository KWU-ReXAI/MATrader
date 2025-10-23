import os
import logging
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import deque
import threading
import random
from trading import Trader
from environment import Environment
from parameters import parameters

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
lock = threading.Lock()
random.seed(42)
np.random.seed(42)

from network import MultiAgentTransformer
from utils.memory import PPOReplayBuffer
class MATagent:
	def __init__(self, stock_codes:list, num_of_stock, phase, testNum, quarter, train_chart_data,test_chart_data, training_data, test_data,
				delayed_reward_threshold=.05, balance =10000, lr = 0.001,output_path='', test=False,
				network_path=None, load_network_path = None, window_size = 5, ppo_epoch=15, num_mini_batch=1):
		# data
		self.phase = phase
		self.testNum = testNum
		self.quarter = quarter
		self.train_chart_data = train_chart_data
		self.test_chart_data = test_chart_data
		self.training_data = training_data
		self.test_data = test_data

		# paths and stock code
		self.stock_codes = stock_codes
		self.network_path = network_path

		# buffer setting
		self.buffer = PPOReplayBuffer(num_of_stock, self.test_data.shape[2], parameters.NUM_ACTIONS)

		# parameters
		self.ppo_epoch = ppo_epoch
		self.num_mini_batch = num_mini_batch
		self.window_size = window_size
		self.test = test
		self.act_dim = parameters.NUM_ACTIONS
		self.n_agents = num_of_stock
		self.inp_dim = self.test_data.shape[2] #학습 데이터 크기
		self.lr = lr
		self.balance = balance
		self.delayed_reward_threshold = delayed_reward_threshold
		self.batch_size = parameters.BATCH_SIZE
		# Create networks
		self.network = MultiAgentTransformer(self.test_data.shape[2], parameters.NUM_ACTIONS, num_of_stock, 1, 64, 1, lr=lr)
		if test and os.path.exists(load_network_path+'.pt'):
			self.network.load_model(load_network_path)
		# 로그 등 출력 경로
		self.output_path = output_path

	def run(self,max_episode=100, reward_n_step = 1,noise=0.001,start_epsilon=0.3):
		# path setting
		csv_path = os.path.join(self.output_path, "_action_history_train_"+ str(reward_n_step)+".csv")
		plt_path = os.path.join(self.output_path,"_plt_train_" + str(reward_n_step))
		f = open(csv_path, "w"); f.write("date,stock,price,action,num_stock,portfolio_value\n")

		# environment setting
		environment = Environment(self.train_chart_data, self.training_data)
		trader = Trader(environment, self.balance, self.n_agents, delayed_reward_threshold=self.delayed_reward_threshold)

		info = "[{code} - phase_{phase}_{testNum} {quarter}] LR:{lr} " \
			"DRT:{delayed_reward_threshold}".format(
			code='_'.join(self.stock_codes), lr=self.lr,
			delayed_reward_threshold=self.delayed_reward_threshold,
			phase=self.phase, testNum=self.testNum, quarter=self.quarter
		)
		logging.info(info)
		epsilon = start_epsilon
		tqdm_e = tqdm(range(max_episode), desc='Score', leave=True, unit=" episodes")
		stock_rate = 0.001
		return_pv = 0.0
		for e in tqdm_e:
			# Reset episode
			episode, recode =0, 0
			if e == (max_episode -1) : recode = 1
			# reset
			environment.reset()
			trader.reset()
			self.buffer.clear()

			next_state, done = environment.build_state() # 초기 state: next_state
			while True:
				state = next_state
				if done: break
				# Actor picks an action (following the deterministic policy) and retrieve reward
				action, action_log_prob, value_pred  = self.network.act(np.array(state))

				# 행동 -> reward(from trading), next_state, done(from env)
				_, future_reward, sharpe_reward = trader.act(action[0], self.stock_codes, f, recode)

				next_state, done = environment.build_state() # 액션 취한 후 next_state 구하기 위함
				self.buffer.add(state, action[0], sharpe_reward, done, action_log_prob[0], value_pred[0])
				episode += 1

			last_value = np.zeros(self.n_agents)
			self.buffer.compute_returns_and_advantages(last_value)
			total_policy_loss, total_value_loss = 0, 0
			for ppo_epoch in range(self.ppo_epoch):
				# 버퍼에서 미니배치 샘플링
				data_generator = self.buffer.sample(num_mini_batch=self.num_mini_batch)

				for sample in data_generator:
					obs_batch, actions_batch, old_action_log_probs_batch, \
						value_preds_batch, return_batch, adv_targ = sample

					# 모델 업데이트
					loss_info = self.network.update(
						obs_batch,
						actions_batch,
						old_action_log_probs_batch,
						value_preds_batch,
						return_batch,
						adv_targ
					)
					total_policy_loss += loss_info['policy_loss']
					total_value_loss += loss_info['value_loss']

			# 학습에 사용된 데이터 폐기
			self.buffer.clear()

			max_episode_digit = len(str(max_episode))
			epoch_str = str(e + 1).rjust(max_episode_digit, '0')
			avg_policy_loss = total_policy_loss / (self.ppo_epoch * self.num_mini_batch)
			avg_value_loss = total_value_loss / (self.ppo_epoch * self.num_mini_batch)
			logging.info("[{}]-[{} - phase_{}_{} {}][Epoch {}/{}][Policy Loss: {:.3f} Value Loss: {:.3f}]"
				"#Buy:{} #Sell:{} #Hold:{} "
				"#Stocks:{} PV:{:,.0f}".format(threading.currentThread().getName(),
					'_'.join(self.stock_codes), self.phase, self.testNum, self.quarter, epoch_str, max_episode,
					avg_policy_loss, avg_value_loss,
					trader.num_buy, trader.num_sell, trader.num_hold,
					trader.num_stocks,	trader.portfolio_value))
			if recode == 1 :
				environment.plt_result(plt_path, self.stock_codes)
				return_pv = trader.portfolio_value
				store_network_path = self.network_path + "_"+str(e)
				print(store_network_path)
				self.network.save_model(store_network_path)
		return return_pv

	def backtest(self, reward_n_step=1):
		csv_path = os.path.join(self.output_path, "td3_test_action_history_" + str(reward_n_step) + ".csv")
		plt_path = os.path.join(self.output_path, "td3_test_plt_" + str(reward_n_step))
		f = open(csv_path, "w")
		f.write("date,stock,price,action,num_stock,portfolio_value\n")

		info = "[{code} - phase_{phase}_{testNum} {quarter}] LR:{lr} " \
			   "DRT:{delayed_reward_threshold}".format(
			code='_'.join(self.stock_codes), lr=self.lr,
			delayed_reward_threshold=self.delayed_reward_threshold,
			phase = self.phase, testNum = self.testNum, quarter=self.quarter
		)
		logging.info(info)

		# environment setting
		environment = Environment(self.test_chart_data, self.test_data)
		trader = Trader(environment, self.balance, self.n_agents, delayed_reward_threshold=self.delayed_reward_threshold)

		# reset
		environment.reset()
		trader.reset()

		memory_pv = []; memory_reward = []
		state, done = environment.build_state()
		while True:
			if done: break
			# Actor picks an action (following the deterministic policy) and retrieve reward
			action, _, _ = self.network.act(np.array(state))
			# 행동 -> reward(from trading), next_state, done(from env)
			reward, *_ = trader.act(action[0], self.stock_codes, f, 1)
			memory_reward.append(reward)
			memory_pv.append(trader.portfolio_value)
			state, done = environment.build_state()
		# Sharpe Ratio
		rr = trader.portfolio_value / self.balance - 1
		sr = 0 if len(memory_reward) <= 1 else np.mean(memory_reward) * np.sqrt(len(memory_reward))/ (np.std(memory_reward) + 1e-10)
		mdd = 0 if len(memory_pv) <= 1 else max((peak_pv - pv) / peak_pv for peak_pv, pv \
												in zip(itertools.accumulate(memory_pv, max), memory_pv))
		logging.info("[{}]-[{} - phase_{}_{} {}]"
					 "#Buy:{} #Sell:{} #Hold:{} "
					 "#Stocks:{} PV:{:,.0f} SR {:.2f} MDD {:.2f}".format(threading.currentThread().getName(), '_'.join(self.stock_codes),
															  self.phase, self.testNum, self.quarter, trader.num_buy, trader.num_sell,
															  trader.num_hold,
															  trader.num_stocks, trader.portfolio_value,
															  sr, mdd))
		environment.plt_result(plt_path, self.stock_codes)
		df = pd.DataFrame({'quarter': [self.quarter], 'rr': [rr], 'sr': [sr], 'mdd': [mdd]})
		return df