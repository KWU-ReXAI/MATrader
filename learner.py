import os
import logging
import itertools
import numpy as np
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

from network import TD3_network
from utils.memory import TD3_MemoryBuffer
class TD3_Agent:
	def __init__(self, stock_codes:list, num_of_stock, phase, testNum, quarter, train_chart_data,test_chart_data, training_data, test_data,
				delayed_reward_threshold=.05, balance =10000, lr = 0.001,output_path='', test=False,
				value_network_path=None, policy_network_path=None, load_value_network_path = None, load_policy_network_path = None,
				window_size = 5):
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
		self.policy_network_path = policy_network_path
		self.value_network_path = value_network_path

		# buffer setting
		self.buffer = TD3_MemoryBuffer(parameters.REPLAY_MEM_SIZE)
		self.n_steps_buffer = deque()

		# parameters
		self.window_size = window_size
		self.test = test
		self.act_dim = num_of_stock
		self.inp_dim = self.test_data.shape[2] #학습 데이터 크기
		self.lr = lr
		self.balance = balance
		self.delayed_reward_threshold = delayed_reward_threshold
		self.batch_size = parameters.BATCH_SIZE
		# Create networks
		self.network = TD3_network(self.inp_dim, self.act_dim, lr, parameters.TAU, self.window_size)
		if test and os.path.exists(load_policy_network_path+'.pt'):
			self.network.load_weights(load_policy_network_path,load_value_network_path)
		# 로그 등 출력 경로
		self.output_path = output_path

	def bellman(self, rewards, target1, target2, dones,gammas):
		critic_target = np.array(target1)
		for i in range(target1.shape[0]):
			if dones[i]:
				critic_target[i] = rewards[i]
			else:
				critic_target[i] = np.minimum(critic_target[i],target2[i])
				critic_target[i] = rewards[i] + gammas[i] * critic_target[i]
		return critic_target


	def update_models(self, iters, episode,noise):
		std_value = 2
		# std_value = iters % 4 + 2
		entropy_loss, actor_loss, critic_loss,loss = 0, 0, 0, 0
		states, actions, imitation_actions, rewards, dones, next_states, _, gammas, price,_ = self.buffer.sample_batch(parameters.BATCH_SIZE)
		n_polices = self.network.actor_target_predict(next_states)
		n_polices = np.array(n_polices)
		n_polices = self.plus_update_noise(n_polices,noise,self.act_dim)
		target_q1, target_q2 = self.network.critic_target_predict(np.asarray(next_states),n_polices)
		target_q = self.bellman(rewards,target_q1,target_q2,dones,gammas)
		# Train critic
		critic_loss = self.network.critic_train(states, actions, target_q)
		
		##if t mod d then ##
		if episode % std_value == 0:
			actor_loss = self.network.actor_train(states, imitation_actions, price)
			self.network.transfer_weights()
		return entropy_loss,actor_loss, loss, critic_loss

	def plus_noise(self,policy,exploration_noise,act_size):
		for index in range(len(policy)):
			for i in range(act_size):
				noise = np.random.normal(0,exploration_noise,1)
				policy[index][i] += noise
				policy[index][i] = np.clip(policy[index][i],-1.0 * parameters.NUM_ACTIONS,1.0 * parameters.NUM_ACTIONS)
			return policy
	def plus_update_noise(self,policy,exploration_noise,act_size):
		for index in range(len(policy)):
			for i in range(act_size):
				noise = np.random.normal(0,exploration_noise,1)
				noise = np.clip(noise,-1,1)
				policy[index][i] += noise
				policy[index][i] = np.clip(policy[index][i], -1.0 * parameters.NUM_ACTIONS, 1.0 * parameters.NUM_ACTIONS)
			return policy

	def run(self,max_episode=100, reward_n_step = 1,noise=0.001,start_epsilon=0.3):
		# path setting
		csv_path = os.path.join(self.output_path, "_action_history_train_"+ str(reward_n_step)+".csv")
		plt_path = os.path.join(self.output_path,"_plt_train_" + str(reward_n_step))
		f = open(csv_path, "w"); f.write("date,stock,price,action,num_stock,portfolio_value\n")

		# environment setting
		environment = Environment(self.train_chart_data, self.training_data)
		trader = Trader(environment, self.balance, self.act_dim, delayed_reward_threshold=self.delayed_reward_threshold)

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
			self.n_steps_buffer.clear()
			self.buffer.clear()

			self.network.copy_weights()
			next_state, done = environment.build_state() # 초기 state: next_state
			update_noise = 0.7
			while True:				
				# 환경에서 가격 얻기
				curr_prices = environment.curr_price()
				next_prices = environment.next_price()

				state = next_state
				if done: break
				# Actor picks an action (following the deterministic policy) and retrieve reward
				policy = self.network.actor_predict(np.array(state))
				policy = self.plus_noise(np.array(policy),noise,policy.shape[1])
				action = policy[0]

				# Pick imitation action
				# 매매 타입 3개일 때만 적용 가능!
				imitation_action = np.zeros(self.act_dim)
				for stock, curr_price in enumerate(curr_prices):
					if next_prices is None: imitation_action[stock] = 0 # 홀딩
					elif next_prices[stock] > curr_price * (1+stock_rate): imitation_action[stock] = -2 # 매수
					elif next_prices[stock] < curr_price * (1-stock_rate): imitation_action[stock] = 2 # 매도
					else : imitation_action[stock] = 0

				# 행동 -> reward(from trading), next_state, done(from env)
				_, reward = trader.act(action, self.stock_codes, f, recode)

				next_state, done = environment.build_state() # 액션 취한 후 next_state 구하기 위함
				self.n_steps_buffer.append((state, policy, imitation_action, reward, done, next_state, environment.curr_price()))
				# act에서 env.idx + 1을 했으므로 curr_price가 next_price임

				if len(self.n_steps_buffer) >= reward_n_step:
					state, policy, imitation_action, reward, _, _, prev_price = self.n_steps_buffer.popleft()
					_, _, _, _, done, next_state, price = self.n_steps_buffer[-1]
					# done이 True이면, next_state가 None이므로 ReplayBuffer에서 뽑을 때 타입이 안 맞아서 에러남
					# 따라서 타입을 맞추기 위해 next_state = state로 넣어줌(학습에는 쓰이지 않음)
					# price 또한 마지막 날(done == True)에는 None이므로 에러가 나서 price = prev_price
					# 물론 price는 마지막 날이어도 학습에 포함되므로 임시로 아래와 같이 설정, 추후 개선 예정
					if done:
						next_state = state
						price = prev_price

					# discount_reward = R_t+1 + r*R_t+2 + ... r^n-1 * R_t+n
					# gamma = r^n
					discount_reward = reward
					gamma = parameters.GAMMA
					for(_,_,_,r_i,_,_,_) in self.n_steps_buffer:
						discount_reward += r_i * gamma
						gamma *= parameters.GAMMA
					# Add outputs to memory buffer
					# ReplayBuffer는 state, action, critic_state, reward, done, next_state, critic_next_state, gamma, price 가 입력
					# critic_***는 사실 필요없으므로 critic_state 자리엔 imitation_action을, critic_next_state에는 그냥 next_state를 한번 더 넣은 것
					# 이게 좀 더럽다 싶으면 utils/memory에서 buffer를 고치면 되지만... 일단은 이렇게 둠
					self.buffer.memorize(state, policy[0], imitation_action, discount_reward,done,next_state,next_state,gamma,price)
					self.update_models(e, episode,update_noise) 
				episode += 1

			max_episode_digit = len(str(max_episode))
			epoch_str = str(e + 1).rjust(max_episode_digit, '0')
			logging.info("[{}]-[{} - phase_{}_{} {}][Epoch {}/{}][EPSILON {:.5f}]"
				"#Buy:{} #Sell:{} #Hold:{} "
				"#Stocks:{} PV:{:,.0f}".format(threading.currentThread().getName(),
					'_'.join(self.stock_codes), self.phase, self.testNum, self.quarter, epoch_str, max_episode,epsilon,
					trader.num_buy, trader.num_sell, trader.num_hold,
					trader.num_stocks,	trader.portfolio_value))
			if recode == 1 :
				environment.plt_result(plt_path, self.stock_codes)
				return_pv = trader.portfolio_value
			store_policy_network_path = self.policy_network_path + "_"+str(e)
			store_critic_network_path = self.value_network_path + "_"+str(e)

			print(store_policy_network_path)
			self.network.save(store_policy_network_path,store_critic_network_path)
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
		trader = Trader(environment, self.balance, self.act_dim, delayed_reward_threshold=self.delayed_reward_threshold)

		# reset
		environment.reset()
		trader.reset()

		memory_pv = []; memory_reward = []
		state, done = environment.build_state()
		while True:
			if done: break
			# Actor picks an action (following the deterministic policy) and retrieve reward
			policy = self.network.actor_predict(np.array(state))
			action= policy[0]
			reward, _ = trader.act(action, self.stock_codes, f, 1)
			memory_reward.append(reward)
			memory_pv.append(trader.portfolio_value)
			state, done = environment.build_state()
		# Sharpe Ratio
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

		return trader.portfolio_value