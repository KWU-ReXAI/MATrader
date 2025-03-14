import os
import logging
import numpy as np
from tqdm import tqdm
from collections import deque
import threading
import random
from trading import Trader
from environment import Environment
from parameters import Agent_Memory, parameters
# from feature_network import SDAE
# import wandb

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
lock = threading.Lock()
random.seed(42)
np.random.seed(42)

from network import TD3_network
from utils.memory import TD3_MemoryBuffer
class TD3_Agent:
	def __init__(self, stock_code,chart_data,training_data, delayed_reward_threshold=.05,
				balance =10000, lr = 0.001,output_path='', reuse_model=True,value_network_path=None,
				policy_network_path=None, load_value_network_path = None, load_policy_network_path = None, window_size = 5):
		#Initialization
		# 환경 설정
		self.chart_data = chart_data
		self.stock_code = stock_code
		
		self.policy_network_path = policy_network_path
		self.value_network_path = value_network_path
		self.window_size = window_size
		self.reuse_model = reuse_model
		#environment setting
		self.environment = Environment(chart_data, training_data)
		self.trader = Trader(self.environment, balance, delayed_reward_threshold=delayed_reward_threshold)
		self.memory = Agent_Memory()
		self.buffer = TD3_MemoryBuffer(parameters.REPLAY_MEM_SIZE)
		self.n_steps_buffer = deque()
		self.training_data = training_data
		self.sample = None
		self.training_data_idx = -1
		# Environment and parameters
		self.act_dim = parameters.NUM_ACTIONS
		self.inp_dim = self.training_data.shape[2] #학습 데이터 크기
		self.lr = lr
		self.delayed_reward_threshold = delayed_reward_threshold
		self.batch_size = parameters.BATCH_SIZE
		# Create networks
		self.network = TD3_network(self.inp_dim, self.act_dim, lr, parameters.TAU, self.window_size)
		if reuse_model and os.path.exists(load_value_network_path):
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
		std_value = iters % 4 + 2
		entropy_loss, actor_loss, critic_loss,loss = 0, 0, 0, 0
		states, actions, imitation_actions, rewards, dones, next_states, _, gammas, price,_ = self.buffer.sample_batch(parameters.BATCH_SIZE)
		n_polices = self.network.actor_target_predict(next_states)
		n_polices = np.array(n_polices)
		n_polices = self.plus_update_noise(n_polices,noise,self.act_dim)
		target_q1, target_q2 = self.network.critic_target_predict(np.asarray(next_states),n_polices)
		target_q = self.bellman(rewards,target_q1,target_q2,dones,gammas)
		# Train critic
		#self.wandb.log({"critic_loss":critic_loss})
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
			return policy
	def plus_update_noise(self,policy,exploration_noise,act_size):
		for index in range(len(policy)):
			for i in range(act_size):
				noise = np.random.normal(0,exploration_noise,1)
				noise = np.clip(noise,-1,1)
				policy[index][i] += noise
			return policy

	def run(self,max_episode=100, reward_n_step = 1,noise=0.001,start_epsilon=0.3):
		#db_name = "{}-TD3_discrete_imitation [noise {:.1f}] [window size {}] [lr {:.3f}]".format(self.stock_code,noise, self.window_size, self.lr)
		csv_path = os.path.join(self.output_path, "_action_history_train_"+ str(reward_n_step)+".csv")
		plt_path = os.path.join(self.output_path,"_plt_train_" + str(reward_n_step))
		#self.wandb = wandb.init(project="TD3",name=db_name)
		f = open(csv_path, "w"); f.write("date,price,action,num_stock,portfolio_value\n")

		info = "[{code}] LR:{lr} " \
			"DRT:{delayed_reward_threshold}".format(
			code=self.stock_code, lr=self.lr,
			delayed_reward_threshold=self.delayed_reward_threshold
		)
		logging.info(info)
		epsilon = start_epsilon
		tqdm_e = tqdm(range(max_episode), desc='Score', leave=True, unit=" episodes")
		store_policy_network_path = ' '; store_ciritc_network_path = ' '
		stock_rate = 0.001
		for e in tqdm_e:
			# Reset episode
			imitation_action = [0,0,1]
			episode, recode =0, 0
			pv = 0
			if e == (max_episode -1) : recode = 1
			self.reset()
			self.network.copy_weights()
			next_state, done = self.environment.build_state() # 초기 state: next_state
			update_noise = 0.7
			while True:				
				# 환경에서 가격 얻기
				date = self.environment.get_date()
				curr_price = self.environment.curr_price()
				next_price = self.environment.next_price()

				state = next_state
				if done: break
				# Actor picks an action (following the deterministic policy) and retrieve reward
				policy = self.network.actor_predict(np.array(state))
				policy = self.plus_noise(np.array(policy),noise,policy.shape[1])
				action, confidence = self.network.select_action(policy)

				# Pick imitation action
				if next_price == None: imitation_action = [0,0,1]
				elif next_price > curr_price * (1+stock_rate): imitation_action = [1,0,0]; 
				elif next_price < curr_price * (1-stock_rate): imitation_action = [0,1,0]; 
				else : imitation_action = [0,0,1]; 

				# 행동 -> reward(from trader), next_state, done(from env)
				reward, _ = self.trader.act(action, confidence, f, recode)
				next_state, done = self.environment.build_state() # 액션 취한 후 next_state 구하기 위함
				self.n_steps_buffer.append((state, policy, imitation_action, reward, done, next_state, next_price))
				# next_price 저장 이유: price는 actor의 price network에서 state(X일 전부터 오늘까지 가격)로부터 내일의 종가를 예측하는 모델을 만들기 위함
				# 따라서 오늘의 가격이 아닌, 내일의 종가를 주어야 함

				if len(self.n_steps_buffer) >= (parameters.N_STEP_RETURNS):
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
			pv = self.trader.balance + self.trader.prev_price * self.trader.num_stocks * (1- parameters.TRADING_TAX)

			max_episode_digit = len(str(max_episode))
			epoch_str = str(e + 1).rjust(max_episode_digit, '0')
			logging.info("[{}]-[{}][Epoch {}/{}][EPSILON {:.5f}]"
				"#Buy:{} #Sell:{} #Hold:{} "
				"#Stocks:{} PV:{:,.0f}".format(threading.currentThread().getName(),
					self.stock_code, epoch_str,max_episode,epsilon,self.trader.num_buy,
					self.trader.num_sell, self.trader.num_hold, self.trader.num_stocks,
					pv))
			if recode == 1 : self.environment.plt_result(plt_path)
			store_policy_network_path = self.policy_network_path + "_"+str(e)
			store_ciritc_network_path = self.value_network_path + "_"+str(e)

			print(store_policy_network_path)
			self.network.save(store_policy_network_path,store_ciritc_network_path)
		
	def reset(self):
		self.environment.reset()
		self.trader.reset()
		self.n_steps_buffer.clear()
		self.buffer.clear()