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
import talib
# learner.py 파일 상단에 RealtimeEnvironment 임포트 추가
from environment import Environment, RealtimeEnvironment
# 'time'과 'datetime'도 임포트 되어있는지 확인
import time
from datetime import datetime
import time
from datetime import datetime

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
				window_size = 5, api_handler=None):
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

		self.api = api_handler  # api_handler를 클래스 속성으로 저장

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
				"""for stock, curr_price in enumerate(curr_prices):
					if next_prices is None: imitation_action[stock] = 0 # 홀딩
					elif next_prices[stock] > curr_price * (1+stock_rate): imitation_action[stock] = -2 # 매수
					elif next_prices[stock] < curr_price * (1-stock_rate): imitation_action[stock] = 2 # 매도
					else : imitation_action[stock] = 0"""
				if environment.idx >= 20:
					for stock_idx in range(self.act_dim):
						# 이동평균 계산에 필요한 과거 종가 데이터 추출 (현재 포함 21일치)
						# chart_data shape: (거래일, 종목수, 피처), 종가(PRICE_IDX)는 4번 인덱스
						prices = environment.chart_data[environment.idx-20 : environment.idx+1, stock_idx, 4].astype('double')

						# ta-lib으로 단기(5일), 장기(20일) 이동평균 시리즈 계산
						short_ma = talib.SMA(prices, timeperiod=5)
						long_ma = talib.SMA(prices, timeperiod=20)

						# 교차를 확인하기 위해 현재 값(-1)과 직전 값(-2)을 사용
						# Golden Cross (매수): 이전에는 단기<장기, 현재는 단기>장기
						if short_ma[-2] < long_ma[-2] and short_ma[-1] > long_ma[-1]:
							imitation_action[stock_idx] = -2  # 매수 신호
						# Dead Cross (매도): 이전에는 단기>장기, 현재는 단기<장기
						elif short_ma[-2] > long_ma[-2] and short_ma[-1] < long_ma[-1]:
							imitation_action[stock_idx] = 2  # 매도 신호
						else:
							imitation_action[stock_idx] = 0  # 홀딩

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

	def trade_realtime(self):
		"""
        실시간으로 주식 데이터를 받아 모델의 예측에 따라 자동으로 매매를 수행합니다.
        """
		logging.info("Starting realtime trading...")

		# 1. 실시간 환경 구성
		environment = RealtimeEnvironment(
			api_handler=self.api,
			stock_codes=self.stock_codes,
			fmpath=os.path.join(self.output_path, 'feature_model'),  # 학습된 피처 모델 경로
			window_size=self.window_size,
			feature_window=1  # feature.py의 기본값과 동일하게 설정
		)

		# 2. 매매 루프 시작 (장 시작 ~ 장 마감)
		while True:
			try:
				# 현재 시간 확인
				now = datetime.now().time()

				# 장 운영 시간: 09:00 ~ 15:30
				if now < datetime.strptime("09:00", "%H:%M").time() or now > datetime.strptime("15:30", "%H:%M").time():
					logging.info("Market is closed. Waiting...")
					time.sleep(60)
					continue

				# 3. 최신 시장 상태 받아오기
				state, done = environment.build_state()
				if done:
					logging.error("Failed to build state from market data. Retrying in 1 minute.")
					time.sleep(60)
					continue

				# 4. 모델을 통해 행동 결정
				policy = self.network.actor_predict(np.array(state))
				action = policy[0]
				logging.info(f"Model action policy: {action}")

				# 5. 행동에 따른 주문 실행
				current_prices = environment.curr_price()
				account_balance = self.api.get_account_balance()

				if account_balance is None:
					logging.error("Failed to get account balance.")
					time.sleep(60)
					continue

				cash_per_stock = account_balance['deposit'] // len(self.stock_codes)

				# 보유 주식 정보를 dictionary 형태로 변환하여 쉽게 접근
				owned_stocks = {stock['iscd']: stock for stock in account_balance.get('holdings', [])}

				for i, stock_code in enumerate(self.stock_codes):
					decision = action[i]
					current_price = int(current_prices[i])

					# 매수 결정 (action 값이 특정 임계값 이상일 때)
					if decision > 1.5:  # 매수 강도 임계값 (조정 가능)
						if cash_per_stock > current_price:
							qty_to_buy = cash_per_stock // current_price
							logging.info(f"Attempting to BUY {qty_to_buy} shares of {stock_code} at {current_price}")
							response = self.api.order_cash(stock_code, qty_to_buy, current_price, "02")  # "02": 매수
							logging.info(f"BUY Order Response: {response}")
						else:
							logging.info(f"Not enough cash to buy {stock_code}.")

					# 매도 결정 (action 값이 특정 임계값 이하일 때)
					elif decision < -1.5:  # 매도 강도 임계값 (조정 가능)
						if stock_code in owned_stocks:
							qty_to_sell = owned_stocks[stock_code]['qty']
							logging.info(f"Attempting to SELL {qty_to_sell} shares of {stock_code} at {current_price}")
							response = self.api.order_cash(stock_code, qty_to_sell, current_price, "01")  # "01": 매도
							logging.info(f"SELL Order Response: {response}")
						else:
							logging.info(f"No shares of {stock_code} to sell.")

					# 홀딩
					else:
						logging.info(f"Holding {stock_code}.")

			except Exception as e:
				logging.error(f"An error occurred in the trading loop: {e}", exc_info=True)

			# 6. 다음 1분까지 대기
			logging.info("Waiting for the next minute...")
			time.sleep(60)
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲