import numpy as np
from parameters import parameters
from collections import deque

class Trader:
	def __init__(self, environment, balance, n_agents, delayed_reward_threshold=.05):
		# 환경
		self.environment = environment

		# 지연보상 임계치
		self.delayed_reward_threshold = delayed_reward_threshold

		# Trader 클래스의 속성
		self.initial_balance = balance  # 초기 자본금
		self.n_agents = n_agents # 거래하는 종목 수
		self.act_dim = parameters.NUM_ACTIONS # 종목별 거래 타입 개수(3개: 매수, 매도, 홀딩)

		# 포트폴리오 관련
		#self.balance = np.full(n_agents, balance // n_agents, dtype=np.int64)  # 종목별 잔고: 동등하게 분배
		self.balance = balance

		self.num_stocks = np.zeros(n_agents, dtype=np.int64)  # 종목별 보유 주식 수
		# 포트폴리오 가치: balance + num_stocks * {현재 주식 가격} * (1-수수료)
		self.portfolio_value = balance
		self.prev_portfolio_value = balance

		self.num_buy = 0  # 매수 횟수
		self.num_sell = 0  # 매도 횟수
		self.num_hold = 0  # 홀딩 횟수

		self.initial_stock_balance = balance // n_agents


		# 샤프지수 계산을 위한 최근 수익률 저장소
		self.reward_window = 20
		self.reward_history = deque(maxlen=self.reward_window)

	def reset(self):
		#self.balance = np.full(self.n_agents, self.initial_balance // self.n_agents, dtype=np.int64)
		self.balance = self.initial_balance
		self.num_stocks = np.zeros(self.n_agents, dtype=np.int64)
		self.portfolio_value = self.initial_balance
		self.prev_portfolio_value = self.initial_balance
		self.num_buy = 0
		self.num_sell = 0
		self.num_hold = 0

		self.reward_history.clear()

	def map_action(self, action):
		# 매매 타입 가지수(매수, 매도, 홀딩)에 따른 범위의 경계값 생성
		# 예: action이 -3~3이라면 points = [-3, -1, 1, 3]
		points = np.linspace(-1 * self.act_dim, self.act_dim, self.act_dim + 1)
		points[-1] += 1e-10 # 경계값 예외 위함
		# values: 범위에 속하는 위치
		# 예: -2.8 -> [-3, -1] 사이 => 1번째 범위
		values = np.digitize(action, points)
		values -= 1
		return values

	def validate_action(self, action, buy_index, sell_index):
		if len(buy_index) > 0:
			# 사는 종목들이 각각 적어도 1주를 살 수 있는지 확인
			for index in buy_index:
				if self.balance // self.n_agents < self.environment.curr_price()[index]:
					action[index] = parameters.ACTION_HOLD
		if len(sell_index) > 0:
			# 주식 잔고가 있는지 확인
			for index in sell_index:
				if self.num_stocks[index] <= 0:
					action[index] = parameters.ACTION_HOLD
		return action

	def act(self, action, stock_codes, f, recode):
		# action = self.map_action(action)
		buy_index = np.where(action == parameters.ACTION_BUY)[0]
		sell_index = np.where(action == parameters.ACTION_SELL)[0]
		action = self.validate_action(action, buy_index, sell_index)

		# 환경에서 가격 얻기
		date = self.environment.get_date()
		curr_prices = self.environment.curr_price()


		for stock, stock_action in enumerate(action):
			trading_unit = self.num_stocks[stock]
			curr_price = curr_prices[stock]
			# 매수
			if stock_action == parameters.ACTION_BUY:
				# 매수할 단위를 판단
				if recode: self.environment.set_buy_signal(stock)
				trading_unit = (self.balance // self.n_agents) // curr_price
				self.num_stocks[stock] += trading_unit  # 보유 주식 수를 갱신
				self.balance -= curr_price * trading_unit * (1 + parameters.TRADING_CHARGE)  # 보유 현금을 갱신
				self.num_buy += 1  # 매수 횟수 증가
			# 매도
			elif stock_action == parameters.ACTION_SELL:
				if recode: self.environment.set_sell_signal(stock)
				invest_amount = curr_price * trading_unit * (1 - parameters.TRADING_TAX - parameters.TRADING_CHARGE)
				self.num_stocks[stock] -= trading_unit  # 보유 주식 수를 갱신
				self.balance += invest_amount  # 보유 현금을 갱신
				self.num_sell += 1  # 매도 횟수 증가
			# 홀딩
			elif stock_action == parameters.ACTION_HOLD:
				self.num_hold += 1  # 홀딩 횟수 증가

		# 포트폴리오 가치 갱신
		self.prev_portfolio_value = self.portfolio_value
		self.portfolio_value = self.balance + \
							   np.sum(curr_prices * self.num_stocks * (1 - parameters.TRADING_TAX - parameters.TRADING_CHARGE))

		reward = (self.portfolio_value - self.prev_portfolio_value)/self.prev_portfolio_value

		next_prices = self.environment.next_price()
		future_pv = self.balance + \
							   np.sum(next_prices * self.num_stocks * (1 - parameters.TRADING_TAX - parameters.TRADING_CHARGE))
		future_reward = (future_pv - self.portfolio_value)/self.portfolio_value
		if recode:
			for idx, curr_price in enumerate(curr_prices):
				f.write(str(date) +"," + str(stock_codes[idx]).zfill(6) + "," + str(curr_price) +"," + str(action[idx]) +","\
						+ str(self.num_stocks[idx]) +"," + str(self.portfolio_value) + "\n")
					
		self.environment.idx += 1

		####################################################################################################################
		self.reward_history.append(future_reward)
		total_sharpe_reward = 0 if len(self.reward_history) <= 1 else np.mean(self.reward_history) / (np.std(self.reward_history) + 1e-10)
		
		return reward, future_reward, total_sharpe_reward