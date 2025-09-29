import numpy as np
from parameters import parameters
from collections import deque

class Trader:
	def __init__(self, environment, balance, act_dim, delayed_reward_threshold=.05):
		# 환경
		self.environment = environment

		# 지연보상 임계치
		self.delayed_reward_threshold = delayed_reward_threshold

		# Trader 클래스의 속성
		self.initial_balance = balance  # 초기 자본금
		self.act_dim = act_dim # 종목 수 * 액션 수
		self.n_stocks = act_dim // parameters.NUM_ACTIONS # 종목 수
		# 포트폴리오 관련
		self.balance = np.full(self.n_stocks, balance // self.n_stocks, dtype=np.int64)  # 종목별 잔고: 동등하게 분배
		self.cash = balance % self.n_stocks # 종목 별로 잔고 동등하게 나누고 남은 현금
		self.num_stocks = np.zeros(self.n_stocks, dtype=np.int64)  # 종목별 보유 주식 수
		# 포트폴리오 가치: balance + num_stocks * {현재 주식 가격} * (1-수수료)
		self.portfolio_value = balance
		self.prev_portfolio_value = balance

		self.num_buy = 0  # 매수 횟수
		self.num_sell = 0  # 매도 횟수
		self.num_hold = 0  # 홀딩 횟수

	def reset(self):
		self.balance = np.full(self.n_stocks, self.initial_balance // self.n_stocks, dtype=np.int64)
		self.cash = self.initial_balance % self.n_stocks
		self.num_stocks = np.zeros(self.n_stocks, dtype=np.int64)
		self.portfolio_value = self.initial_balance
		self.prev_portfolio_value = self.initial_balance
		self.num_buy = 0
		self.num_sell = 0
		self.num_hold = 0

	def map_action(self, action):
		scores_per_stock = action.reshape(self.n_stocks, parameters.NUM_ACTIONS)
		discrete_actions = np.argmax(scores_per_stock, axis=1)
		return discrete_actions

	def validate_action(self, action, buy_index, sell_index):
		if len(buy_index) > 0:
			# 사는 종목들이 각각 적어도 1주를 살 수 있는지 확인
			for index in buy_index:
				if self.balance[index] < self.environment.curr_price()[index]:
					action[index] = parameters.ACTION_HOLD
		if len(sell_index) > 0:
			# 주식 잔고가 있는지 확인
			for index in sell_index:
				if self.num_stocks[index] <= 0:
					action[index] = parameters.ACTION_HOLD
		return action

	def act(self, action, stock_codes, f, recode):
		action = self.map_action(action)
		buy_index = np.where(action == parameters.ACTION_BUY)[0]
		sell_index = np.where(action == parameters.ACTION_SELL)[0]
		action = self.validate_action(action, buy_index, sell_index)

		# 환경에서 가격 얻기
		date = self.environment.get_date()
		curr_prices = self.environment.curr_price()

		# 즉시 보상 초기화
		#current_portfolio = self.balance + curr_price * self.num_stocks * (1- parameters.TRADING_TAX)
		#if (current_portfolio - self.prev_protfolio_value) / self.prev_protfolio_value >= self.delayed_reward_threshold and self.num_stocks > 0 : action = parameters.ACTION_SELL
		for stock, stock_action in enumerate(action):
			trading_unit = self.num_stocks[stock]
			curr_price = curr_prices[stock]
			# 매수
			if stock_action == parameters.ACTION_BUY:
				# 매수할 단위를 판단
				if recode: self.environment.set_buy_signal(stock)
				trading_unit = self.balance[stock] // curr_price
				self.num_stocks[stock] += trading_unit  # 보유 주식 수를 갱신
				self.balance[stock] -= curr_price * trading_unit * (1 + parameters.TRADING_CHARGE)  # 보유 현금을 갱신
				self.num_buy += 1  # 매수 횟수 증가
			# 매도
			elif stock_action == parameters.ACTION_SELL:
				if recode: self.environment.set_sell_signal(stock)
				invest_amount = curr_price * trading_unit * (1 - parameters.TRADING_TAX - parameters.TRADING_CHARGE)
				self.num_stocks[stock] -= trading_unit  # 보유 주식 수를 갱신
				self.balance[stock] += invest_amount  # 보유 현금을 갱신
				self.num_sell += 1  # 매도 횟수 증가
			# 홀딩
			elif stock_action == parameters.ACTION_HOLD:
				self.num_hold += 1  # 홀딩 횟수 증가

		# 포트폴리오 가치 갱신
		self.prev_portfolio_value = self.portfolio_value
		self.portfolio_value = np.sum(self.balance) + \
							   np.sum(curr_prices * self.num_stocks * (1 - parameters.TRADING_TAX - parameters.TRADING_CHARGE)) + self.cash
		
		reward = (self.portfolio_value - self.prev_portfolio_value)/self.prev_portfolio_value
		next_prices = self.environment.next_price()
		future_pv = np.sum(self.balance) + \
							   np.sum(next_prices * self.num_stocks * (1 - parameters.TRADING_TAX - parameters.TRADING_CHARGE)) + self.cash
		future_reward = (future_pv - self.portfolio_value)/self.portfolio_value
		if recode:
			for idx, curr_price in enumerate(curr_prices):
				f.write(str(date) +"," + str(stock_codes[idx]).zfill(6) + "," + str(curr_price) +"," + str(action[idx]) +","\
						+ str(self.num_stocks[idx]) +"," + str(self.portfolio_value) + "\n")
					
		self.environment.idx += 1
		
		return reward, future_reward