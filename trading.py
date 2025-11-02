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
		self.balance = balance
		self.num_stocks = np.zeros(n_agents, dtype=np.int64)  # 종목별 보유 주식 수
		# 포트폴리오 가치: balance + num_stocks * {현재 주식 가격} * (1-수수료)
		self.portfolio_value = balance
		self.prev_portfolio_value = balance

		self.num_buy = 0  # 매수 횟수
		self.num_sell = 0  # 매도 횟수
		self.num_hold = 0  # 홀딩 횟수

	def reset(self):
		self.balance = self.initial_balance
		self.num_stocks = np.zeros(self.n_agents, dtype=np.int64)
		self.portfolio_value = self.initial_balance
		self.prev_portfolio_value = self.initial_balance
		self.num_buy = 0
		self.num_sell = 0
		self.num_hold = 0

	def act(self, action, stock_codes, f, recode):
		# 매수를 원하는 주식들
		buy_count = np.sum(action == parameters.ACTION_BUY)
		cash = self.balance // buy_count if buy_count > 0 else 0

		# 환경에서 가격 얻기
		date = self.environment.get_date()
		curr_prices = self.environment.curr_price()

		for stock, stock_action in enumerate(action):
			curr_price = curr_prices[stock]
			trading_unit = cash // (curr_price * (1 + parameters.TRADING_CHARGE))
			# 매수
			if stock_action == parameters.ACTION_BUY and trading_unit > 0:
				# 매수할 단위를 판단
				if recode: self.environment.set_buy_signal(stock)
				self.num_stocks[stock] += trading_unit  # 보유 주식 수를 갱신
				self.balance -= curr_price * trading_unit * (1 + parameters.TRADING_CHARGE)  # 보유 현금을 갱신
				self.num_buy += 1  # 매수 횟수 증가
			# 매도
			elif stock_action == parameters.ACTION_SELL and self.num_stocks[stock] > 0:
				if recode: self.environment.set_sell_signal(stock)
				invest_amount = curr_price * self.num_stocks[stock] * (1 - parameters.TRADING_TAX - parameters.TRADING_CHARGE)
				self.num_stocks[stock] = 0  # 보유 주식 수를 갱신
				self.balance += invest_amount  # 보유 현금을 갱신
				self.num_sell += 1  # 매도 횟수 증가
			# 홀딩
			else:
				self.num_hold += 1  # 홀딩 횟수 증가

		# 포트폴리오 가치 갱신
		self.prev_portfolio_value = self.portfolio_value
		self.portfolio_value = self.balance + \
							   np.sum(curr_prices * self.num_stocks * (1 - parameters.TRADING_TAX - parameters.TRADING_CHARGE))
		
		reward = (self.portfolio_value - self.prev_portfolio_value)/self.prev_portfolio_value
		next_prices = self.environment.next_price()
		future_pv = self.balance + \
							   np.sum(next_prices * self.num_stocks * (1 - parameters.TRADING_TAX - parameters.TRADING_CHARGE))
		future_log_reward = np.log(future_pv) - np.log(self.portfolio_value)
		if recode:
			for idx, curr_price in enumerate(curr_prices):
				f.write(str(date) +"," + str(stock_codes[idx]).zfill(6) + "," + str(curr_price) +"," + str(action[idx]) +","\
						+ str(self.num_stocks[idx]) +"," + str(self.portfolio_value) + "\n")
					
		self.environment.idx += 1
		
		return reward, future_log_reward