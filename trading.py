import numpy as np
from parameters import parameters
from collections import deque

class Trader:
	def __init__(
		self, environment, balance, delayed_reward_threshold=.05):
		# 환경
		self.environment = environment

		# 지연보상 임계치
		self.delayed_reward_threshold = delayed_reward_threshold

		# Trader 클래스의 속성
		self.initial_balance = balance  # 초기 자본금
		self.balance = balance  # 현재 현금 잔고
		self.num_stocks = 0  # 보유 주식 수
		# 포트폴리오 가치: balance + num_stocks * {현재 주식 가격} * (1-수수료)
		self.portfolio_value = balance
		self.prev_portfolio_value = balance

		self.num_buy = 0  # 매수 횟수
		self.num_sell = 0  # 매도 횟수
		self.num_hold = 0  # 홀딩 횟수

	def reset(self):
		self.balance = self.initial_balance
		self.num_stocks = 0
		self.portfolio_value = self.initial_balance
		self.prev_portfolio_value = self.initial_balance
		self.num_buy = 0
		self.num_sell = 0
		self.num_hold = 0

	def validate_action(self, action):
		if action == parameters.ACTION_BUY:
			# 적어도 1주를 살 수 있는지 확인
			if self.balance < self.environment.curr_price():
				return False
		elif action == parameters.ACTION_SELL:
			# 주식 잔고가 있는지 확인 
			if self.num_stocks <= 0:
				return False
		return True

	def act(self, action, confidence, f, recode):
		if not self.validate_action(action):
			action = parameters.ACTION_HOLD

		# 환경에서 가격 얻기
		date = self.environment.get_date()
		price = self.environment.curr_price()

		# 즉시 보상 초기화
		trading_unit = self.num_stocks
		#current_portfolio = self.balance + price * self.num_stocks * (1- parameters.TRADING_TAX)
		#if (current_portfolio - self.prev_protfolio_value) / self.prev_protfolio_value >= self.delayed_reward_threshold and self.num_stocks > 0 : action = parameters.ACTION_SELL
		# 매수
		if action == parameters.ACTION_BUY:
			# 매수할 단위를 판단
			if recode: self.environment.set_buy_signal()
			trading_unit = self.balance // (price * (1 + parameters.TRADING_TAX))
			self.num_stocks += trading_unit  # 보유 주식 수를 갱신
			self.balance -= (price * (1 + parameters.TRADING_TAX)) * trading_unit  # 보유 현금을 갱신
			self.num_buy += 1  # 매수 횟수 증가
		# 매도
		elif action == parameters.ACTION_SELL:
			# 매도
			if recode: self.environment.set_sell_signal()
			invest_amount = price * trading_unit * (1 - parameters.TRADING_TAX)
			self.num_stocks -= trading_unit  # 보유 주식 수를 갱신
			self.balance += invest_amount  # 보유 현금을 갱신
			self.num_sell += 1  # 매도 횟수 증가
		# 홀딩
		elif action == parameters.ACTION_HOLD:
			self.num_hold += 1  # 홀딩 횟수 증가

		# 포트폴리오 가치 갱신
		self.prev_portfolio_value = self.portfolio_value
		self.portfolio_value = self.balance + price * self.num_stocks * (1- parameters.TRADING_TAX)
		
		sharpe_reward = (self.portfolio_value - self.prev_portfolio_value)/self.prev_portfolio_value

		if recode:
			f.write(str(date) +"," + str(price) +"," + str(action) +","\
					+ str(self.num_stocks) +"," + str(self.portfolio_value) + "\n")
					
		self.environment.idx += 1
		
		return sharpe_reward, self.portfolio_value

'''
	def imitative_reward(self,action):
		curr_portfolio_value = self.balance + self.environment.curr_price() * self.trading_unit * (1- parameters.TRADING_TAX)
		prev_portfolio_value = self.balance + self.prev_price * self.trading_unit * (1- parameters.TRADING_TAX)
			# 즉시 보상 - 수익률
		if action == parameters.ACTION_SELL:
			immediate_reward = (prev_portfolio_value - curr_portfolio_value) / curr_portfolio_value
		elif action == parameters.ACTION_HOLD: immediate_reward = 0
		else: immediate_reward = (curr_portfolio_value - prev_portfolio_value) / prev_portfolio_value  
		return immediate_reward

	def get_reward(self):
		state,action, trading_unit, prev_balance = self.memory.popleft()
		
		curr_price = self.environment.curr_price()
		curr_portfolio_value = prev_balance + curr_price * trading_unit * (1- parameters.TRADING_TAX)
		prev_portfolio_value = prev_balance + self.prev_price * trading_unit * (1- parameters.TRADING_TAX)
			# 즉시 보상 - 수익률
		if action == parameters.ACTION_SELL:
			immediate_reward = (prev_portfolio_value - curr_portfolio_value) / curr_portfolio_value
		elif action == parameters.ACTION_HOLD: immediate_reward = 0
		else: immediate_reward = (curr_portfolio_value - prev_portfolio_value) / prev_portfolio_value
		return state, action, immediate_reward, self.prev_price

	def memory_state(self, state, action):
		self.memory.append((state,action,self.trading_unit,self.balance))

	def action_critic_get_reward(self):
		state,critic_state,policy,action, trading_unit, prev_balance = self.memory.popleft()
		curr_price = self.environment.curr_price()
		curr_portfolio_value = prev_balance + curr_price * trading_unit
		
		prev_portfolio_value = prev_balance + self.prev_price * trading_unit
			# 즉시 보상 - 수익률
		# if action == parameters.ACTION_SELL:
		#     immediate_reward = (prev_portfolio_value - curr_portfolio_value) / curr_portfolio_value
		# elif action == parameters.ACTION_HOLD: immediate_reward = 0
		# else: immediate_reward = (curr_portfolio_value - prev_portfolio_value) / prev_portfolio_value
		immediate_reward = (curr_portfolio_value - prev_portfolio_value) / prev_portfolio_value
		return state,critic_state,policy, immediate_reward,action, self.prev_price

	def action_critic_memory_state(self, state, critic_state,policy, action):
		#print(sortino)
		self.memory.append((state,critic_state,policy,action,self.trading_unit,self.balance))
'''