import numpy as np
from parameters import parameters
from collections import deque
class Trader:

    def __init__(
        self, environment, balance,delayed_reward_threshold=.05):
        # Environment 객체
        # 현재 주식 가격을 가져오기 위해 환경 참조
        self.environment = environment
        # 지연보상 임계치
        self.delayed_reward_threshold = delayed_reward_threshold

        # Trader 클래스의 속성
        self.initial_balance = balance  # 초기 자본금
        self.base_portfolio_value = balance
        self.prev_portfolio_value = balance
        self.balance = balance  # 현재 현금 잔고
        self.prev_balance = balance 
        self.num_stocks = 0  # 보유 주식 수
        self.memory = deque()
        # 포트폴리오 가치: balance + num_stocks * {현재 주식 가격}
        self.portfolio_value = 0
        self.num_buy = 0  # 매수 횟수
        self.num_sell = 0  # 매도 횟수
        self.num_hold = 0  # 홀딩 횟수
        self.immediate_reward = 0  # 즉시 보상
        self.profitloss = 0  # 현재 손익
        self.base_profitloss = 0  # 직전 지연 보상 이후 손익
        self.prev_price = self.environment.curr_price()
        self.pp_price = self.environment.curr_price()
        self.memory_earning_reward = []
        self.memory_minus_earning_reward = []

    def reset(self):
        self.balance = self.initial_balance
        self.prev_portfolio_value = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.immediate_reward = 0
        self.ratio_portfolio_value = 0
        self.memory.clear()
        self.trading_unit = 0
        self.prev_price = self.environment.curr_price()
        self.pp_price = self.environment.curr_price()

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

    def act(self, action, confidence,f,recode):
        if not self.validate_action(action):
            action = parameters.ACTION_HOLD
        # 환경에서 현재 가격 얻기
        date = self.environment.get_date()
        curr_price = self.environment.curr_price()
        # 즉시 보상 초기화
        self.trading_unit = self.num_stocks
        #current_portfolio = self.balance + curr_price * self.num_stocks * (1- parameters.TRADING_TAX)
        #if (current_portfolio - self.prev_protfolio_value) / self.prev_protfolio_value >= self.delayed_reward_threshold and self.num_stocks > 0 : action = parameters.ACTION_SELL
        # 매수
        if action == parameters.ACTION_BUY:
            # 매수할 단위를 판단
            if recode: self.environment.set_buy_signal()
            self.trading_unit = self.balance // curr_price
            self.num_buy += 1  # 매수 횟수 증가
            self.balance -= curr_price * self.trading_unit  # 보유 현금을 갱신
            self.num_stocks += self.trading_unit  # 보유 주식 수를 갱신
        # 매도
        elif action == parameters.ACTION_SELL:
            # 매도
            if recode: self.environment.set_sell_signal()
            invest_amount = curr_price * self.trading_unit * (1- parameters.TRADING_TAX)
            self.num_stocks -= self.trading_unit  # 보유 주식 수를 갱신
            self.balance += invest_amount  # 보유 현금을 갱신
            self.num_sell += 1  # 매도 횟수 증가
        # 홀딩
        elif action == parameters.ACTION_HOLD:
            self.num_hold += 1  # 홀딩 횟수 증가

        # 포트폴리오 가치 갱신
        portfolio_value = self.balance + curr_price * self.num_stocks * (1- parameters.TRADING_TAX)
        
        sharpe_reward = (portfolio_value - self.prev_portfolio_value)/self.prev_portfolio_value
        self.prev_portfolio_value = portfolio_value
        self.pp_price = self.prev_price
        self.prev_price = curr_price

        if recode:
            f.write(str(date) +"," + str(curr_price) +"," + str(action) +","\
                    + str(self.num_stocks) +"," + str(portfolio_value) + "\n")
        self.environment.idx += 1
        return sharpe_reward,portfolio_value
