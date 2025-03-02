from drawing import plt_result
import numpy as np
from parameters import parameters
from collections import deque

class Environment:
    PRICE_IDX = 4  # 종가의 위치

    def __init__(self, chart_data=None, training_data=None):
        self.chart_data = chart_data
        self.plt_data = chart_data.copy()
        self.plt_data['buy_signal'] = 0.0; self.plt_data['sell_signal'] = 0.0
        self.plt_data = self.plt_data.set_index('date')
        self.training_data = training_data
        self.observation = None
        self.idx = 0
        self.training_data_idx = 0

    def reset(self):
        self.observation = None
        self.idx = 0
        self.training_data_idx = 0

    def observe(self):
        if len(self.chart_data) > self.idx + 1:
            self.observation = self.chart_data.iloc[self.idx]
            self.idx += 1
            return self.observation
        return None
    
    def build_state(self):
        if len(self.training_data) > self.training_data_idx + 1:
            self.sample = self.training_data[self.training_data_idx]
            self.training_data_idx += 1
            return self.sample.tolist(),False
        return None,True

    def get_price(self):
        self.observe()
        if self.observation is not None:
            return self.observation[self.PRICE_IDX]
        return None
    
    def prev_price(self):
        stock_price = self.chart_data.iloc[self.idx-1]
        return stock_price[self.PRICE_IDX]

    def curr_price(self):
        if len(self.chart_data) >= self.idx + 1:
            stock_price = self.chart_data.iloc[self.idx]
            return stock_price[self.PRICE_IDX]
        return None
    
    def next_price(self):
        if len(self.chart_data) >= self.idx + 2:
            stock_price = self.chart_data.iloc[self.idx+1]
            return stock_price[self.PRICE_IDX]
        return None

    def get_date(self):
        curr_data = self.chart_data.iloc[self.idx]
        return curr_data[0]

    def set_chart_data(self, chart_data):
        self.chart_data = chart_data

    def set_buy_signal(self):
        index = self.plt_data.index[self.idx-1]
        price = self.plt_data.iloc[self.idx-1][self.PRICE_IDX-1]
        self.plt_data.at[index,'buy_signal'] = price
    def set_sell_signal(self):
        index = self.plt_data.index[self.idx-1]
        price = self.plt_data.iloc[self.idx-1][self.PRICE_IDX-1]
        self.plt_data.at[index,'sell_signal'] = price