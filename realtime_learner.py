import os
import time
import dotenv
import argparse
import numpy as np
from network import TD3_network
from feature import Cluster_Data
from parameters import parameters
from kis_api import KISApiHandler

FEATURES = 26

class RealtimeEnvironment:

    def __init__(self, stock_codes, window_size, fmpath, feature_window, api):
        self.stock_codes = stock_codes
        self.window_size = window_size
        self.fmpath = fmpath
        self.feature_window = feature_window
        self.api = api

    def build_state(self, stocks):
        datas = []
        for stock in stocks:
            df = self.api.get_minute_candles_all_today(stock, max_pages=3)
            df.drop(columns=['time', 'acml_tr_pbmn', 'datetime'], inplace=True)
            df['adj close'] = df['close']
            columns = ['date', 'open', 'high', 'low', 'close', 'adj close', 'volume']
            df = df[columns]
            dtype_map = {col: 'int64' for col in df.columns if col != 'date'}
            df = df.astype(dtype_map)
            datas.append(df)
        test_datas = []
        for idx, data in enumerate(datas):
            test_feature = Cluster_Data(stocks[idx], data, self.window_size, self.fmpath, self.feature_window,
                                        train=False)
            test_data = test_feature.load_data(0, 0, realtime=True)
            test_datas.append(test_data)
        test_data = np.stack(test_datas, axis=2)
        test_data = test_data.reshape(test_data.shape[0], test_data.shape[1], -1)
        return test_data # done은 일단 뺐음

    def get_holdings(self):
        balance = self.api.get_account_balance()
        if balance is None:
            raise Exception('No balance available')
        return balance['deposit'], balance['holdings']

class RealTimeTrader:
    def __init__(self, environment, stocks, act_dim, api):
        # 환경
        self.environment = environment
        self.stocks = stocks
        self.api = api
        # Trader 클래스의 속성
        self.initial_balance = None
        self.act_dim = act_dim  # 종목 수 * 액션 수
        self.n_stocks = act_dim // parameters.NUM_ACTIONS  # 종목 수
        # 포트폴리오 관련
        self.balance = None
        self.running = False
        self.charge = 0.000142
        self.tax = 0.0015

    def map_action(self, action):
        scores_per_stock = action.reshape(self.n_stocks, parameters.NUM_ACTIONS)
        discrete_actions = np.argmax(scores_per_stock, axis=1)
        return discrete_actions

    def map_holdings(self, holdings, stock):
        for holding in holdings:
            if holding['iscd'] == stock:
                return holding['qty'], holding['cur_price']
        return 0, self.api.get_current_price(stock)

    def act(self, action):
        action = self.map_action(action)
        if not self.running:
            balance, holdings = self.environment.get_holdings()
            self.initial_balance = balance
            self.balance = np.full(self.n_stocks, balance // self.n_stocks, dtype=np.int64)  # 종목별 잔고: 동등하게 분배
            self.running = True
        else: _, holdings = self.environment.get_holdings()

        # 주식별 매매
        for stock, stock_action in enumerate(action):
            trading_unit, curr_price = self.map_holdings(holdings, self.stocks[stock])
            # 매수
            if stock_action == parameters.ACTION_BUY:
                trading_unit = int(self.balance[stock] // (curr_price * (1 + self.charge)))
                if trading_unit <= 0: action[stock] = parameters.ACTION_HOLD
                else:
                    res = self.api.order_cash(self.stocks[stock], trading_unit, curr_price, '02', order_type='00')
                    if res and res.get('rt_cd') == '0':
                        print('매수 주문 완료')
                        self.balance[stock] -= curr_price * trading_unit * (1 + self.charge)	# 보유 현금을 갱신
                    else: action[stock] = parameters.ACTION_HOLD
            # 매도
            elif stock_action == parameters.ACTION_SELL:
                if trading_unit <= 0: action[stock] = parameters.ACTION_HOLD
                else:
                    res = self.api.order_cash(self.stocks[stock], trading_unit, curr_price, '01', order_type='00')
                    if res and res.get('rt_cd') == '0':
                        print('매도 주문 완료')
                        self.balance[stock] += curr_price * trading_unit * (1 - self.charge - self.tax)	# 보유 현금을 갱신
                    else: action[stock] = parameters.ACTION_HOLD

        return action

class RealTimeAgent:
    def __init__(self, stock_codes:list, fmpath = None,
                 load_value_network_path = None, load_policy_network_path = None,
                 window_size = 10):

        # paths and stock code
        self.stock_codes = stock_codes
        self.fmpath = fmpath

        # parameters
        self.window_size = window_size
        self.act_dim = len(stock_codes) * parameters.NUM_ACTIONS
        self.inp_dim = len(stock_codes) * FEATURES
        # Create networks
        self.network = TD3_network(self.inp_dim, self.act_dim, 0, 0, self.window_size)
        if os.path.exists(load_policy_network_path+'.pt'):
            self.network.load_weights(load_policy_network_path,load_value_network_path)

    def realtime_trade(self, app_key, app_secret, acnt_no):
        # API setting
        api = KISApiHandler(app_key, app_secret, acnt_no, is_mock=True)
        # environment setting
        environment = RealtimeEnvironment(self.stock_codes, 10, self.fmpath, 1, api)
        trader = RealTimeTrader(environment, self.stock_codes, self.act_dim, api)

        while True:
            state = environment.build_state(self.stock_codes)
            policy = self.network.actor_predict(np.array(state))
            action = policy[0]
            real_actions = trader.act(action)
            for idx, real_action in enumerate(real_actions):
                print(f'Stock: {self.stock_codes[idx]}, Action: {real_action}')
            time.sleep(60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stocks', nargs='+')
    parser.add_argument('--model_dir', default=' ')
    parser.add_argument('--window_size', default=10)
    parser.add_argument('--model_version', default=29)
    args = parser.parse_args()

    dotenv.load_dotenv()
    load_value_network_path = os.path.join(parameters.BASE_DIR, 'output', args.model_dir,
        f'phase_1_1', '2022_Q1', 'value_{}'.format(args.model_version))
    load_policy_network_path = os.path.join(parameters.BASE_DIR, 'output', args.model_dir,
        f'phase_1_1', '2022_Q1', 'policy_{}'.format(args.model_version))
    feature_model_path = os.path.join(parameters.BASE_DIR, 'output', args.model_dir,
        f'phase_1_1', '2022_Q1', 'feature_model')

    learner = RealTimeAgent(**{'stock_codes': args.stocks, 'fmpath': feature_model_path,
                'load_policy_network_path' : load_policy_network_path, 'load_value_network_path' : load_value_network_path,
                'window_size': args.window_size})
    learner.realtime_trade(os.getenv("APP_KEY"), os.getenv("APP_SECRET"), os.getenv("ACNT_NO"))