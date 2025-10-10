import os
import sys
import time
import dotenv
import logging
import argparse
import numpy as np
from network import MultiAgentTransformer
from feature import Cluster_Data
from parameters import parameters
from kis_api import KISApiHandler
from phase import phase2quarter

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
        test_data = np.stack(test_datas, axis=1)
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
        self.act_dim = act_dim  # 액션 수
        self.n_stocks = len(stocks)
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
                return holding['qty']
        return 0

    def act(self, action):
        # action = self.map_action(action)
        if not self.running:
            balance, holdings = self.environment.get_holdings()
            self.initial_balance = balance
            self.balance = np.full(self.n_stocks, balance // self.n_stocks, dtype=np.int64)  # 종목별 잔고: 동등하게 분배
            self.running = True
        else: _, holdings = self.environment.get_holdings()

        # 주식별 매매
        for stock, stock_action in enumerate(action):
            trading_unit = self.map_holdings(holdings, self.stocks[stock])
            # 매수
            if stock_action == parameters.ACTION_BUY:
                curr_price, _ = self.api.get_asking_price(self.stocks[stock])
                trading_unit = int(self.balance[stock] // (curr_price * (1 + self.charge))) if curr_price else 0
                if trading_unit <= 0: action[stock] = parameters.ACTION_HOLD
                else:
                    res = self.api.order_cash(self.stocks[stock], trading_unit, curr_price, '02', order_type='00')
                    if res and res.get('rt_cd') == '0':
                        logging.info(f'{self.stocks[stock]} 매수 주문 접수')
                        self.balance[stock] -= curr_price * trading_unit * (1 + self.charge)	# 보유 현금을 갱신
                    else: action[stock] = parameters.ACTION_HOLD
            # 매도
            elif stock_action == parameters.ACTION_SELL:
                _, curr_price = self.api.get_asking_price(self.stocks[stock])
                if trading_unit <= 0 or not curr_price: action[stock] = parameters.ACTION_HOLD
                else:
                    res = self.api.order_cash(self.stocks[stock], trading_unit, curr_price, '01', order_type='00')
                    if res and res.get('rt_cd') == '0':
                        logging.info(f'{self.stocks[stock]} 매도 주문 접수')
                        self.balance[stock] += curr_price * trading_unit * (1 - self.charge - self.tax)	# 보유 현금을 갱신
                    else: action[stock] = parameters.ACTION_HOLD

        return action

class RealTimeAgent:
    def __init__(self, stock_codes:list, fmpath = None,
                 load_network_path = None, window_size = 10):

        # paths and stock code
        self.stock_codes = stock_codes
        self.fmpath = fmpath

        # parameters
        self.window_size = window_size
        self.act_dim = parameters.NUM_ACTIONS
        self.inp_dim = FEATURES
        # Create networks
        self.network = MultiAgentTransformer(self.inp_dim, self.act_dim, len(stock_codes), 1, 64, 1)
        if os.path.exists(load_network_path+'.pt'):
            self.network.load_model(load_network_path)

    def realtime_trade(self, app_key, app_secret, acnt_no):
        # API setting
        api = KISApiHandler(app_key, app_secret, acnt_no, is_mock=True)
        # environment setting
        environment = RealtimeEnvironment(self.stock_codes, 10, self.fmpath, 1, api)
        trader = RealTimeTrader(environment, self.stock_codes, self.act_dim, api)

        while True:
            logging.info('분봉 매매 시작')
            state = environment.build_state(self.stock_codes)
            policy, _, _ = self.network.act(np.array(state))
            action = policy[0]
            real_actions = trader.act(action)
            logging.info('60초간 대기...')
            time.sleep(60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stock_dir', type=str)
    parser.add_argument('--model_dir', default=' ')
    parser.add_argument('--window_size', default=10)
    parser.add_argument('--model_version', default=29)
    args = parser.parse_args()

    dotenv.load_dotenv()
    output_path = os.path.join(parameters.BASE_DIR, 'output')
    file_handler = logging.FileHandler(filename=os.path.join(
        output_path, f"{'realtime'}.log"), encoding='utf-8')
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setLevel(logging.DEBUG)
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(format="%(message)s",
                        handlers=[file_handler, stream_handler], level=logging.DEBUG)
    quarters_df = phase2quarter(args.stock_dir)
    for row in quarters_df.itertuples():
        load_network_path = os.path.join(parameters.BASE_DIR, 'output', args.model_dir,
            f'phase_{row.phase}_{row.testNum}', row.quarter, 'mat_{}'.format(args.model_version))
        feature_model_path = os.path.join(parameters.BASE_DIR, 'output', args.model_dir,
            f'phase_{row.phase}_{row.testNum}', row.quarter, 'feature_model')

        learner = RealTimeAgent(**{'stock_codes': row.stock_codes, 'fmpath': feature_model_path,
                    'load_network_path' : load_network_path, 'window_size': args.window_size})
        learner.realtime_trade(os.getenv("APP_KEY"), os.getenv("APP_SECRET"), os.getenv("ACNT_NO"))