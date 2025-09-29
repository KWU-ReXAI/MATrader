import os
import argparse
import numpy as np
from network import TD3_network
from feature import Cluster_Data
from parameters import parameters
from realtime_trading import RealTimeTrader

FEATURES = 26

class RealtimeEnvironment:

    def __init__(self, stock_codes, window_size, fmpath, feature_window):
        self.stock_codes = stock_codes
        self.window_size = window_size
        self.fmpath = fmpath
        self.feature_window = feature_window

    def build_state(self, stocks):
        datas = []
        test_datas = []
        for idx, data in enumerate(datas):
            test_feature = Cluster_Data(stocks[idx], data, self.window_size, self.fmpath, self.feature_window,
                                        train=False)
            test_data = test_feature.load_data(0, 0, realtime=True)
            test_datas.append(test_data)
        test_data = np.stack(test_datas, axis=2)
        test_data = test_data.reshape(test_data.shape[0], test_data.shape[1], -1)
        return test_data # done은 일단 뺐음

    def curr_price(self):
        price = 0
        return price

class RealTimeAgent:
    def __init__(self, stock_codes:list, balance =100000000, fmpath = None,
                 load_value_network_path = None, load_policy_network_path = None,
                 window_size = 10):

        # paths and stock code
        self.stock_codes = stock_codes
        self.fmpath = fmpath

        # parameters
        self.window_size = window_size
        self.act_dim = len(stock_codes) * parameters.NUM_ACTIONS
        self.inp_dim = len(stock_codes) * FEATURES
        self.balance = balance
        # Create networks
        self.network = TD3_network(self.inp_dim, self.act_dim, 0, 0, self.window_size)
        if os.path.exists(load_policy_network_path+'.pt'):
            self.network.load_weights(load_policy_network_path,load_value_network_path)


    def realtime_trade(self):
        # environment setting
        environment = RealtimeEnvironment(self.stock_codes, 10, self.fmpath, 1)
        trader = RealTimeTrader(environment, self.balance, self.act_dim)

        # reset
        trader.reset()

        while True:
            state = environment.build_state(self.stock_codes)
            policy = self.network.actor_predict(np.array(state))
            action = policy[0]
            reward, _ = trader.act(action)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stocks', nargs='+')
    parser.add_argument('--model_dir', default=' ')
    parser.add_argument('--model_version', default=29)
    parser.add_argument('--balance', type=int, default=100000000)
    args = parser.parse_args()

    load_value_network_path = os.path.join(parameters.BASE_DIR, 'output', args.model_dir,
        f'phase_4_1', '2025_Q2', 'value_{}'.format(args.model_version))
    load_policy_network_path = os.path.join(parameters.BASE_DIR, 'output', args.model_dir,
        f'phase_4_1', '2025_Q2', 'value_{}'.format(args.model_version))
    feature_model_path = os.path.join(parameters.BASE_DIR, 'output', args.model_dir,
        f'phase_4_1', '2025_Q2', 'feature_model')

    learner = RealTimeAgent(**{'stock_codes': args.stocks, 'balance': args.balance, 'fmpath': feature_model_path,
                'load_policy_network_path' : load_policy_network_path, 'load_value_network_path' : load_value_network_path,
                'window_size': args.window_size})
    learner.realtime_trade()