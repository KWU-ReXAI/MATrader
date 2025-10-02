import os
import sys
import logging
import argparse
import json
from parameters import parameters
import data_manager
from learner import TD3_Agent
from phase import phase2quarter
from kis_api import KISApiHandler
import numpy as np
from dotenv import load_dotenv

load_dotenv()

APP_KEY = os.getenv("APP_KEY")
APP_SECRET = os.getenv("APP_SECRET")
ACNT_NO = os.getenv("ACNT_NO")
IS_MOCK = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ... (argparse 인자 설정은 기존과 동일)
    parser.add_argument('--stock_dir', type=str)
    parser.add_argument('--output_name', type=str, default='result')
    parser.add_argument('--ver', choices=['ROK', 'USA', 'ETF'], default='ROK')
    parser.add_argument('--algorithm', choices=['td3', 'dsl', 'gdpg', 'gdqn', 'candle', 'attention', 'irdpg'],
                        default='td3')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--realtime', action='store_true')
    parser.add_argument('--model_dir', default=' ')
    parser.add_argument('--model_version', default=29)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--balance', type=int, default=100000000)
    parser.add_argument('--max_episode', type=int, default=30)
    parser.add_argument('--delayed_reward_threshold', type=float, default=0.03)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--window_size', type=int, default=10)
    parser.add_argument('--feature_window', type=int, default=1)
    parser.add_argument('--num_step', type=int, default=2)
    parser.add_argument('--start_epsilon', type=float, default=0.02)
    parser.add_argument('--noise', type=float, default=0.7)
    parser.add_argument('--debug', type=bool, default=False)
    args = parser.parse_args()

    # ... (output_path, logging 설정은 기존과 동일)
    if args.debug:
        output_path = os.path.join(parameters.BASE_DIR,
                                   'output/debug'.format(args.output_name))
    else:
        output_path = os.path.join(parameters.BASE_DIR,
                                   'output/{}'.format(args.output_name))
    if not os.path.isdir(output_path): os.makedirs(output_path)
    with open(os.path.join(output_path, 'params.json'), 'w') as f:
        f.write(json.dumps(vars(args)))
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    file_handler = logging.FileHandler(filename=os.path.join(
        output_path, "{}.log".format(args.output_name)), encoding='utf-8')
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setLevel(logging.DEBUG)
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(format="%(message)s",
                        handlers=[file_handler, stream_handler], level=logging.DEBUG)

    if args.realtime:
        logging.info("실시간 거래 모드를 시작합니다.")

        quarters_df = phase2quarter(args.stock_dir)
        last_phase_info = quarters_df.iloc[-1]
        stock_codes_to_trade = last_phase_info['stock_codes']

        num_features_per_stock = 26  # candle(4) + overlay(8) + volume(3) + volatility(3) + momentum(8)
        correct_inp_dim = len(stock_codes_to_trade) * num_features_per_stock
        dummy_test_data = np.zeros((1, args.window_size, correct_inp_dim))

        model_base_path = os.path.join(parameters.BASE_DIR, 'output', args.model_dir,
                                       f'phase_{last_phase_info.phase}_{last_phase_info.testNum}',
                                       last_phase_info.quarter)

        load_value_network_path = os.path.join(model_base_path, f'value_{args.model_version}')
        load_policy_network_path = os.path.join(model_base_path, f'policy_{args.model_version}')
        feature_model_path = os.path.join(model_base_path, 'feature_model')

        # API 핸들러 생성
        api_handler = KISApiHandler(APP_KEY, APP_SECRET, ACNT_NO, IS_MOCK)
        api_handler.get_access_token()

        # [수정] fmpath를 Learner 생성 시점에 전달
        learner = TD3_Agent(
            stock_codes=stock_codes_to_trade,
            num_of_stock=len(stock_codes_to_trade),
            test_data=dummy_test_data,
            output_path=model_base_path,
            policy_network_path=load_policy_network_path,
            load_policy_network_path=load_policy_network_path,
            load_value_network_path=load_value_network_path,
            window_size=args.window_size,
            api_handler=api_handler,
            phase=last_phase_info.phase, testNum=last_phase_info.testNum, quarter=last_phase_info.quarter,
            train_chart_data=None, test_chart_data=None, training_data=None,
            balance=args.balance, lr=args.lr, test=True
        )

        learner.trade_realtime(fmpath = feature_model_path)

    # 훈련 또는 백테스트 로직
    else:
        quarters_df = phase2quarter(args.stock_dir)
        for row in quarters_df.itertuples():
            quarter_path = os.path.join(output_path, f'phase_{row.phase}_{row.testNum}', row.quarter)
            if not os.path.isdir(quarter_path): os.makedirs(quarter_path)
            if not row.stock_codes:
                continue

            policy_network_path = os.path.join(quarter_path, 'policy')
            value_network_path = os.path.join(quarter_path, 'value')

            if args.test:
                load_value_network_path = os.path.join(parameters.BASE_DIR, 'output', args.model_dir,
                                                       f'phase_{row.phase}_{row.testNum}', row.quarter,
                                                       'value_{}'.format(args.model_version))
                load_policy_network_path = os.path.join(parameters.BASE_DIR, 'output', args.model_dir,
                                                        f'phase_{row.phase}_{row.testNum}', row.quarter,
                                                        'policy_{}'.format(args.model_version))
                feature_model_path = os.path.join(parameters.BASE_DIR, 'output', args.model_dir,
                                                  f'phase_{row.phase}_{row.testNum}', row.quarter, 'feature_model')
            else:
                load_value_network_path = " "
                load_policy_network_path = " "
                feature_model_path = os.path.join(quarter_path, 'feature_model')
                if not os.path.isdir(feature_model_path): os.makedirs(feature_model_path)

            train_chart_data, test_chart_data, training_data, test_data = data_manager.load_data(
                os.path.join(parameters.BASE_DIR, f'data/{args.ver}/'),
                row.stock_codes, feature_model_path,
                row.train_start, row.train_end, row.test_start, row.test_end,
                window_size=args.window_size,
                feature_window=args.feature_window,
                train=not args.test)

            common_params = {
                'stock_codes': row.stock_codes,
                'num_of_stock': len(row.stock_codes),
                'train_chart_data': train_chart_data, 'test_chart_data': test_chart_data,
                'training_data': training_data, 'test_data': test_data,
                'delayed_reward_threshold': args.delayed_reward_threshold,
                'balance': args.balance,
                'api_handler': None
            }

            learner = TD3_Agent(
                **{**common_params, 'output_path': quarter_path, 'lr': args.lr, 'test': args.test, 'phase': row.phase,
                   'testNum': row.testNum, 'quarter': row.quarter, 'value_network_path': value_network_path,
                   'policy_network_path': policy_network_path,
                   'load_policy_network_path': load_policy_network_path,
                   'load_value_network_path': load_value_network_path,
                   'window_size': args.window_size})

            if not args.test:
                learner.run(args.max_episode, args.num_step, args.noise, args.start_epsilon)
            learner.backtest(args.num_step)