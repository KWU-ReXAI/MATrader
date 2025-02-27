import os
import sys
import logging
import argparse
import json
from parameters import parameters
import test_data_manager
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stock_code', default=' ')
    parser.add_argument('--ver', choices=['KOREA','USA','ETF'], default='USA')
    parser.add_argument('--algorithm', choices=['td3','dsl','gdpg','gdqn','candle','attention','irdpg'], default='td3')
    parser.add_argument('--model_version')
    parser.add_argument('--start_date', default='20170101')
    parser.add_argument('--end_date', default='20171231')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--balance', type=int, default=10000)
    parser.add_argument('--delayed_reward_threshold', type=float, default=0.05)
    parser.add_argument('--output_name', default=' ')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--window_size', type=int, default=1)
    parser.add_argument('--feature_window', type=int, default=1)
    parser.add_argument('--num_step', type=int, default=1)
    parser.add_argument('--start_epsilon', type=float, default=0.02)
    parser.add_argument('--noise', type=float, default=0.00001)
    parser.add_argument('--max_episode', type=int, default=1)
    args = parser.parse_args()

    # 출력 경로 설정
    output_path = os.path.join(parameters.BASE_DIR, 
        'output/{}_{}_{}_{}_{}'.format(args.output_name,args.window_size,args.num_step,args.start_epsilon,args.noise))
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    feature_model_path = os.path.join(output_path,'feature_model')
    if not os.path.isdir(feature_model_path): os.makedirs(feature_model_path)
    # 파라미터 기록
    with open(os.path.join(output_path, 'params.json'), 'w') as f:
        f.write(json.dumps(vars(args)))
    
    # 로그 기록 설정
    file_handler = logging.FileHandler(filename=os.path.join(
        output_path, "{}.log".format(args.output_name)), encoding='utf-8')
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setLevel(logging.DEBUG)
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(format="%(message)s",
        handlers=[file_handler, stream_handler], level=logging.DEBUG)
    

    load_value_network_path = os.path.join(
        output_path, 'value_{}_{}'.format(args.output_name, args.model_version))
    load_policy_network_path = os.path.join(
        output_path, 'policy_{}_{}'.format(args.output_name,args.model_version))
   
    common_params = {}

    from test_learner import TD3_Agent
    # from test_learner import CANDLE_Agent, GDPG_Agent, TD3_Agent, GDQN_Agent, ATTENTION_Agent,Imitative_Agent
    # 차트 데이터, 학습 데이터 준비
    chart_data, training_data = test_data_manager.test_load_data(
        os.path.join(parameters.BASE_DIR, 
        'data/{}/{}.csv'.format(args.ver, args.stock_code)), feature_model_path, 
        args.start_date, args.end_date, window_size=args.window_size, algorithm=args.algorithm)
    # 공통 파라미터 설정
    common_params = {'delayed_reward_threshold': args.delayed_reward_threshold,
                'balance' : args.balance}
    # 강화학습 시작
    common_params.update({'stock_code': args.stock_code,
            'chart_data': chart_data, 'training_data': training_data})

    if args.algorithm == 'td3':
        learner = TD3_Agent(**{**common_params, 'output_path': output_path, 'lr': args.lr, 
                    'load_policy_network_path' : load_policy_network_path, 'load_value_network_path' : load_value_network_path,
                    'window_size': args.window_size})
        learner.run(args.max_episode, args.num_step)
    # if args.algorithm == 'candle':
    #     learner = CANDLE_Agent(**{**common_params, 'output_path': output_path, 'lr': args.lr,
    #                 'load_network_path' : load_policy_network_path, 'window_size': args.window_size})
    #     learner.run(args.max_episode, args.num_step)
    # elif args.algorithm == 'gdpg':
    #     learner = GDPG_Agent(**{**common_params, 'output_path': output_path, 'lr': args.lr,
    #                 'load_policy_network_path' : load_policy_network_path, 'load_value_network_path' : load_value_network_path,
    #                 'window_size': args.window_size})
    #     learner.run(args.max_episode, args.num_step)
    # elif args.algorithm == 'gdqn':
    #     learner = GDQN_Agent(**{**common_params, 'output_path': output_path, 'lr': args.lr, 
    #                 'load_network_path' : load_policy_network_path, 'window_size': args.window_size})
    #     learner.run(args.max_episode, args.num_step)
    # elif args.algorithm == 'td3':
    #     learner = TD3_Agent(**{**common_params, 'output_path': output_path, 'lr': args.lr, 
    #                 'load_policy_network_path' : load_policy_network_path, 'load_value_network_path' : load_value_network_path,
    #                 'window_size': args.window_size})
    #     learner.run(args.max_episode, args.num_step)
    # elif args.algorithm == 'attention':
    #     learner = ATTENTION_Agent(**{**common_params, 'output_path': output_path, 'lr': args.lr,
    #                 'load_network_path' : load_policy_network_path, 'window_size': args.window_size})
    #     learner.run(args.max_episode, args.num_step)
    # elif args.algorithm == 'irdpg':
    #     learner = Imitative_Agent(**{**common_params, 'output_path': output_path, 'lr': args.lr,
    #                 'load_policy_network_path' : load_policy_network_path, 'load_value_network_path' : load_value_network_path,
    #                 'window_size': args.window_size})
    #     learner.run(args.max_episode, args.num_step, args.noise,args.start_epsilon)
