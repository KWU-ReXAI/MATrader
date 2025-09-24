import os
import sys
import logging
import argparse
import json
from parameters import parameters
import data_manager
from learner import TD3_Agent
from phase import phase2quarter
from kis_api import KISApiHandler  # 필요한 시점에 임포트
import numpy as np
from dotenv import load_dotenv  # 1. 라이브러리 임포트

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv() # 2. 이 함수가 .env 파일을 읽어 환경 변수로 설정합니다.

APP_KEY = os.getenv("APP_KEY")
APP_SECRET = os.getenv("APP_SECRET")
ACNT_NO = os.getenv("ACNT_NO")  # 모의투자 계좌
IS_MOCK = True  # True: 모의투자, False: 실전투자

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--stock_dir', type=str)
	#parser.add_argument('--output_name', default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
	parser.add_argument('--output_name', type=str,default='result')
	parser.add_argument('--ver', choices=['ROK','USA','ETF'], default='ROK')
	parser.add_argument('--algorithm', choices=['td3','dsl','gdpg','gdqn','candle', 'attention','irdpg'], default='td3')
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
	parser.add_argument('--debug',type=bool,default=False)
	args = parser.parse_args()

	if args.debug:
		output_path = os.path.join(parameters.BASE_DIR,
								   'output/debug'.format(args.output_name))
	else:
		output_path = os.path.join(parameters.BASE_DIR,
								   'output/{}'.format(args.output_name))
	if not os.path.isdir(output_path): os.makedirs(output_path)

	# 파라미터 기록
	with open(os.path.join(output_path, 'params.json'), 'w') as f:
		f.write(json.dumps(vars(args)))

	# 로그 기록 설정
	logging.getLogger('matplotlib').setLevel(logging.WARNING)
	file_handler = logging.FileHandler(filename=os.path.join(
		output_path, "{}.log".format(args.output_name)), encoding='utf-8')
	stream_handler = logging.StreamHandler(sys.stdout)
	file_handler.setLevel(logging.DEBUG)
	stream_handler.setLevel(logging.INFO)
	logging.basicConfig(format="%(message)s",
		handlers=[file_handler, stream_handler], level=logging.DEBUG)

	quarters_df = phase2quarter(args.stock_dir)

	for row in quarters_df.itertuples():
		quarter_path = os.path.join(output_path, f'phase_{row.phase}_{row.testNum}', row.quarter)
		if not os.path.isdir(quarter_path): os.makedirs(quarter_path)
		### 뽑힌 주식 없을 때 예외처리: 폴더만 생성 ###
		if not row.stock_codes:
			continue

		# 모델 경로 준비
		policy_network_path = os.path.join(
			quarter_path, 'policy')
		value_network_path = os.path.join(
			quarter_path, 'value')

		# 모델 재사용
		if args.test:
			load_value_network_path = os.path.join(parameters.BASE_DIR, 'output', args.model_dir,
				f'phase_{row.phase}_{row.testNum}', row.quarter, 'value_{}'.format(args.model_version))
			load_policy_network_path = os.path.join(parameters.BASE_DIR, 'output', args.model_dir,
				f'phase_{row.phase}_{row.testNum}', row.quarter, 'policy_{}'.format(args.model_version))
			feature_model_path = os.path.join(parameters.BASE_DIR, 'output', args.model_dir,
				f'phase_{row.phase}_{row.testNum}', row.quarter, 'feature_model')
		elif args.realtime:
			recent_phase = 4
			recent_testNum = 29
			recent_quarter = f"2025_Q2"
			load_value_network_path = os.path.join(parameters.BASE_DIR, 'output', args.model_dir,
												   f'phase_{recent_phase}_{row.testNum}', recent_quarter,
												   'value_{}'.format(args.model_version))
			load_policy_network_path = os.path.join(parameters.BASE_DIR, 'output', args.model_dir,
													f'phase_{recent_phase}_{row.testNum}', recent_quarter,
													'policy_{}'.format(args.model_version))
			feature_model_path = os.path.join(parameters.BASE_DIR, 'output', args.model_dir,
											  f'phase_{recent_phase}_{row.testNum}', recent_quarter, 'feature_model')
		else:
			load_value_network_path = " "
			load_policy_network_path = " "
			# feature model: FCM, PCA 모델
			feature_model_path = os.path.join(quarter_path, 'feature_model')
			if not os.path.isdir(feature_model_path): os.makedirs(feature_model_path)

		common_params = {'delayed_reward_threshold': args.delayed_reward_threshold,
						 'balance': args.balance}
		if not args.realtime:
			# 차트 데이터, 학습 데이터 준비
			train_chart_data, test_chart_data, training_data, test_data = data_manager.load_data(
				os.path.join(parameters.BASE_DIR,
							 'data/{}/'.format(args.ver)), row.stock_codes, feature_model_path,
				row.train_start, row.train_end, row.test_start, row.test_end, window_size=args.window_size,
				feature_window=args.feature_window,
				algorithm=args.algorithm, train=not args.test)
			# 공통 파라미터 설정

			# 강화학습 시작
			common_params.update({'stock_codes': row.stock_codes,
								  'num_of_stock': len(row.stock_codes),
								  'train_chart_data': train_chart_data, 'test_chart_data': test_chart_data,
								  'training_data': training_data, 'test_data': test_data})

		else:
			common_params.update({'stock_codes': row.stock_codes,
								  'num_of_stock': len(row.stock_codes),
								  'train_chart_data': None,
								  'test_chart_data': None,
								  'training_data': None,
								  # test_data는 inp_dim 계산에 사용될 수 있으므로
								  # 형식에 맞는 작은 numpy 배열을 전달합니다.
								  'test_data': np.zeros((1, args.window_size, 1))
								  })

		# f.open
		if args.algorithm == 'td3':
			if args.realtime:
				api_handler = KISApiHandler(APP_KEY, APP_SECRET, ACNT_NO, IS_MOCK)
			else:
				api_handler = None
			common_params['api_handler'] = api_handler




			learner = TD3_Agent(**{**common_params, 'output_path': quarter_path, 'lr': args.lr, 'test': args.test, 'phase': row.phase,
						'testNum': row.testNum, 'quarter': row.quarter, 'value_network_path' : value_network_path, 'policy_network_path' : policy_network_path,
						'load_policy_network_path' : load_policy_network_path, 'load_value_network_path' : load_value_network_path,
						'window_size': args.window_size,'api_handler':api_handler})
			if args.realtime:
				api_handler.get_access_token()
				learner.trade_realtime()
			else:
				if not args.test: learner.run(args.max_episode, args.num_step, args.noise, args.start_epsilon)
				learner.backtest(args.num_step)

