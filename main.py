import os
import sys
import logging
import argparse
import json
from parameters import parameters
import data_manager
from learner import TD3_Agent
from phase import phase2date
PHASE = 4

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--stock_code')
	parser.add_argument('--ver', choices=['ROK','USA','ETF'], default='ROK')
	parser.add_argument('--algorithm', choices=['td3','dsl','gdpg','gdqn','candle', 'attention','irdpg'], default='td3')
	parser.add_argument('--test', default=False)
	parser.add_argument('--model_version', default=29)
	parser.add_argument('--lr', type=float, default=0.001)
	parser.add_argument('--balance', type=int, default=100000000)
	parser.add_argument('--max_episode', type=int, default=30)
	parser.add_argument('--delayed_reward_threshold', type=float, default=0.03)
	parser.add_argument('--workers', type=int, default=1)
	parser.add_argument('--window_size', type=int, default=10)
	parser.add_argument('--feature_window', type=int, default=1)
	parser.add_argument('--num_step', type=int, default=1)
	parser.add_argument('--start_epsilon', type=float, default=0.02)
	parser.add_argument('--noise', type=float, default=0.3)
	args = parser.parse_args()

	# 출력 경로 설정
	output_path = os.path.join(parameters.BASE_DIR,
		'output/{}_{}_{}_{}_{}'.format(args.stock_code,args.window_size, args.max_episode, args.lr,args.noise))
	if not os.path.isdir(output_path): os.makedirs(output_path)

	# 파라미터 기록
	with open(os.path.join(output_path, 'params.json'), 'w') as f:
		f.write(json.dumps(vars(args)))

	# 로그 기록 설정
	file_handler = logging.FileHandler(filename=os.path.join(
		output_path, "{}.log".format(args.stock_code)), encoding='utf-8')
	stream_handler = logging.StreamHandler(sys.stdout)
	file_handler.setLevel(logging.DEBUG)
	stream_handler.setLevel(logging.INFO)
	logging.basicConfig(format="%(message)s",
		handlers=[file_handler, stream_handler], level=logging.DEBUG)

	for phase in range(1, PHASE+1):
		phase_path = os.path.join(output_path, 'phase_{}'.format(phase))
		start_date, end_date = phase2date(phase)

		# feature model: FCM, PCA 모델 -> 추후 저장 가능
		feature_model_path = os.path.join(phase_path, 'feature_model')
		if not os.path.isdir(feature_model_path): os.makedirs(feature_model_path)

		# 모델 경로 준비
		policy_network_path = os.path.join(
			phase_path, 'policy_{}'.format(args.stock_code))
		value_network_path = os.path.join(
			phase_path, 'value_{}'.format(args.stock_code))

		# 모델 재사용
		if args.test:
			load_value_network_path = os.path.join(
				phase_path, 'value_{}_{}'.format(args.stock_code, args.model_version))
			load_policy_network_path = os.path.join(
				phase_path, 'policy_{}_{}'.format(args.stock_code,args.model_version))
		else: load_value_network_path = " "; load_policy_network_path = " "

		# 차트 데이터, 학습 데이터 준비
		train_chart_data, test_chart_data, training_data, test_data = data_manager.load_data(
			os.path.join(parameters.BASE_DIR,
			'data/{}/{}.csv'.format(args.ver, args.stock_code)), feature_model_path,
			start_date, end_date, window_size=args.window_size, feature_window=args.feature_window, algorithm=args.algorithm)

		# 공통 파라미터 설정
		common_params = {'delayed_reward_threshold': args.delayed_reward_threshold,
					'balance' : args.balance}
		# 강화학습 시작
		common_params.update({'stock_code': args.stock_code,
							  'train_chart_data': train_chart_data, 'test_chart_data': test_chart_data,
							  'training_data': training_data,'test_data': test_data})
		# f.open
		if args.algorithm == 'td3':
			learner = TD3_Agent(**{**common_params, 'output_path': phase_path, 'lr': args.lr, 'test': args.test, 'phase': phase,
						'value_network_path' : value_network_path, 'policy_network_path' : policy_network_path,
						'load_policy_network_path' : load_policy_network_path, 'load_value_network_path' : load_value_network_path,
						'window_size': args.window_size})
			if not args.test: learner.run(args.max_episode, args.num_step, args.noise,args.start_epsilon)
			learner.backtest(args.num_step)
