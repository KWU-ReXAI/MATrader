import os
import sys
import json
import torch
import random
import logging
import argparse
import datetime
import numpy as np
import pandas as pd
import data_manager
from learner import MATagent
from phase import phase2quarter
from parameters import parameters

SEED = 42
def set_seed(seed):
	"""
	모든 라이브러리의 랜덤 시드를 고정하는 함수
	"""
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--stock_dir', type=str)
	parser.add_argument('--output_name', default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
	parser.add_argument('--ver', choices=['ROK','USA','ETF'], default='ROK')
	parser.add_argument('--algorithm', choices=['mat','td3','dsl','gdpg','gdqn','candle', 'attention','irdpg'], default='mat')
	parser.add_argument('--test', action='store_true')
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
	parser.add_argument('--ppo_epoch', type=int, default=15)
	parser.add_argument('--num_mini_batch', type=int, default=1)
	args = parser.parse_args()

	set_seed(SEED)
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
	dfs = []
	for row in quarters_df.itertuples():
		quarter_path = os.path.join(output_path, f'phase_{row.phase}_{row.testNum}', row.quarter)
		if not os.path.isdir(quarter_path): os.makedirs(quarter_path)
		### 뽑힌 주식 없을 때 예외처리: 폴더만 생성 ###
		if not row.stock_codes:
			continue

		# 모델 경로 준비
		network_path = os.path.join(
			quarter_path, 'mat')

		# 모델 재사용
		if args.test:
			load_network_path = os.path.join(parameters.BASE_DIR, 'output', args.model_dir,
				f'phase_{row.phase}_{row.testNum}', row.quarter, 'mat_{}'.format(args.model_version))
			feature_model_path = os.path.join(parameters.BASE_DIR, 'output', args.model_dir,
				f'phase_{row.phase}_{row.testNum}', row.quarter, 'feature_model')
		else:
			load_network_path = " "
			# feature model: FCM, PCA 모델
			feature_model_path = os.path.join(quarter_path, 'feature_model')
			if not os.path.isdir(feature_model_path): os.makedirs(feature_model_path)

		# 차트 데이터, 학습 데이터 준비
		train_chart_data, test_chart_data, training_data, test_data = data_manager.load_data(
			os.path.join(parameters.BASE_DIR,
			'data/{}/'.format(args.ver)), row.stock_codes, feature_model_path,
			row.train_start, row.train_end, row.test_start, row.test_end, window_size=args.window_size, train=not args.test)
		# 공통 파라미터 설정
		common_params = {'delayed_reward_threshold': args.delayed_reward_threshold,
					'balance' : args.balance}
		# 강화학습 시작
		common_params.update({'stock_codes': row.stock_codes,
							  'num_of_stock': len(row.stock_codes),
							  'train_chart_data': train_chart_data, 'test_chart_data': test_chart_data,
							  'training_data': training_data,'test_data': test_data})
		# f.open
		if args.algorithm == 'mat':
			learner = MATagent(**{**common_params, 'output_path': quarter_path, 'lr': args.lr, 'test': args.test, 'phase': row.phase,
						'testNum': row.testNum, 'quarter': row.quarter, 'network_path' : network_path,
						'load_network_path' : load_network_path,
						'window_size': args.window_size,
						'ppo_epoch': args.ppo_epoch,
						'num_mini_batch': args.num_mini_batch})
			if not args.test: learner.run(args.max_episode, args.num_step, args.noise,args.start_epsilon)
			df = learner.backtest(args.num_step)
			dfs.append(df)
	full_df = pd.concat(dfs)
	full_df.to_csv(os.path.join(output_path, f'{args.output_name}_result.csv'), index=False, encoding='utf-8-sig')
