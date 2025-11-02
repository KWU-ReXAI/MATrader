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
import matplotlib.pyplot as plt
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
	parser.add_argument('--model_version', default=99)
	parser.add_argument('--lr', type=float, default=0.001)
	parser.add_argument('--balance', type=int, default=100000000)
	parser.add_argument('--max_episode', type=int, default=100)
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

	plt.rcParams['font.family'] = 'Malgun Gothic'
	plt.rcParams['axes.unicode_minus'] = False
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
	dfs_buy_hold = []
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
		df_buy_hold = data_manager.buy_hold_pv(os.path.join(parameters.BASE_DIR,
			'data/{}/'.format(args.ver)), row.stock_codes, row.test_start, row.test_end, row.phase, row.quarter)
		dfs_buy_hold.append(df_buy_hold)
	full_df = pd.concat(dfs)
	full_df_buy_hold = pd.concat(dfs_buy_hold)
	full_df.to_csv(os.path.join(output_path, f'{args.output_name}_result.csv'), index=False, encoding='utf-8-sig')


	def calculate_annualized_return(rr_series):
		"""
        분기별 수익률(rr) 시리즈를 받아
        연간 환산 기하 평균 수익률을 계산합니다.
        """
		if rr_series.empty:
			return np.nan

		# (1+r1)*(1+r2)*... - 1
		cumulative_return = (1 + rr_series).prod() - 1

		return cumulative_return


	# --- 5. Phase별 데이터 집계 ---

	def aggregate_by_phase(df):
		"""
        데이터프레임을 phase별로 그룹화하여
        3가지 핵심 지표를 계산합니다.
        """
		# 1. 연간 환산 수익률 (기하 평균)
		ann_rr = df.groupby('phase')['rr'].apply(calculate_annualized_return)

		# 2. 평균 Sharpe Ratio
		mean_sr = df.groupby('phase')['sr'].mean()

		# 3. 평균 MDD
		mean_mdd = df.groupby('phase')['mdd'].mean()

		# 결과 데이터프레임 생성
		summary = pd.DataFrame({
			'Annualized RR': ann_rr,
			'Mean SR': mean_sr,
			'Mean MDD': mean_mdd
		})
		return summary


	# 각 전략별로 집계
	model_summary = aggregate_by_phase(full_df)
	bnh_summary = aggregate_by_phase(full_df_buy_hold)

	# --- 6. 플로팅을 위한 데이터 준비 ---
	# 비교를 위해 두 요약 데이터를 합칩니다.
	df_plot_rr = pd.DataFrame({
		'Model': model_summary['Annualized RR'],
		'Buy and Hold': bnh_summary['Annualized RR']
	})

	df_plot_sr = pd.DataFrame({
		'Model': model_summary['Mean SR'],
		'Buy and Hold': bnh_summary['Mean SR']
	})

	df_plot_mdd = pd.DataFrame({
		'Model': model_summary['Mean MDD'],
		'Buy and Hold': bnh_summary['Mean MDD']
	})

	# --- 7. 그래프 그리기 (1x3 서브플롯) ---
	fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
	colors = ['#78b0e2', '#dd3333']  # Model (Blue), B&H (Red)

	# 플롯 1: 연간 환산 수익률
	df_plot_rr.plot(kind='bar', ax=axes[0], color=colors, rot=0)
	axes[0].set_title('Phase별 수익률', fontsize=14)
	axes[0].set_ylabel('Rate of Return')
	axes[0].axhline(0, color='black', linewidth=0.8)
	axes[0].grid(True, linestyle=':', alpha=0.7, axis='y')
	axes[0].legend()

	# 플롯 2: 평균 Sharpe Ratio
	df_plot_sr.plot(kind='bar', ax=axes[1], color=colors, rot=0)
	axes[1].set_title('Phase별 평균 Sharpe Ratio', fontsize=14)
	axes[1].set_ylabel('Mean SR')
	axes[1].axhline(0, color='black', linewidth=0.8)
	axes[1].grid(True, linestyle=':', alpha=0.7, axis='y')
	axes[1].legend()

	# 플롯 3: 평균 MDD
	df_plot_mdd.plot(kind='bar', ax=axes[2], color=colors, rot=0)
	axes[2].set_title('Phase별 평균 MDD', fontsize=14)
	axes[2].set_ylabel('Mean MDD')
	# MDD는 음수가 없으므로 0선은 생략 가능 (필요시 axhline 추가)
	axes[2].grid(True, linestyle=':', alpha=0.7, axis='y')
	axes[2].legend()

	# 전체 제목 및 레이아웃 조정
	fig.suptitle('모델 vs Buy & Hold 성과 비교', fontsize=18, y=1.03)
	plt.tight_layout(rect=[0, 0.03, 1, 0.95])

	# 그래프 저장 및 출력
	plt.savefig(f'{output_path}/performance_comparison.png', dpi=300, bbox_inches='tight')
