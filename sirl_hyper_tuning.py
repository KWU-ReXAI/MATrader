import os
from dotenv import load_dotenv
import sys
import logging
import argparse
import json
from parameters import parameters
import data_manager
from learner import TD3_Agent
from phase import phase2date
import wandb

load_dotenv()

PHASE = 4

def sweep(code):
	with wandb.init():
		config = wandb.config
			
		stock_code = code
		ver = "ROK"
		algorithm = "td3"
		test = False
		model_version = "29" # 없애는게 좋을 듯
		lr = config.lr
		balance = 10000000
		max_episode = config.max_episode
		delayed_reward_threshold = 0.03
		output_name = code + "_TD3"
		workers = 1
		window_size = config.window_size
		feature_window = config.feature_window
		num_step = 2
		start_epsilon = 0.02
		noise = config.noise
		

		# 출력 경로 설정
		output_path = os.path.join(parameters.BASE_DIR,
			'output/{}_{}_{}_{}_{}'.format(output_name, window_size, max_episode, lr,noise))
		if not os.path.isdir(output_path): os.makedirs(output_path)

		# # 파라미터 기록
		# with open(os.path.join(output_path, 'params.json'), 'w') as f:
		# 	f.write(json.dumps(vars(args)))

		# 로그 기록 설정
		file_handler = logging.FileHandler(filename=os.path.join(
			output_path, "{}.log".format(output_name)), encoding='utf-8')
		stream_handler = logging.StreamHandler(sys.stdout)
		file_handler.setLevel(logging.DEBUG)
		stream_handler.setLevel(logging.INFO)
		logging.basicConfig(format="%(message)s",
			handlers=[file_handler, stream_handler], level=logging.DEBUG)
		
		# Walk-forward validation 내부 루프
		all_pv = [] # 모든 phase의 평균 pv 모음
		for phase in range(1, PHASE+1):
			all_pv_per_phase = [] # 한 phase의 n회치 pv 모음
			phase_path = os.path.join(output_path, 'phase_{}'.format(phase))
			start_date, end_date = phase2date(phase)

			# feature model: FCM, PCA 모델 -> 추후 저장 가능
			feature_model_path = os.path.join(phase_path, 'feature_model')
			if not os.path.isdir(feature_model_path): os.makedirs(feature_model_path)

			# 모델 경로 준비
			policy_network_path = os.path.join(
				phase_path, 'policy_{}'.format(output_name))
			value_network_path = os.path.join(
				phase_path, 'value_{}'.format(output_name))

			# 모델 재사용
			if test:
				load_value_network_path = os.path.join(
					phase_path, 'value_{}_{}'.format(output_name, model_version))
				load_policy_network_path = os.path.join(
					phase_path, 'policy_{}_{}'.format(output_name, model_version))
			else: load_value_network_path = " "; load_policy_network_path = " "

			# 차트 데이터, 학습 데이터 준비
			train_chart_data, test_chart_data, training_data, test_data = data_manager.load_data(
				os.path.join(parameters.BASE_DIR,
				'data/{}/{}.csv'.format(ver, stock_code)), feature_model_path,
				start_date, end_date, window_size=window_size, feature_window=feature_window, algorithm=algorithm)

			# 공통 파라미터 설정
			common_params = {'delayed_reward_threshold': delayed_reward_threshold,
						'balance' : balance}
			# 강화학습 시작
			common_params.update({'stock_code': stock_code,
									'train_chart_data': train_chart_data, 'test_chart_data': test_chart_data,
									'training_data': training_data,'test_data': test_data})
			# f.open
			if algorithm == 'td3':
				learner = TD3_Agent(**{**common_params, 'output_path': phase_path, 'lr': lr, 'test': test, 'phase': phase,
							'value_network_path' : value_network_path, 'policy_network_path' : policy_network_path,
							'load_policy_network_path' : load_policy_network_path, 'load_value_network_path' : load_value_network_path,
							'window_size': window_size})
				pv = learner.run(max_episode, num_step, noise,start_epsilon)
				all_pv_per_phase.append(pv)
			avg_pv_per_phase = sum(all_pv_per_phase) / len(all_pv_per_phase)
			
			all_pv.append(avg_pv_per_phase)
		avg_pv = sum(all_pv) / len(all_pv)

		# 하이퍼파라미터 최적화 과정에서 pv값의 변화 기록
		for phase, pv in enumerate(all_pv):
			wandb.log({f"phase{phase}": pv})
		wandb.log({"walk_forward_score": avg_pv})

# 2: 검색 공간 정의하기
sweep_config = {
    "method": "bayes",
    "metric": {
        "goal": "maximize",
        "name": "walk_forward_score"
    },
    "parameters": {
        "lr": {"min": 0.001, "max": 0.1, "q": 0.001, "distribution": "q_uniform"},
        "window_size": {"values": [10, 15, 20]},
        "feature_window": {"values": [10, 15, 20]},
        "noise": {"min": 0.1, "max": 0.9, "q":0.1, "distribution": "q_uniform"},
        "max_episode": {"min": 10, "max": 30, "distribution": "int_uniform"}
    },
}

wandb.login(key=os.environ.get("WANDB_API_KEY"))

# 3: 스윕 시작하기
code_list = ["SK_Innovation", "S_hynix","Samsung_Electronics", "NAVER_Corp", "LG_Electronics", "Hyundai_Motor"]

for code in code_list:
	sweep_id = wandb.sweep(sweep=sweep_config, project=f"sirl_optimizer_{code}")
	wandb.agent(sweep_id, function=lambda: sweep(code), count=50)