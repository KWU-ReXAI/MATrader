import time
import numpy as np

from kis_api import KISApiHandler
from environment import Environment
import os
import sys
import logging
import argparse
import json
import datetime
from parameters import parameters
import data_manager
from learner import TD3_Agent
from phase import phase2quarter


# ------------------------------
# 설정 영역
# ------------------------------
# 실제 투자 시 모의투자 계좌번호와 발급받은 API 키를 입력하세요.
APP_KEY = os.getenv("APP_KEY")
APP_SECRET = os.getenv("APP_SECRET")
ACNT_NO = os.getenv("ACNT_NO")  # 모의투자 계좌
IS_MOCK = True  # True: 모의투자, False: 실전투자

# 거래할 주식 목록 (모델 학습 시 사용했던 종목과 동일해야 함)
STOCK_CODES = ['005930', '000660', '035720']
WINDOW_SIZE = 10  # 학습 시 설정했던 window_size

# 학습된 모델 가중치 파일이 있는 경로
# 예: output/result/phase_p4_1/2023_Q4/
MODEL_PATH = "output/result/phase_4_1/2025_Q2/"
# 가장 성능이 좋았던 에포크의 모델 버전을 지정
MODEL_VERSION = 29

# 특징 추출 모델(FCM, PCA)이 저장된 경로
FEATURE_MODEL_PATH = os.path.join(MODEL_PATH, 'feature_model')


# ------------------------------
# 액션 변환 함수
# ------------------------------
def map_action(policy):
    # -3~3 사이의 policy 값을 매수(0), 홀드(1), 매도(2)로 변환
    points = np.linspace(-1 * parameters.NUM_ACTIONS, parameters.NUM_ACTIONS, parameters.NUM_ACTIONS + 1)
    points[-1] += 1e-10
    values = np.digitize(policy, points) - 1
    return values


# ------------------------------
# 메인 실행 함수
# ------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stock_dir', type=str)
    # parser.add_argument('--output_name', default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--output_name', type=str, default='result')
    parser.add_argument('--ver', choices=['ROK', 'USA', 'ETF'], default='ROK')
    parser.add_argument('--algorithm', choices=['td3', 'dsl', 'gdpg', 'gdqn', 'candle', 'attention', 'irdpg'],
                        default='td3')
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
    parser.add_argument('--debug', type=bool, default=False)
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
    # 1. API 핸들러 초기화
    api = KISApiHandler(APP_KEY, APP_SECRET, ACNT_NO, IS_MOCK)
    print("KIS API 핸들러 초기화 완료.")

    # 2. 강화학습 에이전트 및 환경 초기화
    # policy, value 네트워크 경로 설정
    policy_net_path = os.path.join(MODEL_PATH, f'policy_{MODEL_VERSION}')
    value_net_path = os.path.join(MODEL_PATH, f'value_{MODEL_VERSION}')

    # TD3_Agent는 네트워크 로딩 용도로만 사용
    learner = TD3_Agent(
        stock_codes=STOCK_CODES, num_of_stock=len(STOCK_CODES), phase=0, testNum=0, quarter='',
        train_chart_data=None, test_chart_data=None, training_data=None, test_data=np.zeros((1, WINDOW_SIZE, 1)),
        # 형식적으로 빈 데이터 전달
        window_size=WINDOW_SIZE, test=True,
        load_policy_network_path=policy_net_path,
        load_value_network_path=value_net_path
    )
    print(f"학습된 모델(ver.{MODEL_VERSION}) 로딩 완료.")

    env = Environment(api, STOCK_CODES, WINDOW_SIZE, FEATURE_MODEL_PATH)
    print("실시간 거래 환경 초기화 완료.")

    # 3. 실시간 거래 루프
    print("======== 실시간 자동매매를 시작합니다. ========")
    while True:
        try:
            # 장 시간인지 확인 (09:00 ~ 15:30)
            now = datetime.now()
            is_market_open = (now.time() >= datetime.strptime("09:00", "%H:%M").time() and
                              now.time() <= datetime.strptime("15:30", "%H:%M").time())

            if not is_market_open:
                print("장이 열리지 않았습니다. 1분 후 다시 확인합니다.")
                time.sleep(60)
                continue

            # 1단계: 최신 상태(State) 받아오기
            print("\n최신 시장 데이터로 상태(State)를 생성합니다...")
            state, done = env.build_state()
            if done or state is None:
                print("상태 생성에 실패했습니다. 1분 후 재시도합니다.")
                time.sleep(60)
                continue

            # 2단계: 모델을 통해 행동(Action) 결정
            policy = learner.network.actor_predict(state)
            actions = map_action(policy[0])
            print(f"모델 결정 -> Policy: {np.round(policy[0], 2)}, Actions: {actions}")

            # 3단계: 잔고 확인 및 주문 실행
            balance = api.get_account_balance()
            if balance is None:
                print("잔고 조회에 실패했습니다.")
                time.sleep(60)
                continue

            deposit = balance['deposit']  # 예수금
            holdings = {h['iscd']: h for h in balance['holdings']}  # 보유 주식

            for i, code in enumerate(STOCK_CODES):
                action = actions[i]

                if action == parameters.ACTION_BUY:  # 매수
                    # 1주를 살 돈이 있는지 확인
                    current_price = env.curr_price()[i]
                    if deposit >= current_price:
                        print(f"[{code}] 매수 신호 발생. 1주 시장가 매수 주문을 실행합니다.")
                        res = api.order_cash(code, 1, 0, '02', order_type='01')  # 02: 매수, 01: 시장가
                        print(f"  -> 주문 결과: {res}")
                        deposit -= current_price  # 예상 예수금 차감
                    else:
                        print(f"[{code}] 매수 신호 발생했으나, 예수금이 부족합니다.")

                elif action == parameters.ACTION_SELL:  # 매도
                    # 보유 주식이 있는지 확인
                    if code in holdings and holdings[code]['qty'] > 0:
                        qty_to_sell = holdings[code]['qty']
                        print(f"[{code}] 매도 신호 발생. 보유 수량({qty_to_sell}주) 시장가 매도 주문을 실행합니다.")
                        res = api.order_cash(code, qty_to_sell, 0, '01', order_type='01')  # 01: 매도
                        print(f"  -> 주문 결과: {res}")
                    else:
                        print(f"[{code}] 매도 신호 발생했으나, 보유 주식이 없습니다.")

                # 홀드는 아무것도 하지 않음

            # 4단계: 다음 분봉까지 대기
            print("모든 종목 확인 완료. 60초 후 다음 거래를 시작합니다.")
            time.sleep(60)

        except Exception as e:
            print(f"오류 발생: {e}")
            print("60초 후 재시도합니다.")
            time.sleep(60)