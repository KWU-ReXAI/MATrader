# from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot
import os
import sys
import time
import dotenv
import logging
import numpy as np
import pandas as pd
from functools import reduce
from datetime import datetime
from network import MultiAgentTransformer
from feature import Cluster_Data
from parameters import parameters
from kis_api import KISApiHandler

FEATURES = 26

class RealtimeEnvironment:

    def __init__(self, stock_codes, fmpath, api):
        self.stock_codes = stock_codes
        self.fmpath = fmpath
        self.api = api

    def build_state(self, stocks):
        datas = []
        for stock in stocks:
            df = self.api.get_minute_candles_all_today(stock, max_pages=3)
            df.drop(columns=['time', 'acml_tr_pbmn'], inplace=True)
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            df['adj close'] = df['close']
            columns = ['date', 'open', 'high', 'low', 'close', 'adj close', 'volume', 'datetime']
            df = df[columns]
            dtype_map = {col: 'int64' for col in df.columns if col != 'date' and col != 'datetime'}
            df = df.astype(dtype_map)
            datas.append(df)
        common_values = reduce(
            lambda left, right: left.intersection(right),
            [set(df['datetime']) for df in datas]
        )
        datas = [df[df['datetime'].isin(common_values)].reset_index(drop=True) for df in datas]
        datas = [df.drop(columns=['datetime']) for df in datas]
        test_datas = []
        for idx, data in enumerate(datas):
            test_feature = Cluster_Data(stocks[idx], data, 1, self.fmpath,
                                        train=False)
            test_data = test_feature.load_data(0, 0, realtime=True)
            test_datas.append(test_data)
        test_data = np.stack(test_datas, axis=1)
        return test_data # done은 일단 제외

# class TradingWorker(QThread):
class TradingWorker:
    # status_signal = pyqtSignal(str)
    # update_table_signal = pyqtSignal(list)
    # account_summary_signal = pyqtSignal(dict)
    # === 분봉 ===
    # minute_candles_ready = pyqtSignal(str, object)  # code, DataFrame
    # trade_event = pyqtSignal(object)
    def __init__(self, api_handler, auto_execute=False):
        # super().__init__()
        self.api, self.auto_execute = api_handler, auto_execute
        self.is_running = False
        self.stock_list = [
            {'iscd': '005930', 'name': '삼성전자', 'qty': 0},
            {'iscd': '000660', 'name': 'SK하이닉스', 'qty': 0},
            {'iscd': '035720', 'name': '카카오', 'qty': 0}
        ]
        # === for MAT network ===
        self.fmpath = os.path.join('./output', 'realtime_tmp',
            f'phase_4_1', '2025_Q2', 'feature_model')
        load_network_path = os.path.join('./output', 'realtime_tmp',
            f'phase_4_1', '2025_Q2', 'mat_29')
        self.act_dim = parameters.NUM_ACTIONS
        self.inp_dim = FEATURES
        # === Create networks ===
        self.network = MultiAgentTransformer(self.inp_dim, self.act_dim, len(self.stock_list), 1, 64, 1)
        if os.path.exists(load_network_path + '.pt'):
            self.network.load_model(load_network_path)

    def calculate_trading_unit(self, price, balance, n_stocks):
        charge = 0.000142
        cash = balance // n_stocks
        trading_unit = int(cash // (price * (1 + charge))) if price else 0
        return trading_unit

    def run(self):
        self.is_running = True
        mode = "자동매매" if self.auto_execute else "모니터링"
        # self.status_signal.emit(f"{mode} 실행 중")
        logging.info(f"======== [{mode}] 모드를 시작합니다. ========")
        # === environment setting ===
        stock_codes = [stock['iscd'] for stock in self.stock_list]
        environment = RealtimeEnvironment(stock_codes, self.fmpath, self.api)
        while self.is_running:
            try:
                now = datetime.now()
                is_market_open = (datetime.strptime("09:00", "%H:%M").time() <= now.time() <= datetime.strptime("15:30", "%H:%M").time()) and \
                                 now.weekday() < 5

                if not is_market_open:
                    logging.info("장이 열리지 않아 자동매매 로직을 1분간 중지합니다.")
                    for _ in range(60):
                        if not self.is_running: break
                        time.sleep(1)
                    continue

                logging.info("계좌 잔고 정보 조회 시작...")
                balance_data = self.api.get_account_balance()
                if balance_data:
                    # self.account_summary_signal.emit(balance_data)
                    # 초기 stock 수량 정보 불러오기
                    for holding in balance_data['holdings']:
                        for stock in self.stock_list:
                            if stock['iscd'] == holding['iscd']:
                                stock['qty'] = holding['qty']
                    # self.update_table_signal.emit(self.stock_list)
                    logging.info("계좌 잔고 정보 업데이트 완료.")
                else:
                    logging.warning("계좌 잔고 정보 조회에 실패했습니다.")

                state = environment.build_state(stock_codes)
                policy, _, _ = self.network.act(np.array(state))
                for idx, stock in enumerate(self.stock_list):
                    if not self.is_running: break
                    buy_signal = True if policy[0][idx] == parameters.ACTION_BUY else False
                    sell_signal = True if policy[0][idx] == parameters.ACTION_SELL else False
                    buy_price, sell_price = self.api.get_asking_price(stock['iscd'])
                    stock['price'] = self.api.get_current_price(stock['iscd'])
                    trading_unit = self.calculate_trading_unit(buy_price, balance_data['deposit'], len(self.stock_list))
                    stock['signal'] = "대기"
                    if buy_signal and trading_unit > 0:
                        stock['signal'] = "매수 신호"
                        logging.info(f"[{stock['name']}] 매수 신호 발생!")
                        if self.auto_execute:
                            res = self.api.order_cash(stock['iscd'], trading_unit, 0, '02', order_type='01')
                            if res and res.get('rt_cd') == '0':
                                logging.info("매수 주문 완료")
                                stock['qty'] += trading_unit
                                # self.trade_event.emit({
                                #     "code": stock['iscd'],
                                #     "name": stock['name'],
                                #     "side": "BUY",
                                #     "qty": trading_unit,
                                #     "price": buy_price,
                                #     "ts": datetime.now().isoformat()   # 차트에서 가장 가까운 분봉에 매핑
                                # })
                    elif sell_signal and stock.get('qty', 0) > 0:
                        stock['signal'] = "매도 신호"
                        logging.info(f"[{stock['name']}] 매도 신호 발생!")
                        if self.auto_execute:
                            res = self.api.order_cash(stock['iscd'], stock['qty'], 0, '01', order_type='01')
                            if res and res.get('rt_cd') == '0':
                                logging.info("매도 주문 완료")
                                # self.trade_event.emit({
                                #     "code": stock['iscd'],
                                #     "name": stock['name'],
                                #     "side": "SELL",
                                #     "qty": stock['qty'],
                                #     "price": sell_price,
                                #     "ts": datetime.now().isoformat()
                                # })
                                stock['qty'] = 0

                    # self.update_table_signal.emit(self.stock_list)
                balance_data = self.api.get_account_balance()
                if balance_data:
                    # self.account_summary_signal.emit(balance_data)
                    logging.info("거래 후 계좌 잔고 정보 업데이트 완료.")
                else:
                    logging.warning("계좌 잔고 정보 조회에 실패했습니다.")
                logging.info(f"모든 종목 확인 완료. {180}초 후 다시 시작합니다.")
                for _ in range(180):
                    if not self.is_running: break
                    time.sleep(1)

            except Exception as e:
                logging.error(f"워커 스레드 실행 중 오류 발생: {e}")
                logging.info("60초 후 다시 시도합니다.")
                time.sleep(60)

        # self.status_signal.emit("대기중")
        logging.info("======== 워커 스레드가 정상적으로 중지되었습니다. ========")

    def stop(self):
        self.is_running = False

    # @pyqtSlot(str, bool)  # code, all_today
    def fetch_minute_candles(self, code: str, all_today: bool):
        try:
            if all_today:
                df = self.api.get_minute_candles_all_today(code)
            else:
                df = self.api.get_minute_candles(code)
            # self.minute_candles_ready.emit(code, df)
        except Exception as e:
            # self.minute_candles_ready.emit(code, pd.DataFrame({"error":[str(e)]}))
            print(f'error: {str(e)}') # pyQt 적용 시 삭제!!!

if __name__ == '__main__':
    dotenv.load_dotenv()
    api = KISApiHandler(os.getenv("APP_KEY"), os.getenv("APP_SECRET"), os.getenv("ACNT_NO"), is_mock=True)
    output_path = os.path.join(parameters.BASE_DIR, 'output')
    file_handler = logging.FileHandler(filename=os.path.join(
        output_path, f"{'realtime'}.log"), encoding='utf-8')
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setLevel(logging.DEBUG)
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(format="%(message)s",
                        handlers=[file_handler, stream_handler], level=logging.DEBUG)
    worker = TradingWorker(api, True)
    worker.run()