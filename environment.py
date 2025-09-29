import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import data_manager

class Environment:
    DATE = 0        # 날짜의 위치
    PRICE_IDX = 4   # 종가의 위치
    BUY = -2        # 매수 시그널 위치
    SELL = -1       # 매도 시그널 위치
    def __init__(self, chart_data=None, training_data=None):
        self.chart_data = chart_data
        self.plt_data = chart_data.copy()
        # plt_data: (거래일별, 종목별, 가격 데이터들) -> 가격 데이터들 마지막에 매수, 매도 시그널 차원 추가
        self.plt_data = np.pad(self.plt_data, pad_width=((0, 0), (0, 0), (0, 2)), mode='constant', constant_values=0)
        self.training_data = training_data
        self.observation = None
        self.idx = 0
        self.training_data_idx = 0

    def reset(self):
        self.observation = None
        self.idx = 0
        self.training_data_idx = 0

    # def observe(self):
    #     if len(self.chart_data) > self.idx + 1:
    #         self.observation = self.chart_data.iloc[self.idx]
    #         self.idx += 1
    #         return self.observation
    #     return None
    
    def build_state(self):
        if len(self.training_data) > self.training_data_idx + 1:
            self.sample = self.training_data[self.training_data_idx]
            self.training_data_idx += 1
            return self.sample.tolist(),False
        return None,True

    # def get_price(self):
    #     self.observe()
    #     if self.observation is not None:
    #         return self.observation[self.PRICE_IDX]
    #     return None
    #
    # def prev_price(self):
    #     stock_price = self.chart_data.iloc[self.idx-1]
    #     return stock_price[self.PRICE_IDX]

    def curr_price(self):
        if len(self.chart_data) > self.idx:
            stock_price = self.chart_data[self.idx]
            return stock_price[:, self.PRICE_IDX].astype(np.float32)
        return None
    
    def next_price(self):
        if len(self.chart_data) > self.idx + 1:
            stock_price = self.chart_data[self.idx+1]
            return stock_price[:, self.PRICE_IDX].astype(np.float32)
        return None

    def get_date(self):
        curr_data = self.chart_data[self.idx]
        return curr_data[0][0]

    # def set_chart_data(self, chart_data):
    #     self.chart_data = chart_data

    def set_buy_signal(self, stock):
        price = self.plt_data[self.idx, stock, self.PRICE_IDX]
        self.plt_data[self.idx, stock, self.BUY] = price

    def set_sell_signal(self, stock):
        price = self.plt_data[self.idx, stock, self.PRICE_IDX]
        self.plt_data[self.idx, stock, self.SELL] = price

    def plt_result(self,path, stock_codes):
        plt.style.use('fivethirtyeight')
        fig, ax = plt.subplots(figsize=(32, 16))

        colors = plt.cm.get_cmap('tab10', len(stock_codes))
        for idx, stock in enumerate(stock_codes):
            color = colors(idx)
            # 범례 중복 표시 방지(마지막 그래프 그릴 때만 범례 추가)
            buy_signal = 'Buy' if idx == len(stock_codes) - 1 else None
            sell_signal = 'Sell' if idx == len(stock_codes) - 1 else None
            # 종목별 종가 그리기
            ax.plot(self.plt_data[:, idx, self.DATE], self.plt_data[:, idx, self.PRICE_IDX],
                    label=f'{stock} Close Price', color=color, alpha=0.4)
            # 매수 시그널 그리기
            buy_mask = self.plt_data[:, idx, self.BUY] > 0
            ax.scatter(self.plt_data[buy_mask, idx, self.DATE], self.plt_data[buy_mask, idx, self.BUY],
                       color='green', marker='^', label=buy_signal, alpha=1)
            # 매도 시그널 그리기
            sell_mask = self.plt_data[:, idx, self.SELL] > 0
            ax.scatter(self.plt_data[sell_mask, idx, self.DATE], self.plt_data[sell_mask, idx, self.SELL],
                       color='red', marker='v', label=sell_signal, alpha=1)

        # x축의 날짜 개수 조절
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=10))

        ax.set_xlabel('Stock Data Time', fontsize=18)
        ax.set_ylabel('Close Price', fontsize=18)

        ax.legend(loc='best')

        plt.savefig(path + "_all_stocks.png")
        plt.close(fig)


class RealtimeEnvironment:
    DATE = 0  # 날짜의 위치
    PRICE_IDX = 4  # 종가의 위치
    BUY = -2  # 매수 시그널 위치
    SELL = -1  # 매도 시그널 위치
    def __init__(self, api_handler, stock_codes, fmpath, window_size, feature_window):
        """
        실시간 거래 환경을 위한 클래스
        """
        self.api_handler = api_handler
        self.stock_codes = stock_codes
        self.fmpath = fmpath
        self.window_size = window_size
        self.feature_window = feature_window

        self.chart_data = None  # 현재 시세 데이터를 저장할 변수

    def build_state(self):
        """
        data_manager를 통해 실시간 데이터를 가져와 모델의 입력(state)으로 변환합니다.
        """
        # 1단계에서 만든 실시간 데이터 로더 함수 호출
        state, chart_data = data_manager.load_realtime_data(
            self.api_handler,
            self.stock_codes,
            self.fmpath,
            self.window_size,
            self.feature_window
        )

        if state is None:
            return None, True  # 데이터를 가져오지 못하면 종료 신호(True) 반환

        self.chart_data = chart_data

        # state와 종료 여부(False) 반환
        return state, False

    def curr_price(self):
        if self.chart_data is not None:
            # (1, num_stocks, num_features) -> (num_stocks, num_features)
            chart_data_squeezed = np.squeeze(self.chart_data, axis=0)
            return chart_data_squeezed[:, self.PRICE_IDX].astype(np.float32)
        return None
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲