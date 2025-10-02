import pandas as pd
import numpy as np
from feature import Cluster_Data
from realtime_feature import Realtime_Data
from tqdm import tqdm
from kis_api import KISApiHandler  # KISApiHandler 임포트
from datetime import datetime, timedelta # 시간 관련 모듈 임포트
import logging

def load_data(fpath, stocks:list, fmpath,train_start, train_end, test_start, test_end, window_size=1,feature_window = 1,algorithm='td3', train=True):
    # 특정 주식의 거래 정지 기간이 훈련(테스트) 기간 내 포함되면,
    # 모든 주식들 그 기간에 거래 안 함
    dfs = []
    for stock in stocks:
        path = f'{fpath}{stock}.csv'
        data = pd.read_csv(path, thousands=',',
            converters={'date': lambda x: str(x)})
        data.columns = ['date', 'open', 'high', 'low', 'close', 'adj close', 'volume']
        # 기간 필터링
        data.replace(0, pd.NA, inplace=True)
        dfs.append(data)
    rows_to_keep = ~pd.concat([df.isna().any(axis=1) for df in dfs], axis=1).any(axis=1)
    df_stocks = [df[rows_to_keep] for df in dfs]
    df_stocks = [df.reset_index(drop=True) for df in df_stocks]

    if train:
        train_chart_datas = []
        train_datas = []
    test_chart_datas = []; test_datas = []

    for idx, data in enumerate(tqdm(df_stocks, desc='Data preprocessing')):
        if train:
            train_feature = Cluster_Data(stocks[idx], data,train_start, train_end, window_size, fmpath, feature_window, train=True)
            train_data = train_feature.load_data()
            train_chart_data = data[(data['date'] >= str(train_start)) & (data['date'] <= str(train_end))].dropna()
        test_feature = Cluster_Data(stocks[idx], data, test_start, test_end, window_size, fmpath, feature_window, train=False)
        test_data = test_feature.load_data()
        test_chart_data = data[(data['date'] >= str(test_start)) & (data['date'] <= str(test_end))].dropna()

        if train:
            train_datas.append(train_data)
            train_chart_datas.append(train_chart_data)
        test_datas.append(test_data); test_chart_datas.append(test_chart_data)
    if train:
        train_chart_data = np.stack(train_chart_datas, axis=1)
        train_data = np.stack(train_datas, axis=2)
        train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], -1)
    test_chart_data = np.stack(test_chart_datas, axis=1)
    test_data = np.stack(test_datas, axis=2)
    test_data = test_data.reshape(test_data.shape[0], test_data.shape[1], -1)
    # train/test data: (거래일 수, time window, 종목 수 * feature 수)
    # train/test chart data: (거래일 수, 종목 수, 가격 데이터 특징 수)
    if not train:
        train_chart_data = None
        train_data = None
    return train_chart_data, test_chart_data, train_data, test_data

"""def load_realtime_data(api_handler, stocks: list, fmpath, window_size=1, feature_window=1):
        # 특정 주식의 거래 정지 기간이 훈련(테스트) 기간 내 포함되면,
        # 모든 주식들 그 기간에 거래 안 함
    dfs = []
    for stock in stocks:
        data = api_handler.get_minute_candles_all_today(stocks)
        data.columns = ['date', 'open', 'high', 'low', 'close', 'adj close', 'volume']
            # 기간 필터링
        data.replace(0, pd.NA, inplace=True)
        dfs.append(data)
    rows_to_keep = ~pd.concat([df.isna().any(axis=1) for df in dfs], axis=1).any(axis=1)
    df_realtime = [df[rows_to_keep] for df in dfs]
    df_realtime = [df.reset_index(drop=True) for df in df_realtime]
    df_realtime['date'] = df_realtime['dateTime'].dt.strftime('%Y%m%d%H%M%S')

    realtime_chart_datas = []
    realtime_datas = []

    for idx, data in tqdm(enumerate(df_realtime), desc='Data preprocessing'):
        realtime_feature = Cluster_Data(
                stock=stocks[idx],
                data=df_realtime,
                start=df_realtime['date'].iloc[0],
                end=df_realtime['date'].iloc[-1],
                window_size=window_size,
                fmpath=fmpath,
                feature_window=feature_window,
                train=False
            )
        realtime_data = realtime_feature.load_data()
        realtime_chart_data = data.dropna()

        realtime_datas.append(realtime_data);
        realtime_chart_datas.append(realtime_chart_data)

    realtime_chart_data = np.stack(realtime_chart_datas, axis=1)
    realtime_data = np.stack(realtime_datas, axis=2)
    realtime_data = realtime_data.reshape(realtime_data.shape[0], realtime_data.shape[1], -1)
    # train/test data: (거래일 수, time window, 종목 수 * feature 수)
    # train/test chart data: (거래일 수, 종목 수, 가격 데이터 특징 수)

    return realtime_chart_data, realtime_data"""


def load_realtime_data(api_handler, stock_codes, fmpath=None, window_size=None, feature_window=None):
    try:
        latest_states = []
        latest_charts = []

        # 1. 각 종목에 대해 개별적으로 API 호출 및 데이터 처리
        for stock_code in stock_codes:
            df_realtime = api_handler.get_minute_candles_all_today(stock_code)

            if len(df_realtime) < window_size:
                logging.warning(f"[{stock_code}] 데이터 부족 (필요: {window_size}, 현재: {len(df_realtime)})")
                return None, None

            # 수정: API 응답에 맞게 컬럼을 준비합니다.
            #df_realtime = df_realtime.rename(columns={'close': 'adj close'})
            df_realtime['adj close'] = df_realtime['close']
            df_realtime['date'] = df_realtime['datetime'].dt.strftime('%Y%m%d%H%M%S')
            final_columns = ['date', 'open', 'high', 'low', 'close', 'adj close', 'volume']
            df_realtime = df_realtime[final_columns]

            # 2. 훈련 때와 동일한 방식으로 피처 생성 (train=False)
            feature_generator = Realtime_Data(
                stock=stock_code,
                data=df_realtime,
                #start=df_realtime['date'].iloc[0],
                #end=df_realtime['date'].iloc[-1],
                window_size=window_size,
                fmpath=fmpath,
                feature_window=feature_window,
                train=False
            )
            all_feature_data = feature_generator.load_data(0,0)

            # 3. 수정: 생성된 전체 피처 중 "가장 마지막 state"만 사용합니다.
            latest_states.append(all_feature_data[-1])

            # 현재가 조회를 위한 가장 마지막 차트 데이터
            chart_point = df_realtime[['date', 'open', 'high', 'low', 'close', 'adj close', 'volume']].iloc[
                          -1:].to_numpy()
            latest_charts.append(chart_point)

        # (종목 수, window_size, 피처 수) -> (1, window_size, 종목 수 * 피처 수)
        state = np.stack(latest_states, axis=1)  # (window_size, num_stocks, features)
        state = state.reshape(1, window_size, -1)  # (1, window_size, num_stocks * features)

        chart_data = np.stack(latest_charts, axis=1)  # (1, num_stocks, price_features)

        # 5. 수정: (state, chart_data) 순서로 반환합니다.
        return state, chart_data

    except Exception as e:
        logging.error(f"실시간 데이터 처리 중 오류 발생: {e}", exc_info=True)
        return None, None