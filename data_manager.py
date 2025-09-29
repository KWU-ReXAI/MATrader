import pandas as pd
import numpy as np
from feature import Cluster_Data
from tqdm import tqdm
from kis_api import KISApiHandler  # KISApiHandler 임포트
from datetime import datetime, timedelta # 시간 관련 모듈 임포트

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

    for idx, data in tqdm(enumerate(df_stocks), desc='Data preprocessing'):
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


# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# 실시간 데이터 처리를 위해 새롭게 추가된 함수입니다.
def load_realtime_data(api_handler: KISApiHandler, stocks: list, fmpath: str, window_size: int = 1,
                       feature_window: int = 1):
    """
    KIS API를 통해 종목들의 실시간 분봉 데이터를 가져와 전처리하고,
    학습된 모델에 입력할 수 있는 최종 feature 형태로 반환합니다.

    Args:
        api_handler (KISApiHandler): KIS API 통신을 위한 핸들러 객체
        stocks (list): 종목 코드 리스트
        fmpath (str): FCM, PCA 모델이 저장된 경로
        window_size (int): 모델 입력으로 사용할 시간 윈도우 크기
        feature_window (int): 피처 생성 시 사용할 윈도우 크기

    Returns:
        np.array: 모델에 입력될 최종 feature 데이터 (1, window_size, num_stocks * num_features)
        np.array: 현재 시점의 가격 데이터 (chart_data 형태)
    """
    all_stocks_data = []
    chart_datas = []

    # 기술적 지표 계산을 위해 최소 200개의 데이터 포인트를 가져옵니다. (필요에 따라 조절)
    # KIS API는 한번에 약 30개 정도의 분봉을 제공하므로 여러번 호출해야 할 수 있습니다.
    # get_minute_candles_all_today 함수를 사용하면 편리합니다.
    print("Fetching realtime minute data from KIS API...")
    for stock_code in tqdm(stocks, desc="Fetching stock data"):
        # KIS API에서 분봉 데이터 가져오기
        df = api_handler.get_minute_candles_all_today(stock_code)

        if df.empty:
            print(f"Warning: Could not fetch data for {stock_code}. Skipping.")
            continue

        # 데이터프레임 컬럼명 맞추기 및 전처리
        """df.rename(columns={
            "stck_bsop_date": "date",
            "stck_oprc": "open",
            "stck_hgpr": "high",
            "stck_lwpr": "low",
            "stck_prpr": "close",
            "cntg_vol": "volume"
        }, inplace=True)"""
        # 필요한 컬럼만 선택
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
        # 데이터 타입을 숫자로 변경
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])

        # feature.py에 데이터를 전달하기 위한 준비
        # start, end 날짜는 전체 데이터를 사용하도록 설정
        start_date = datetime.strptime(df.iloc[0]['date'], '%Y%m%d').strftime('%Y-%m-%d')
        end_date = datetime.strptime(df.iloc[-1]['date'], '%Y%m%d').strftime('%Y-%m-%d')

        # Cluster_Data를 사용하여 feature 생성 (train=False 모드로)
        feature_generator = Cluster_Data(df, start_date, end_date, window_size, fmpath, feature_window, train=False)
        processed_data = feature_generator.load_data()

        # 모델에 입력할 데이터는 마지막 window_size 만큼만 사용
        all_stocks_data.append(processed_data[-1])  # 마지막 윈도우 데이터만 추가
        chart_datas.append(df.tail(1).to_numpy())  # 현재 가격 데이터 추가

    if not all_stocks_data:
        print("Error: No data could be fetched for any stock.")
        return None, None

    # 여러 종목의 데이터를 하나로 합치기
    # (num_stocks, window_size, num_features) -> (window_size, num_stocks, num_features)
    stacked_data = np.stack(all_stocks_data, axis=1)

    # 최종 모델 입력 형태로 변환
    # (window_size, num_stocks, num_features) -> (1, window_size, num_stocks * num_features)
    final_data = stacked_data.reshape(1, stacked_data.shape[0], -1)

    # 현재 가격 데이터도 stacking
    chart_data = np.stack(chart_datas, axis=1)

    return final_data, chart_data
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲