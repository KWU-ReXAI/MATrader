import pandas as pd
import numpy as np
from feature import Cluster_Data
from tqdm import tqdm

# date_from: (train_start, test_start), date_to: (train_end, test_end)
def load_data(fpath, stocks:list, fmpath,date_from, date_to, window_size=1,feature_window = 1,algorithm='td3',pretraining=False):
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

    train_chart_datas = []; test_chart_datas = []; train_datas = []; test_datas = []
    # 튜플 형태의 날짜 분리
    train_start, test_start = date_from
    train_end, test_end = date_to
    for data in tqdm(df_stocks, desc='Data preprocessing'):
        feature = Cluster_Data(data,date_from,date_to,window_size,fmpath,feature_window)
        train_data, test_data = feature.load_data()
        train_chart_data = data[(data['date'] >= str(train_start)) & (data['date'] <= str(train_end))].dropna()
        test_chart_data = data[(data['date'] >= str(test_start)) & (data['date'] <= str(test_end))].dropna()

        train_datas.append(train_data); test_datas.append(test_data)
        train_chart_datas.append(train_chart_data); test_chart_datas.append(test_chart_data)
    train_chart_data = np.stack(train_chart_datas, axis=1); test_chart_data = np.stack(test_chart_datas, axis=1)
    train_data = np.stack(train_datas, axis=2); test_data = np.stack(test_datas, axis=2)
    train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], -1)
    test_data = test_data.reshape(test_data.shape[0], test_data.shape[1], -1)
    # train/test data: (거래일 수, time window, 종목 수 * feature 수)
    # train/test chart data: (거래일 수, 종목 수, 가격 데이터 특징 수)
    return train_chart_data, test_chart_data, train_data, test_data

if __name__ == '__main__':
    load_data('./data/ROK/', ['000150', '000210', '003200'], './', ('2016-01-01', '2020-07-01'), ('2020-06-30', '2021-06-30'), window_size=10)