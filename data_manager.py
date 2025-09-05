import pandas as pd
from feature import Cluster_Data

# date_from: (train_start, test_start), date_to: (train_end, test_end)
def load_data(fpath, stocks:list, fmpath,date_from, date_to, window_size=1,feature_window = 1,algorithm='td3',pretraining=False):
    path = f'{fpath}{stocks[0]}.csv'
    data = pd.read_csv(path, thousands=',',
        converters={'date': lambda x: str(x)})
    data.columns = ['date', 'open', 'high', 'low', 'close', 'adj close', 'volume']
    # 기간 필터링
    data['date'] = data['date'][:].str.replace('-', '')
    data.replace(0, pd.NA, inplace=True)
    data.dropna(inplace=True, ignore_index=True)
    
    feature = None
    if algorithm == 'td3':
        feature = Cluster_Data(data,date_from,date_to,window_size,fmpath,feature_window)

    # 튜플 형태의 날짜 분리
    train_start, test_start = date_from
    train_end, test_end = date_to

    # chart data: train, test 각각 구함
    train_chart_data = data[(data['date'] >= str(train_start)) & (data['date'] <= str(train_end))]
    train_chart_data = train_chart_data.dropna()
    test_chart_data = data[(data['date'] >= str(test_start)) & (data['date'] <= str(test_end))]
    test_chart_data = test_chart_data.dropna()
    
    training_data, test_data = feature.load_data()
    return train_chart_data, test_chart_data, training_data, test_data

if __name__ == '__main__':
    load_data('./data/ROK/000050.csv', [], './', ('20160101', '20200701'), ('20200630', '20210630'), window_size=10)