import pandas as pd
from feature import Cluster_Data

# date_from: (train_start, test_start), date_to: (train_end, test_end)
def load_data(fpath, fmpath,date_from, date_to, window_size=1,feature_window = 1,algorithm='TD3',pretraining=False):
    
    data = pd.read_csv(fpath, thousands=',', 
        converters={'date': lambda x: str(x)})
    data.columns = ['date', 'open', 'high', 'low', 'close', 'adj close', 'volume']
    # 기간 필터링
    data['date'] = data['date'][:].str.replace('-', '')
    data.drop(0,axis=0)
    data = data.dropna()
    
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
