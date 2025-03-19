import pandas as pd

from test_feature import Cluster_Test_Data
def test_load_data(fpath, fmpath,date_from, date_to, window_size=1,feature_window=1,algorithm='TD3',pretraining=False):

    data = pd.read_csv(fpath, thousands=',', 
        converters={'date': lambda x: str(x)})
    data.columns = ['date', 'open', 'high', 'low', 'close', 'adj close', 'volume']
    
    # 기간 필터링
    data['date'] = data['date'][:].str.replace('-', '')
    data.drop(0,axis=0)
    data = data.dropna()
    feature = None
    if algorithm == 'td3': 
        feature = Cluster_Test_Data(data,date_from,date_to,window_size,fmpath,feature_window)

    chart_data = data[(data['date'] >= date_from) & (data['date'] <= date_to)]
    chart_data = chart_data.dropna()

    if algorithm == 'irdpg':
        start_index = len(data[(data['date'] < date_from)]) - window_size
        end_index = len(data[(data['date'] <= date_to)])
        training_data = data[start_index:end_index]
    else: training_data = feature.load_data()
    # 기간 필터링
    return chart_data, training_data
