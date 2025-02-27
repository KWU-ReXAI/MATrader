import pandas as pd
import numpy as np
# from feature import TD3_data, GRU_data, CANDLE_data,ATTENTION_data,DSL_data,Cluster_Data
from feature import GRU_data, Cluster_Data

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
    elif algorithm == 'gdpg' or algorithm =='gdqn': 
        feature = GRU_data(data,date_from,date_to,window_size,fmpath,feature_window)  
    # if algorithm == 'candle': 
    #     feature = CANDLE_data(data,date_from,date_to,window_size,fmpath,feature_window)
    # elif algorithm == 'td3': 
    #     feature = Cluster_Data(data,date_from,date_to,window_size,fmpath,feature_window)
    # elif algorithm == 'dsl': 
    #     feature = DSL_data(data,date_from,date_to,window_size,fmpath,feature_window)
    # elif algorithm == 'gdpg' or algorithm =='gdqn': 
    #     feature = GRU_data(data,date_from,date_to,window_size,fmpath,feature_window)
    # elif algorithm == 'attention':
    #     feature = ATTENTION_data(data, date_from, date_to, window_size,fmpath,feature_window)
    
    chart_data = data[(data['date'] >= date_from) & (data['date'] <= date_to)]
    chart_data = chart_data.dropna()
    if algorithm == 'irdpg':
        start_index = len(data[(data['date'] < date_from)]) - window_size
        end_index = len(data[(data['date'] <= date_to)])
        training_data = data[start_index:end_index]
    else: training_data = feature.load_data()
    # 기간 필터링
    # if algorithm == 'attention':
    #     start_index = len(data[(data['date'] < date_from)]) -1
    #     end_index = len(data[(data['date'] <= date_to)])
    #     data = data[start_index:end_index]
    #     chart_data = data.dropna()
    return chart_data, training_data

# from test_feature import GRU_test_data, TD3_test_data, Cluster_Test_Data, DSL_test_data
# def test_load_data(fpath, fmpath,date_from, date_to, window_size=1,feature_window=1,algorithm='TD3',pretraining=False):

#     data = pd.read_csv(fpath, thousands=',', 
#         converters={'date': lambda x: str(x)})
#     data.columns = ['date', 'open', 'high', 'low', 'close', 'adj close', 'volume']
    
#     # 기간 필터링
#     data['date'] = data['date'][:].str.replace('-', '')
#     data.drop(0,0)
#     data = data.dropna()
#     feature = None
#     if algorithm == 'candle': 
#         feature = CANDLE_data(data,date_from,date_to,window_size,fmpath,feature_window)
#     elif algorithm == 'td3': 
#         feature = Cluster_Test_Data(data,date_from,date_to,window_size,fmpath,feature_window)
#     elif algorithm == 'dsl': 
#         feature = DSL_test_data(data,date_from,date_to,window_size,fmpath,feature_window)
#     elif algorithm == 'gdpg' or algorithm =='gdqn': 
#         feature = GRU_test_data(data,date_from,date_to,window_size,fmpath,feature_window)
#     elif algorithm == 'attention':
#         feature = ATTENTION_data(data, date_from, date_to, window_size,fmpath,feature_window)

#     chart_data = data[(data['date'] >= date_from) & (data['date'] <= date_to)]
#     chart_data = chart_data.dropna()
#     if algorithm == 'irdpg':
#         start_index = len(data[(data['date'] < date_from)]) - window_size
#         end_index = len(data[(data['date'] <= date_to)])
#         training_data = data[start_index:end_index]
#     else: training_data = feature.load_data()
#     # 기간 필터링
#     return chart_data, training_data