from pykrx import stock
from pykrx import bond
import pykrx
import pandas as pd
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
symbol_path = os.path.join(BASE_DIR + "/data/symbol.csv")

os.makedirs(os.path.join(BASE_DIR + "/data/ROK"), exist_ok=True)

start_date = '2015-01-01'
end_date = '2025-01-31'

kospi = {}

symbol_df = pd.read_csv(symbol_path, encoding='utf-8-sig', dtype={'code': str})

for row in symbol_df.itertuples():
    kospi[row.code.zfill(6)] = row.name
    
for ticker, name in kospi.items():
    data_path = os.path.join(BASE_DIR + f"/data/ROK/{name}.csv")
    
    try:
        df_price = stock.get_market_ohlcv_by_date(fromdate=start_date,
                                          todate=end_date,
                                          ticker=ticker)
        df_price.to_csv(data_path, index=True, encoding='utf-8-sig')
        time.sleep(0.3)  # 요청 사이 딜레이 주기
        print(f"{ticker}:{name}의 가격 데이터를 저장했습니다.")
    except Exception as e:
        print(f"{ticker}:{name}의 가격 데이터를 가져오는 중 오류 발생: {e}")
        
for ticker, name in kospi.items():
    data_path = os.path.join(BASE_DIR + f"/data/ROK/{name}.csv")
    
    try:
        df_price = pd.read_csv(data_path)
        adj_close = df_price["종가"]
        df_price = df_price.drop(columns="등락률")
        df_price.insert(5, "조정종가", adj_close)
        df_price.columns = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
        
        df_price.to_csv(data_path, index=False, encoding='utf-8-sig')
        print(f"{ticker}:{name}의 형식을 수정했습니다.")
    except Exception as e:
        print(f"{ticker}:{name}의 형식을 수정하는 중 오류 발생: {e}")

"""
#stock = ["PG","SYY","ABT","COO","XRAY","VRTX","APA","COP","SLB"]
#stock = ["AAPL","IBM","INTC","FB","NFLX","F","WMT","IRM","VNO","AES","DUK","ETR","FE","PPL"]
#stock = ["OMC","T","EA","CMCSA","K","SYY","WHR","MO","HSY"]
#stock = ["XOM","OXY","AMG","COF","STT","NTRS","ARNC","EXPD","GE"\
#         ,"JCI","AMD","CSCO","ORCL","APD","FCX","NEM","NUE","VMC","HST","DRE","EQR"]
stock = ["VMC","UTX","ARNC","STI","VAR","CTL"]
for row in stock:
    df = yf.download(row,start='2010-01-01')
    output_path = os.path.join(BASE_DIR,'data/USA/{}.csv'.format(row))
    df.to_csv(output_path,mode='w')
"""

"""
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

stock = ["AAPL","AMD","K","DUK","CCL","OXY"]
for row in stock:
    output_path = os.path.join(BASE_DIR,'data/USA/{}.csv'.format(row))
    data = pd.read_csv(output_path, thousands=',', 
        converters={'date': lambda x: str(x)})
    
    original_data = pd.DataFrame({"date":data['Date']})
    start_index = len(original_data[(original_data['date'] < '2019-01-01')])
    end_index = len(original_data[(original_data['date'] < '2020-12-31')])+1
    training_data = data[start_index:end_index]
    num_stock = int(10000 / training_data['Close'].iloc[0])
    balance = 10000 - (num_stock * training_data['Close'].iloc[0])
    pv = balance + (num_stock * training_data['Close'].iloc[-1]*0.998)
    print("{} - num_stock {}, pv {}".format(row,num_stock,pv))
"""
