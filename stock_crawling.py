import yfinance as yf
import pandas as pd
import os

### 코스피 다운로드 ###
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
symbol_path = os.path.join(BASE_DIR + "/data/symbol.csv")
start_date = "2009-01-01"
end_date = "2024-12-31"

# 코스피 리스트
kospi = {}
with open(symbol_path, 'r') as f:
    for line in f:
        code, name, sector = line.strip().split(',')
        kospi[code] = name
    del kospi["code"]

for code, name in kospi.items():
    data_path = os.path.join(BASE_DIR + "/data/ROK/{}.csv".format(name))
            
    # 데이터 다운로드
    data = yf.download(code+".KS", start="2015-10-01", end="2024-3-31")
    # 데이터 CSV로 저장
    data.to_csv(data_path)

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