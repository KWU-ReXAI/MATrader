import yfinance as yf
import pandas as pd
import os




BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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
