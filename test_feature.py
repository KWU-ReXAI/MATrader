import talib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import pickle
from fcmeans import FCM
import numpy as np
import abc
import os
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "0"
np.random.seed(42)
class Data:
    __metaclass__ = abc.ABCMeta
    def __init__(self,data,start_date,end_date,window_size = 1,fmpath = None, feature_window=1):
        self.open = data['open']; self.close = data['close']; self.high = data['high']
        self.low = data['low']; self.adj_close = data['adj close']; self.volume = data['volume']
        self.original_data = pd.DataFrame({"date":data['date']})
        self.start_index = len(self.original_data[(self.original_data['date'] < start_date)])
        self.end_index = len(self.original_data[(self.original_data['date'] < end_date)]) + 1
        self.training_data = self.original_data.copy()
        self.training_data = self.training_data[self.start_index-window_size:self.end_index]
        self.start_date = start_date; self.end_date = end_date
        self.scaler = StandardScaler()
        self.window_size = window_size
        self.feature_window = feature_window
        if feature_window < window_size : self.feature_window = window_size
        self.fmpath = fmpath
    @abc.abstractclassmethod
    def load_data(self):
        pass
    @abc.abstractclassmethod
    def setting(self):
        pass

    def get_stock_data(self):
        open_list = [];close_list = [];high_list = [];low_list = [];volume_list = []
        for i in range(self.start_index-self.window_size, self.end_index):
            open_list.append((self.open[i] - self.close[i-1])/self.close[i-1])
            close_list.append((self.close[i] - self.open[i])/self.open[i])
            high_list.append((self.high[i] - self.open[i])/self.open[i])
            low_list.append((self.low[i] - self.open[i])/self.open[i])
            volume_list.append((self.volume[i] - self.volume[i-1])/self.volume[i-1])

        """
        for i in range(self.start_index, self.end_index):
            open_list.append((self.open[i] - self.close[i-1])/self.close[i-1])
            close_list.append((self.close[i] - self.close[i-1])/self.close[i-1])
            high_list.append((self.high[i] - self.close[i-1])/self.close[i-1])
            low_list.append((self.low[i] - self.close[i-1])/self.close[i-1])
            volume_list.append((self.volume[i] - self.volume[i-1])/self.volume[i-1])
        """
        self.training_data['open'] = np.array(open_list)
        self.training_data['close'] = np.array(close_list)
        self.training_data['high'] = np.array(high_list)
        self.training_data['low'] = np.array(low_list)
        self.training_data['volume'] =np.array(volume_list)

    def denormalize_stock_data(self):
        self.training_data['open'] = self.open; self.training_data['close'] = self.close
        self.training_data['high'] = self.high; self.training_data['low'] = self.low
        self.training_data['volume'] = self.volume
    

from sklearn.decomposition import PCA, KernelPCA
from sklearn.cluster import DBSCAN, KMeans
class Cluster_Test_Data(Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,*kwargs)
        self.train_start_index = len(self.original_data[(self.original_data['date'] < "20140101")]) 
        self.train_end_index = len(self.original_data[(self.original_data['date'] < "20181231")]) + 1
        self.cluster_train_data = self.original_data.copy()
        self.cluster_train_data = self.cluster_train_data[self.train_start_index- self.window_size:self.train_end_index]
        self.data = self.training_data.copy()
        self.n_cluster = 20
        self.pca_dim = 8
    def load_data(self):
        model_data = self.setting()
        return np.array(model_data)

    def setting(self):
        #self.stock()   #5
        self.candle_stick()  #4
        self.overlap()  #5
        self.momentum() #5
        self.volume_indicator() #3
        self.volatility_indicator() #3
        del self.training_data['date']
        windows_data = []
        for index in range(self.window_size,len(self.training_data)):
            data = self.training_data.iloc[index-self.window_size:index]
            windows_data.append(np.array(data))
        return windows_data

    def stock(self):
        self.train_stock_data = self.cluster_train_data.copy()
        self.stock_data = self.data.copy()
        open_list = [];close_list = [];high_list = [];low_list = [];volume_list = []
        for i in range(self.train_start_index - self.window_size, self.train_end_index):
            open_list.append((self.open[i+1] - self.close[i])/self.close[i])
            close_list.append((self.close[i] - self.open[i])/self.open[i])
            high_list.append((self.high[i] - self.open[i])/self.open[i])
            low_list.append((self.low[i] - self.open[i])/self.open[i])
            volume_list.append((self.volume[i] - self.volume[i-1])/self.volume[i-1])
        self.train_stock_data['open'] =  self.winodw_scaler(open_list); self.train_stock_data['close'] =  self.winodw_scaler(close_list)
        #self.train_stock_data['high'] =  self.winodw_scaler(high_list); self.train_stock_data['low'] =  self.winodw_scaler(low_list)
        #self.train_stock_data['volume'] =  self.winodw_scaler(volume_list)
        del self.train_stock_data['date']
        fcm = FCM(n_clusters=self.n_cluster); fcm.fit(np.array(self.train_stock_data))

        open_list = [];close_list = [];high_list = [];low_list = [];volume_list = []
        for i in range(self.start_index-(self.window_size *2), self.end_index):
            open_list.append((self.open[i+1] - self.close[i])/self.close[i])
            close_list.append((self.close[i] - self.open[i])/self.open[i])
            high_list.append((self.high[i] - self.open[i])/self.open[i])
            low_list.append((self.low[i] - self.open[i])/self.open[i])
            volume_list.append((self.volume[i] - self.volume[i-1])/self.volume[i-1])
        self.stock_data['open'] =  self.sliding_winodw_scaler(open_list); self.stock_data['close'] =  self.sliding_winodw_scaler(close_list)
        #self.stock_data['high'] =  self.sliding_winodw_scaler(high_list); self.stock_data['low'] =  self.sliding_winodw_scaler(low_list)
        #self.stock_data['volume'] =  self.sliding_winodw_scaler(volume_list)
        del self.stock_data['date']
        
        cluster_n = fcm.predict(np.array(self.stock_data))
        cluster_center = fcm.centers; d_1 = []; d_2 = []; d_3 = []; d_4 = []; d_5 = []
        for current_center in zip(cluster_n):
            d_1.append(cluster_center[current_center][0])
            #d_2.append(cluster_center[current_center][1])
            #d_3.append(cluster_center[current_center][2])
            #d_4.append(cluster_center[current_center][3])
            #d_5.append(cluster_center[current_center][4])
        self.training_data["stock_center 1"] = d_1
        #self.training_data["stock_center 2"] = d_2
        #self.training_data["stock_center 3"] = d_3
        #self.training_data["stock_center 4"] = d_4
        #self.training_data["stock_center 5"] = d_5
    
    def candle_stick(self):
        self.train_candle_data = self.cluster_train_data.copy()
        self.candle_data = self.data.copy()
        upper = []; lower = []; body= []; colors= []     #upper lenght,lower length,body length,body color 
        for index in range(self.train_start_index- self.window_size, self.train_end_index):
            body.append([np.abs(self.close[index] - self.open[index])])
            if self.close[index] - self.open[index] > 0: 
                upper.append([self.high[index] - self.close[index]])
                lower.append([self.open[index] - self.low[index]])
            else : 
                upper.append([self.high[index] - self.open[index]])
                lower.append([self.close[index] - self.low[index]])
        normal_upper = self.winodw_scaler(upper); normal_lower = self.winodw_scaler(lower); normal_body = self.winodw_scaler(body)
        self.train_candle_data['upper'] = normal_upper; self.train_candle_data['lower'] = normal_lower; self.train_candle_data['body'] = normal_body
        del self.train_candle_data['date']
        
        fcm = FCM(n_clusters=self.n_cluster); fcm.fit(np.array(self.train_candle_data))
        upper = []; lower = []; body= []; colors= []     #upper lenght,lower length,body length,body color 
        for index in range(self.start_index-(self.window_size *2), self.end_index):
            body.append([np.abs(self.close[index] - self.open[index])])
            if self.close[index] - self.open[index] > 0: 
                upper.append([self.high[index] - self.close[index]])
                lower.append([self.open[index] - self.low[index]])
                colors.append(0.0)        #body color is red
            else : 
                upper.append([self.high[index] - self.open[index]])
                lower.append([self.close[index] - self.low[index]])
                colors.append(1.0)                                             #body color is green
        normal_upper = self.sliding_winodw_scaler(upper); normal_lower = self.sliding_winodw_scaler(lower); normal_body = self.sliding_winodw_scaler(body)
        self.candle_data['upper'] = normal_upper; self.candle_data['lower'] = normal_lower; self.candle_data['body'] = normal_body
        self.training_data['color'] = colors[self.window_size:]
        del self.candle_data['date']
        cluster_n = fcm.predict(np.array(self.candle_data))
        cluster_center = fcm.centers; d_1 = []; d_2 = []; d_3 = []; d_4 = []
        for current_center in zip(cluster_n):
            d_1.append(cluster_center[current_center][0])
            d_2.append(cluster_center[current_center][1])
            d_3.append(cluster_center[current_center][2])
        self.training_data["candlestick_center 1"] = d_1
        self.training_data["candlestick_center 2"] = d_2
        self.training_data["candlestick_center 3"] = d_3
    
    def overlap(self):
        self.train_overlap_data = self.cluster_train_data.copy()
        self.overlap_data = self.data.copy()
        upperband, middleband,lowerband = talib.BBANDS(self.close)
        self.train_overlap_data['upperband'], self.overlap_data['upperband'] = self.cluster("upperband",upperband)
        self.train_overlap_data['middleband'], self.overlap_data['middleband']= self.cluster("middleband",middleband)
        self.train_overlap_data['lowerband'], self.overlap_data['lowerband']= self.cluster("lowerband",lowerband)
        
        mama, fama = talib.MAMA(self.close)
        self.train_overlap_data['mama'],self.overlap_data['mama'] = self.cluster("mama",mama)
        self.train_overlap_data['fama'],self.overlap_data['fama'] = self.cluster("fama",fama)

        self.train_overlap_data['DEMA'],self.overlap_data['DEMA'] = self.cluster("DEMA",talib.DEMA(self.close))
        self.train_overlap_data['EMA'],self.overlap_data['EMA'] = self.cluster("EMA",talib.EMA(self.close))
        self.train_overlap_data['HT_TRENDLINE'],self.overlap_data['HT_TRENDLINE'] = self.cluster("HT_TRENDLINE",talib.HT_TRENDLINE(self.close))
        self.train_overlap_data['KAMA'],self.overlap_data['KAMA'] = self.cluster("KAMA",talib.KAMA(self.close))
        self.train_overlap_data['MA'],self.overlap_data['MA'] = self.cluster("MA",talib.MA(self.close))
        self.train_overlap_data['SMA'],self.overlap_data['SMA'] = self.cluster("SMA",talib.SMA(self.close))
        self.train_overlap_data['T3'],self.overlap_data['T3'] = self.cluster("T3",talib.T3(self.close))
        self.train_overlap_data['TEMA'],self.overlap_data['TEMA'] = self.cluster("TEMA",talib.TEMA(self.close))
        self.train_overlap_data['TRIMA'],self.overlap_data['TRIMA'] = self.cluster("TRIMA",talib.TRIMA(self.close))
        self.train_overlap_data['WMA'],self.overlap_data['WMA'] = self.cluster("WMA",talib.WMA(self.close))
        self.train_overlap_data['MIDPOINT'],self.overlap_data['MIDPOINT'] = self.cluster("MIDPOINT",talib.MIDPOINT(self.close))
        self.train_overlap_data['MIDPRICE'],self.overlap_data['MIDPRICE'] = self.cluster("MIDPRICE",talib.MIDPRICE(self.high, self.low))
        self.train_overlap_data['SAR'],self.overlap_data['SAR'] = self.cluster("SAR",talib.SAR(self.high, self.low))
        self.train_overlap_data['SAREXT'],self.overlap_data['SAREXT'] = self.cluster("SAREXT",talib.SAREXT(self.high, self.low))

        
        del self.train_overlap_data['date']
        del self.overlap_data['date']
        pca = PCA(n_components=self.pca_dim) # None -> all feature, 2 -> 2-dimention
        pca.fit(self.train_overlap_data)
        train_pca_data = pca.transform(self.train_overlap_data)
        pca_data = pca.transform(self.overlap_data)

        fcm = FCM(n_clusters=self.n_cluster)
        fcm.fit(train_pca_data)
        cluster_n = fcm.predict(pca_data)

        cluster_center = fcm.centers; d_1 = []; d_2 = []; d_3 = []; d_4 = []; d_5 = []
        d_6 = []; d_7 = []; d_8 = []; d_9 = []; d_10 = []
        for current_center in zip(cluster_n):
            d_1.append(cluster_center[current_center][0]); d_2.append(cluster_center[current_center][1])
            d_3.append(cluster_center[current_center][2]); d_4.append(cluster_center[current_center][3])
            d_5.append(cluster_center[current_center][4]); d_6.append(cluster_center[current_center][5])
            d_7.append(cluster_center[current_center][6]); d_8.append(cluster_center[current_center][7])
            #d_9.append(cluster_center[current_center][8]); d_10.append(cluster_center[current_center][9])
        self.training_data["overlap_center 1"] = d_1; self.training_data["overlap_center 2"] = d_2
        self.training_data["overlap_center 3"] = d_3; self.training_data["overlap_center 4"] = d_4
        self.training_data["overlap_center 5"] = d_5; self.training_data["overlap_center 6"] = d_6
        self.training_data["overlap_center 7"] = d_7; self.training_data["overlap_center 8"] = d_8
        #self.training_data["overlap_center 9"] = d_9; self.training_data["overlap_center 10"] = d_10

    def volume_indicator(self):
        self.train_volume_data = self.cluster_train_data.copy()
        self.volume_data = self.data.copy()

        self.train_volume_data['AD'], self.volume_data['AD'] = self.cluster("AD",talib.AD(self.high, self.low, self.close, self.volume))
        self.train_volume_data['ADOSC'], self.volume_data['ADOSC'] = self.cluster("ADOSC",talib.ADOSC(self.high, self.low, self.close, self.volume))
        self.train_volume_data['AOBVD'], self.volume_data['AOBVD'] = self.cluster("OBV",talib.OBV(self.close, self.volume))
        del self.train_volume_data['date']
        del self.volume_data['date']
        
        fcm = FCM(n_clusters=self.n_cluster); fcm.fit(np.array(self.train_volume_data))
        cluster_n = fcm.predict(np.array(self.volume_data))
        cluster_center = fcm.centers; d_1 = []; d_2 = []; d_3 = []
        for current_center in zip(cluster_n):
            d_1.append(cluster_center[current_center][0])
            d_2.append(cluster_center[current_center][1])
            d_3.append(cluster_center[current_center][2])
        self.training_data["volume_indicator_center 1"] = d_1
        self.training_data["volume_indicator_center 2"] = d_2
        self.training_data["volume_indicator_center 3"] = d_3
    
    def volatility_indicator(self):
        self.train_volatility_data = self.cluster_train_data.copy()
        self.volatility_data = self.data.copy()
        self.train_volatility_data['ATR'], self.volatility_data['ATR'] = self.cluster("ATR",talib.ATR(self.high, self.low, self.close))
        self.train_volatility_data['NATR'], self.volatility_data['NATR'] = self.cluster("NATR",talib.NATR(self.high, self.low, self.close))
        self.train_volatility_data['TRANGE'], self.volatility_data['TRANGE'] = self.cluster("TRANGE",talib.TRANGE(self.high, self.low, self.close))
        del self.train_volatility_data['date']
        del self.volatility_data['date']
        
        fcm = FCM(n_clusters=self.n_cluster); fcm.fit(np.array(self.train_volatility_data))
        cluster_n = fcm.predict(np.array(self.volatility_data))
        cluster_center = fcm.centers; d_1 = []; d_2 = []; d_3 = []
        for current_center in zip(cluster_n):
            d_1.append(cluster_center[current_center][0])
            d_2.append(cluster_center[current_center][1])
            d_3.append(cluster_center[current_center][2])
        self.training_data["volatility_indicator_center 1"] = d_1
        self.training_data["volatility_indicator_center 2"] = d_2
        self.training_data["volatility_indicator_center 3"] = d_3

    def momentum(self):
        self.train_momentum_data = self.cluster_train_data.copy()
        self.momentum_data = self.data.copy()
        fastk, fastd = talib.STOCHRSI(self.close)
        self.train_momentum_data['fastk_rsi'], self.momentum_data['fastk_rsi'] = self.cluster("fastk_rsi",fastk)
        self.train_momentum_data['fastd_rsi'], self.momentum_data['fastd_rsi']= self.cluster("fastd_rsi",fastd)

        slowk, slowd = talib.STOCH(self.high, self.low, self.close)
        self.train_momentum_data['slowk'],self.momentum_data['slowk'] = self.cluster("slowk",slowk) 
        self.train_momentum_data['slowd'],self.momentum_data['slowd'] = self.cluster("slowd",slowd)

        fastk, fastd = talib.STOCHF(self.high, self.low, self.close)
        self.train_momentum_data['fastk'],self.momentum_data['fastk'] = self.cluster("fastk",fastk)
        self.train_momentum_data['fastd'],self.momentum_data['fastd'] = self.cluster("fastd",fastd)


        aroondown, aroonup = talib.AROON(self.high, self.low)
        self.train_momentum_data['aroondown'],self.momentum_data['aroondown'] = self.cluster("aroondown",aroondown)
        self.train_momentum_data['aroonup'],self.momentum_data['aroonup'] = self.cluster("aroonup",aroonup)

        macd, macdsignal, macdhist = talib.MACD(self.close)
        macdext, macdsignal, macdhist = talib.MACDEXT(self.close)
        macdfix, macdsignal, macdhist = talib.MACDFIX(self.close)
        self.train_momentum_data['macd'],self.momentum_data['macd'] = self.cluster("macd",macd)
        self.train_momentum_data['macdext'],self.momentum_data['macdext'] = self.cluster("macdext",macdext)
        self.train_momentum_data['macdfix'],self.momentum_data['macdfix'] = self.cluster("macdfix",macdfix)
        self.train_momentum_data['RSI'],self.momentum_data['RSI'] = self.cluster("RSI",talib.RSI(self.close))
        self.train_momentum_data['MOM'],self.momentum_data['MOM'] = self.cluster("MOM",talib.MOM(self.close))
        self.train_momentum_data['ROCR'],self.momentum_data['ROCR'] = self.cluster("ROCR",talib.ROCR(self.close))
        self.train_momentum_data['ROCP'],self.momentum_data['ROCP'] = self.cluster("ROCP",talib.ROCP(self.close))
        self.train_momentum_data['ROC'],self.momentum_data['ROC'] = self.cluster("ROC",talib.ROC(self.close))
        self.train_momentum_data['PPO'],self.momentum_data['PPO'] = self.cluster("PPO",talib.PPO(self.close))
        self.train_momentum_data['PLUS_DM'],self.momentum_data['PLUS_DM'] = self.cluster("PLUS_DM",talib.PLUS_DM(self.high, self.low))
        self.train_momentum_data['PLUS_DI'],self.momentum_data['PLUS_DI'] = self.cluster("PLUS_DI",talib.PLUS_DI(self.high, self.low, self.close))
        self.train_momentum_data['MINUS_DM'],self.momentum_data['MINUS_DM'] = self.cluster("MINUS_DM",talib.MINUS_DM(self.high, self.low))
        self.train_momentum_data['MINUS_DI'],self.momentum_data['MINUS_DI'] = self.cluster("MINUS_DI",talib.MINUS_DI(self.high, self.low, self.close))
        self.train_momentum_data['MFI'],self.momentum_data['MFI'] = self.cluster("MFI",talib.MFI(self.high, self.low, self.close,self.volume))
        self.train_momentum_data['DX'],self.momentum_data['DX'] = self.cluster("DX",talib.DX(self.high, self.low, self.close))
        self.train_momentum_data['CCI'],self.momentum_data['CCI'] = self.cluster("CCI",talib.CCI(self.high, self.low, self.close))
        self.train_momentum_data['BOP'],self.momentum_data['BOP'] = self.cluster("BOP",talib.BOP(self.open,self.high, self.low, self.close))
        self.train_momentum_data['AROONOSC'],self.momentum_data['AROONOSC'] = self.cluster("AROONOSC",talib.AROONOSC(self.high, self.low))
        self.train_momentum_data['ADXR'],self.momentum_data['ADXR'] = self.cluster("ADXR",talib.ADXR(self.high, self.low, self.close))
        self.train_momentum_data['ADX'],self.momentum_data['ADX'] = self.cluster("ADX",talib.ADX(self.high, self.low, self.close))
        self.train_momentum_data['APO'],self.momentum_data['APO'] = self.cluster("APO",talib.APO(self.close))
        self.train_momentum_data['CMO'],self.momentum_data['CMO'] = self.cluster("CMO",talib.CMO(self.close))
        self.train_momentum_data['TRIX'],self.momentum_data['TRIX'] = self.cluster("TRIX",talib.TRIX(self.close, timeperiod=30))
        self.train_momentum_data['ULTOSC'],self.momentum_data['ULTOSC'] = self.cluster("ULTOSC",talib.ULTOSC(self.high, self.low, self.close))
        self.train_momentum_data['WILLR'],self.momentum_data['WILLR'] = self.cluster("WILLR",talib.WILLR(self.high, self.low, self.close))
        
        del self.train_momentum_data['date']
        del self.momentum_data['date']
        pca = PCA(n_components=self.pca_dim) # None -> all feature, 2 -> 2-dimention
        pca.fit(self.train_momentum_data)
        train_pca_data = pca.transform(self.train_momentum_data)
        pca_data = pca.transform(self.momentum_data)

        fcm = FCM(n_clusters=self.n_cluster)
        fcm.fit(train_pca_data)
        cluster_n = fcm.predict(pca_data)

        cluster_center = fcm.centers; d_1 = []; d_2 = []; d_3 = []; d_4 = []; d_5 = []
        d_6 = []; d_7 = []; d_8 = []; d_9 = []; d_10 = []
        for current_center in zip(cluster_n):
            d_1.append(cluster_center[current_center][0]); d_2.append(cluster_center[current_center][1])
            d_3.append(cluster_center[current_center][2]); d_4.append(cluster_center[current_center][3])
            d_5.append(cluster_center[current_center][4]); d_6.append(cluster_center[current_center][5])
            d_7.append(cluster_center[current_center][6]); d_8.append(cluster_center[current_center][7])
            #d_9.append(cluster_center[current_center][8]); d_10.append(cluster_center[current_center][9])
        self.training_data["momentum_center 1"] = d_1; self.training_data["momentum_center 2"] = d_2
        self.training_data["momentum_center 3"] = d_3; self.training_data["momentum_center 4"] = d_4
        self.training_data["momentum_center 5"] = d_5; self.training_data["momentum_center 6"] = d_6
        self.training_data["momentum_center 7"] = d_7; self.training_data["momentum_center 8"] = d_8
        #self.training_data["momentum_center 9"] = d_9; self.training_data["momentum_center 10"] = d_10

    def cluster(self,name,data):
        train_data = data[self.train_start_index-self.window_size:self.train_end_index]
        train_data = self.winodw_scaler(train_data)
        exec_data = data[self.start_index-(self.window_size *2):self.end_index]
        exec_data = self.sliding_winodw_scaler(exec_data)

        return train_data, exec_data
    
    def sliding_winodw_scaler(self,data):
        data = np.array(data); windows_data = []
        for index in range(self.window_size,len(data)):
            temp = self.scaler.fit_transform(data.reshape(-1,1)[index - self.window_size:index]).reshape(1,-1)[0]
            normalize_data = temp[self.window_size-1]
            windows_data.append(normalize_data)
        return windows_data
    
    def winodw_scaler(self, data):
        data = np.array(data)
        scaler_data = []; s_index = 0; e_index = 0
        while(e_index != len(data)):
            e_index = s_index+self.feature_window
            if e_index > len(data): e_index = len(data)
            temp = self.scaler.fit_transform(data.reshape(-1,1)[s_index:e_index]).reshape(1,-1)[0]
            s_index += self.feature_window
            scaler_data.extend(temp)
        return scaler_data