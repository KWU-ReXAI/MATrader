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


class GRU_data(Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,*kwargs)
        self.periods = [5]

    def BIAS(self,MA, n):
        BIAS = []
        for _ in range(n): BIAS.append(0)
        for i in range(n,len(self.close)):
            BIAS.append((self.close[i] - MA[i])/MA[i] *100)
        return BIAS

    def VR(self, n):
        VR = []
        for _ in range(n): VR.append(0)
        for std in range(n,len(self.close)):
            up,down,nc= 0,0,0
            for i in range(std-n+1, std):
                if i == 0: fluctuation = 0
                else :fluctuation = self.close[i] - self.close[i-1]
                if fluctuation < 0: down += self.volume[i]
                elif fluctuation == 0: nc += self.volume[i]
                else : up += self.volume[i]
            cal_vr = (up + nc/2) / (down + nc/2 + 1e-10) * 100
            VR.append(cal_vr)
        return VR

    def load_data(self):
        model_data = self.setting()
        #model_data = self.sliding_winodw_scaler()
        return np.array(model_data)

    def get_stock_data(self):
        Data.get_stock_data(self)

    def all_normalize_data_set(self):
        self.get_stock_data()
        MACD, macdsignal, macdhist = talib.MACD(self.close, fastperiod=12, slowperiod=26, signalperiod=9) # Moving Average Convergence/Divergence
        OBV = talib.OBV(self.close, self.volume)
        obv_scaler = self.scaler_modeling('obv_scaler',OBV)
        macd_scaler = self.scaler_modeling('macd_scaler',MACD)
        self.training_data['OBV'] = obv_scaler.transform(np.array(OBV).reshape(-1,1)[self.start_index- self.window_size:self.end_index]).reshape(1,-1)[0]
        self.training_data['MACD'] = macd_scaler.transform(np.array(MACD).reshape(-1,1)[self.start_index - self.window_size:self.end_index]).reshape(1,-1)[0]
        for period in self.periods:
            MA = talib.MA(self.close, timeperiod=period) # 5 1week, 20 1month, 60 1/4, 120 1/2
            EMA = talib.EMA(self.close, timeperiod=period)
            BIAS = self.BIAS(MA, period)
            VR = self.VR(period)
            ma_scaler = self.scaler_modeling('MA'+str(period),MA)
            
            ema_scaler = self.scaler_modeling('EMA'+str(period),EMA)
            bias_scaler = self.scaler_modeling('BIAS'+str(period),BIAS)
            vr_scaler = self.scaler_modeling('VR'+str(period),VR)
            self.training_data['MA'+str(period)] = ma_scaler.transform(np.array(MA).reshape(-1,1)[self.start_index- self.window_size:self.end_index]).reshape(1,-1)[0]
            self.training_data['EMA'+str(period)] = ema_scaler.transform(np.array(EMA).reshape(-1,1)[self.start_index- self.window_size:self.end_index]).reshape(1,-1)[0]
            self.training_data['BIAS'+str(period)] = bias_scaler.transform(np.array(BIAS).reshape(-1,1)[self.start_index- self.window_size:self.end_index]).reshape(1,-1)[0]
            self.training_data['VR'+str(period)] = vr_scaler.transform(np.array(VR).reshape(-1,1)[self.start_index- self.window_size:self.end_index]).reshape(1,-1)[0]

    def window_normalize_data_set(self):
        self.get_normalize_stock_data()
        MACD, macdsignal, macdhist = talib.MACD(self.close, fastperiod=12, slowperiod=26, signalperiod=9) # Moving Average Convergence/Divergence
        OBV = talib.OBV(self.close, self.volume)
        self.training_data['OBV'] = self.winodw_scaler(OBV)
        self.training_data['MACD'] = self.winodw_scaler(MACD)
        for period in self.periods:
            MA = talib.MA(self.close, timeperiod=period) # 5 1week, 20 1month, 60 1/4, 120 1/2
            EMA = talib.EMA(self.close, timeperiod=period)
            BIAS = self.BIAS(MA, period)
            VR = self.VR(period)
            self.training_data['MA'+str(period)] = self.winodw_scaler(MA)
            self.training_data['EMA'+str(period)] = self.winodw_scaler(EMA)
            self.training_data['BIAS'+str(period)] = self.winodw_scaler(BIAS)
            self.training_data['VR'+str(period)] = self.winodw_scaler(VR)
    
    def get_normalize_stock_data(self):
        open_list = [0];close_list = [0];high_list = [0];low_list = [];volume_list = [0]
        for i in range(1, len(self.open)):
            open_list.append((self.open[i] - self.close[i-1])/self.close[i-1])
            close_list.append((self.close[i] - self.open[i])/self.open[i])
            high_list.append((self.high[i] - self.open[i])/self.open[i])
            low_list.append((self.low[i] - self.open[i])/self.open[i])
            volume_list.append((self.volume[i] - self.volume[i-1])/self.volume[i-1])
        self.training_data['open'] = self.winodw_scaler(open_list)
        self.training_data['close'] = self.winodw_scaler(close_list)
        self.training_data['high'] = self.winodw_scaler(high_list)
        self.training_data['low'] = self.winodw_scaler(low_list)
        self.training_data['volume'] =self.winodw_scaler(volume_list)

    def setting(self):
        self.window_normalize_data_set()
        #self.all_normalize_data_set()
        del self.training_data['date']
        windows_data = []
        for index in range(self.window_size,len(self.training_data)):
            data = self.training_data.iloc[index-self.window_size:index]
            windows_data.append(np.array(data))
        return windows_data
    
    def scaler_modeling(self,ta_name, ta):
        ta_scaler = StandardScaler().fit(np.array(ta).reshape(-1,1)[self.start_index- self.window_size:self.end_index])
        save_path = os.path.join(self.fmpath,ta_name+'.pkl')
        joblib.dump(ta_scaler,save_path)
        return ta_scaler

    def winodw_scaler(self, ta):
        feature_index = self.start_index - self.feature_window
        minus = self.feature_window - self.window_size
        data = np.array(ta)[feature_index:self.end_index]
        scaler_data = []; s_index = 0; e_index = 0
        while(e_index != len(data)):
            e_index = s_index+self.feature_window
            if e_index > len(data): e_index = len(data)
            var = data.reshape(-1,1)[s_index:e_index]
            np.nan_to_num(var, copy=False)
            temp = self.scaler.fit_transform(var).reshape(1,-1)[0]
            s_index += self.feature_window
            scaler_data.extend(temp)
        scaler_data = scaler_data[minus:]
        return scaler_data

    def sliding_winodw_scaler(self):
        self.sliding_data_set()
        del self.training_data['date']
        windows_data = []
        for index in range(self.window_size,len(self.training_data)):
            data = self.training_data.iloc[index - self.window_size:index]
            data_mean = data.mean(axis=0); data_std = data.std(axis=0)
            normalize = np.nan_to_num((data - data_mean) / data_std)
            windows_data.append(normalize)
        return windows_data
    
    def sliding_data_set(self):
        self.get_stock_data()
        MACD, macdsignal, macdhist = talib.MACD(self.close, fastperiod=12, slowperiod=26, signalperiod=9) # Moving Average Convergence/Divergence
        OBV = talib.OBV(self.close, self.volume)
        self.training_data['OBV'] = OBV[self.start_index-self.window_size:self.end_index]
        self.training_data['MACD'] = MACD[self.start_index-self.window_size:self.end_index]
        for period in self.periods:
            MA = talib.MA(self.close, timeperiod=period) # 5 1week, 20 1month, 60 1/4, 120 1/2
            EMA = talib.EMA(self.close, timeperiod=period)
            BIAS = self.BIAS(MA, period)
            VR = self.VR(period)
            self.training_data['MA'+str(period)] = MA[self.start_index-self.window_size:self.end_index]
            self.training_data['EMA'+str(period)] = EMA[self.start_index-self.window_size:self.end_index]
            self.training_data['BIAS'+str(period)] = BIAS[self.start_index-self.window_size:self.end_index]
            self.training_data['VR'+str(period)] = VR[self.start_index-self.window_size:self.end_index]


from sklearn.decomposition import PCA, KernelPCA
from sklearn.cluster import DBSCAN, KMeans
class Cluster_Data(Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,*kwargs)
        self.data = self.training_data.copy()
        self.n_cluster = 20
        self.pca_dim = 8

    def load_data(self):
        model_data = self.setting()
        return np.array(model_data)

    def stock(self):
        self.stock_data = self.data.copy()
        open_list = [];close_list = [];high_list = [];low_list = [];volume_list = []
        for i in range(self.start_index - self.window_size, self.end_index):
            open_list.append((self.open[i+1] - self.close[i])/self.close[i])
            close_list.append((self.close[i] - self.open[i])/self.open[i])
            high_list.append((self.high[i] - self.open[i])/self.open[i])
            low_list.append((self.low[i] - self.open[i])/self.open[i])
            volume_list.append((self.volume[i] - self.volume[i-1])/self.volume[i-1])
        self.stock_data['open'] =  self.winodw_scaler(open_list); self.stock_data['close'] =  self.winodw_scaler(close_list)
        self.stock_data['high'] =  self.winodw_scaler(high_list); self.stock_data['low'] =  self.winodw_scaler(low_list)
        #self.stock_data['volume'] =  self.winodw_scaler(volume_list)
        del self.stock_data['date']
        fcm = FCM(n_clusters=self.n_cluster); fcm.fit(np.array(self.stock_data))
        cluster_n = fcm.predict(np.array(self.stock_data))
        cluster_center = fcm.centers; d_1 = []; d_2 = []; d_3 = []; d_4 = []; d_5 = []
        for current_center in zip(cluster_n):
            d_1.append(cluster_center[current_center][0])
            d_2.append(cluster_center[current_center][1])
            d_3.append(cluster_center[current_center][2])
            d_4.append(cluster_center[current_center][3])
            #d_5.append(cluster_center[current_center][4])
        self.training_data["stock_center 1"] = d_1
        self.training_data["stock_center 2"] = d_2
        self.training_data["stock_center 3"] = d_3
        self.training_data["stock_center 4"] = d_4
        #self.training_data["stock_center 5"] = d_5

    def setting(self):
        #self.stock()   #5
        self.candle_stick()  #4
        self.overlay()  #5
        self.momentum() #5
        self.volume_indicator() #3
        self.volatility_indicator() #3
        del self.training_data['date']
        windows_data = []
        for index in range(self.window_size,len(self.training_data)):
            data = self.training_data.iloc[index-self.window_size:index]
            windows_data.append(np.array(data))
        return windows_data

    def candle_stick(self):
        self.candle_data = self.data.copy()
        upper = []; lower = []; body= []; colors= []     #upper lenght,lower length,body length,body color   
        d_u = []; d_l = []; d_b = []; states = []
        for index in range(self.start_index- self.window_size, self.end_index):
            body.append([np.abs(self.close[index] - self.open[index])])
            if self.close[index] - self.open[index] > 0: 
                upper.append([self.high[index] - self.close[index]])
                lower.append([self.open[index] - self.low[index]])
                colors.append(0.0)        #body color is red
            else : 
                upper.append([self.high[index] - self.open[index]])
                lower.append([self.close[index] - self.low[index]])
                colors.append(1.0)                                             #body color is green


        self.candle_data['upper'] = self.winodw_scaler(upper); self.candle_data['lower'] = self.winodw_scaler(lower)
        self.candle_data['body'] = self.winodw_scaler(body); self.training_data['color'] = colors
        
        del self.candle_data['date']
        fcm = FCM(n_clusters=self.n_cluster); fcm.fit(np.array(self.candle_data))
        cluster_n = fcm.predict(np.array(self.candle_data))
        cluster_center = fcm.centers; d_1 = []; d_2 = []; d_3 = []; d_4 = []
        for current_center in zip(cluster_n):
            d_1.append(cluster_center[current_center][0])
            d_2.append(cluster_center[current_center][1])
            d_3.append(cluster_center[current_center][2])
        self.training_data["candlestick_center 1"] = d_1
        self.training_data["candlestick_center 2"] = d_2
        self.training_data["candlestick_center 3"] = d_3

    def overlay(self):
        self.overlap_data = self.data.copy()
        upperband, middleband,lowerband = talib.BBANDS(self.close)
        self.overlap_data['upperband'] = self.cluster("upperband",upperband[self.start_index-self.window_size:self.end_index])
        self.overlap_data['middleband']= self.cluster("middleband",middleband[self.start_index-self.window_size:self.end_index])
        self.overlap_data['lowerband']= self.cluster("lowerband",lowerband[self.start_index-self.window_size:self.end_index])
        
        mama, fama = talib.MAMA(self.close)
        self.overlap_data['mama'] = self.cluster("mama",mama[self.start_index-self.window_size:self.end_index])
        self.overlap_data['fama'] = self.cluster("fama",fama[self.start_index-self.window_size:self.end_index])

        self.overlap_data['DEMA'] = self.cluster("DEMA",talib.DEMA(self.close)[self.start_index-self.window_size:self.end_index])
        self.overlap_data['EMA'] = self.cluster("EMA",talib.EMA(self.close)[self.start_index-self.window_size:self.end_index])
        self.overlap_data['HT_TRENDLINE'] = self.cluster("HT_TRENDLINE",talib.HT_TRENDLINE(self.close)[self.start_index-self.window_size:self.end_index])
        self.overlap_data['KAMA'] = self.cluster("KAMA",talib.KAMA(self.close)[self.start_index-self.window_size:self.end_index])
        self.overlap_data['MA'] = self.cluster("MA",talib.MA(self.close)[self.start_index-self.window_size:self.end_index])
        self.overlap_data['SMA'] = self.cluster("SMA",talib.SMA(self.close)[self.start_index-self.window_size:self.end_index])
        self.overlap_data['T3'] = self.cluster("T3",talib.T3(self.close)[self.start_index-self.window_size:self.end_index])
        self.overlap_data['TEMA'] = self.cluster("TEMA",talib.TEMA(self.close)[self.start_index-self.window_size:self.end_index])
        self.overlap_data['TRIMA'] = self.cluster("TRIMA",talib.TRIMA(self.close)[self.start_index-self.window_size:self.end_index])
        self.overlap_data['WMA'] = self.cluster("WMA",talib.WMA(self.close)[self.start_index-self.window_size:self.end_index])
        self.overlap_data['MIDPOINT'] = self.cluster("MIDPOINT",talib.MIDPOINT(self.close)[self.start_index-self.window_size:self.end_index])
        self.overlap_data['MIDPRICE'] = self.cluster("MIDPRICE",talib.MIDPRICE(self.high, self.low)[self.start_index-self.window_size:self.end_index])
        self.overlap_data['SAR'] = self.cluster("SAR",talib.SAR(self.high, self.low)[self.start_index-self.window_size:self.end_index])
        self.overlap_data['SAREXT'] = self.cluster("SAREXT",talib.SAREXT(self.high, self.low)[self.start_index-self.window_size:self.end_index])

        del self.overlap_data['date']
        pca = PCA(n_components=self.pca_dim) # None -> all feature, 2 -> 2-dimention
        pca_data = pca.fit_transform(self.overlap_data)
        
        fcm = FCM(n_clusters=self.n_cluster); fcm.fit(pca_data)
        cluster_n = fcm.predict(pca_data)

        cluster_center = fcm.centers; d_1 = []; d_2 = []; d_3 = []; d_4 = []; d_5 = []
        d_6 = []; d_7 = []; d_8 = []; d_9 = []; d_10 = []
        for current_center in zip(cluster_n):
            d_1.append(cluster_center[current_center][0]); d_2.append(cluster_center[current_center][1])
            d_3.append(cluster_center[current_center][2]); d_4.append(cluster_center[current_center][3])
            d_5.append(cluster_center[current_center][4]); d_6.append(cluster_center[current_center][5])
            d_7.append(cluster_center[current_center][6]); d_8.append(cluster_center[current_center][7])
        self.training_data["overlap_center 1"] = d_1; self.training_data["overlap_center 2"] = d_2
        self.training_data["overlap_center 3"] = d_3; self.training_data["overlap_center 4"] = d_4
        self.training_data["overlap_center 5"] = d_5; self.training_data["overlap_center 6"] = d_6
        self.training_data["overlap_center 7"] = d_7; self.training_data["overlap_center 8"] = d_8
    def volume_indicator(self):
        self.volume_data = self.data.copy()
        self.volume_data['AD'] = self.cluster("AD",talib.AD(self.high, self.low, self.close, self.volume)[self.start_index-self.window_size:self.end_index])
        self.volume_data['ADOSC'] = self.cluster("ADOSC",talib.ADOSC(self.high, self.low, self.close, self.volume)[self.start_index-self.window_size:self.end_index])
        self.volume_data['AOBVD'] = self.cluster("OBV",talib.OBV(self.close, self.volume)[self.start_index-self.window_size:self.end_index])
        del self.volume_data['date']
        
        fcm = FCM(n_clusters=self.n_cluster); fcm.fit(np.array(self.volume_data))
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
        self.volatility_data = self.data.copy()
        self.volatility_data['ATR'] = self.cluster("ATR",talib.ATR(self.high, self.low, self.close)[self.start_index-self.window_size:self.end_index])
        self.volatility_data['NATR'] = self.cluster("NATR",talib.NATR(self.high, self.low, self.close)[self.start_index-self.window_size:self.end_index])
        self.volatility_data['TRANGE'] = self.cluster("TRANGE",talib.TRANGE(self.high, self.low, self.close)[self.start_index-self.window_size:self.end_index])
        del self.volatility_data['date']
        
        fcm = FCM(n_clusters=self.n_cluster); fcm.fit(np.array(self.volatility_data))
        cluster_n = fcm.predict(np.array(self.volatility_data))
        cluster_center = fcm.centers; d_1 = []; d_2 = []; d_3 = []
        for current_center in zip(cluster_n):
            d_1.append(cluster_center[current_center][0])
            d_2.append(cluster_center[current_center][1])
            d_3.append(cluster_center[current_center][2])
        self.training_data["volatility_indicator 1"] = d_1
        self.training_data["volatility_indicator 2"] = d_2
        self.training_data["volatility_indicator 3"] = d_3

    def momentum(self):
        self.momentum_data = self.data.copy()
        fastk, fastd = talib.STOCHRSI(self.close)
        self.momentum_data['fastk_rsi'] = self.cluster("fastk_rsi",fastk[self.start_index-self.window_size:self.end_index])
        self.momentum_data['fastd_rsi'] = self.cluster("fastd_rsi",fastd[self.start_index-self.window_size:self.end_index])
        slowk, slowd = talib.STOCH(self.high, self.low, self.close)
        self.momentum_data['slowk'] = self.cluster("slowk",slowk[self.start_index-self.window_size:self.end_index]) 
        self.momentum_data['slowd'] = self.cluster("slowd",slowd[self.start_index-self.window_size:self.end_index])
        fastk, fastd = talib.STOCHF(self.high, self.low, self.close)
        self.momentum_data['fastk'] = self.cluster("fastk",fastk[self.start_index-self.window_size:self.end_index])
        self.momentum_data['fastd'] = self.cluster("fastd",fastd[self.start_index-self.window_size:self.end_index])
        aroondown, aroonup = talib.AROON(self.high, self.low)
        self.momentum_data['aroondown'] = self.cluster("aroondown",aroondown[self.start_index-self.window_size:self.end_index])
        self.momentum_data['aroonup'] = self.cluster("aroonup",aroonup[self.start_index-self.window_size:self.end_index])
        macd, macdsignal, macdhist = talib.MACD(self.close)
        macdext, macdsignal, macdhist = talib.MACDEXT(self.close)
        macdfix, macdsignal, macdhist = talib.MACDFIX(self.close)
        self.momentum_data['macd'] = self.cluster("macd",macd[self.start_index-self.window_size:self.end_index])
        self.momentum_data['macdext'] = self.cluster("macdext",macdext[self.start_index-self.window_size:self.end_index])
        self.momentum_data['macdfix'] = self.cluster("macdfix",macdfix[self.start_index-self.window_size:self.end_index])
        self.momentum_data['RSI'] = self.cluster("RSI",talib.RSI(self.close)[self.start_index-self.window_size:self.end_index])
        self.momentum_data['MOM'] = self.cluster("MOM",talib.MOM(self.close)[self.start_index-self.window_size:self.end_index])
        self.momentum_data['ROCR'] = self.cluster("ROCR",talib.ROCR(self.close)[self.start_index-self.window_size:self.end_index])
        self.momentum_data['ROCP'] = self.cluster("ROCP",talib.ROCP(self.close)[self.start_index-self.window_size:self.end_index])
        self.momentum_data['ROC'] = self.cluster("ROC",talib.ROC(self.close)[self.start_index-self.window_size:self.end_index])
        self.momentum_data['PPO'] = self.cluster("PPO",talib.PPO(self.close)[self.start_index-self.window_size:self.end_index])
        self.momentum_data['PLUS_DM'] = self.cluster("PLUS_DM",talib.PLUS_DM(self.high, self.low)[self.start_index-self.window_size:self.end_index])
        self.momentum_data['PLUS_DI'] = self.cluster("PLUS_DI",talib.PLUS_DI(self.high, self.low, self.close)[self.start_index-self.window_size:self.end_index])
        self.momentum_data['MINUS_DM'] = self.cluster("MINUS_DM",talib.MINUS_DM(self.high, self.low)[self.start_index-self.window_size:self.end_index])
        self.momentum_data['MINUS_DI'] = self.cluster("MINUS_DI",talib.MINUS_DI(self.high, self.low, self.close)[self.start_index-self.window_size:self.end_index])
        self.momentum_data['MFI'] = self.cluster("MFI",talib.MFI(self.high, self.low, self.close,self.volume)[self.start_index-self.window_size:self.end_index])
        self.momentum_data['DX'] = self.cluster("DX",talib.DX(self.high, self.low, self.close)[self.start_index-self.window_size:self.end_index])
        self.momentum_data['CCI'] = self.cluster("CCI",talib.CCI(self.high, self.low, self.close)[self.start_index-self.window_size:self.end_index])
        self.momentum_data['BOP'] = self.cluster("BOP",talib.BOP(self.open,self.high, self.low, self.close)[self.start_index-self.window_size:self.end_index])
        self.momentum_data['AROONOSC'] = self.cluster("AROONOSC",talib.AROONOSC(self.high, self.low)[self.start_index-self.window_size:self.end_index])
        self.momentum_data['ADXR'] = self.cluster("ADXR",talib.ADXR(self.high, self.low, self.close)[self.start_index-self.window_size:self.end_index])
        self.momentum_data['ADX'] = self.cluster("ADX",talib.ADX(self.high, self.low, self.close)[self.start_index-self.window_size:self.end_index])
        self.momentum_data['APO'] = self.cluster("APO",talib.APO(self.close)[self.start_index-self.window_size:self.end_index])
        self.momentum_data['CMO'] = self.cluster("CMO",talib.CMO(self.close)[self.start_index-self.window_size:self.end_index])
        self.momentum_data['TRIX'] = self.cluster("TRIX",talib.TRIX(self.close, timeperiod=30)[self.start_index-self.window_size:self.end_index])
        self.momentum_data['ULTOSC'] = self.cluster("ULTOSC",talib.ULTOSC(self.high, self.low, self.close)[self.start_index-self.window_size:self.end_index])
        self.momentum_data['WILLR'] = self.cluster("WILLR",talib.WILLR(self.high, self.low, self.close)[self.start_index-self.window_size:self.end_index])
        del self.momentum_data['date']
        pca = PCA(n_components=self.pca_dim) # None -> all feature, 2 -> 2-dimention
        pca_data = pca.fit_transform(self.momentum_data)
        
        fcm = FCM(n_clusters=self.n_cluster); fcm.fit(pca_data)
        cluster_n = fcm.predict(pca_data)

        cluster_center = fcm.centers; d_1 = []; d_2 = []; d_3 = []; d_4 = []; d_5 = []
        d_6 = []; d_7 = []; d_8 = []; d_9 = []; d_10 = []
        for current_center in zip(cluster_n):
            d_1.append(cluster_center[current_center][0]); d_2.append(cluster_center[current_center][1])
            d_3.append(cluster_center[current_center][2]); d_4.append(cluster_center[current_center][3])
            d_5.append(cluster_center[current_center][4]); d_6.append(cluster_center[current_center][5])
            d_7.append(cluster_center[current_center][6]); d_8.append(cluster_center[current_center][7])
            # d_9.append(cluster_center[current_center][8]); d_10.append(cluster_center[current_center][9])
        self.training_data["momentum_center 1"] = d_1; self.training_data["momentum_center 2"] = d_2
        self.training_data["momentum_center 3"] = d_3; self.training_data["momentum_center 4"] = d_4
        self.training_data["momentum_center 5"] = d_5; self.training_data["momentum_center 6"] = d_6
        self.training_data["momentum_center 7"] = d_7; self.training_data["momentum_center 8"] = d_8
        #self.training_data["momentum_center 9"] = d_9; self.training_data["momentum_center 10"] = d_10

    def cluster(self,name,data):
        data = self.winodw_scaler(data)
        return data
    
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

    def sliding_winodw_scaler(self,data):
        data = np.array(data); windows_data = []
        for index in range(self.window_size,len(data)):
            temp = self.scaler.fit_transform(data.reshape(-1,1)[index - self.window_size:index]).reshape(1,-1)[0]
            normalize_data = temp[self.window_size-1]
            windows_data.append(normalize_data)
        return windows_data