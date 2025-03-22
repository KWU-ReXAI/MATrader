import talib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from fcmeans import FCM
import numpy as np
import abc
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "0"
np.random.seed(42)
class Data:
    __metaclass__ = abc.ABCMeta
    def __init__(self,data,start_date,end_date,window_size = 1,fmpath = None, feature_window=1):
        self.open = data['open']; self.close = data['close']; self.high = data['high']
        self.low = data['low']; self.adj_close = data['adj close']; self.volume = data['volume']
        self.original_data = pd.DataFrame({"date":data['date']})

        # (train_start, test_start) = start_date
        # end_date도 마찬가지
        train_start, test_start = start_date
        train_end, test_end = end_date
        self.train_start_idx = len(self.original_data[(self.original_data['date'] < str(train_start))])
        self.test_start_idx = len(self.original_data[(self.original_data['date'] < str(test_start))])
        self.train_end_idx = len(self.original_data[(self.original_data['date'] < str(train_end))]) + 1
        self.test_end_idx = len(self.original_data[(self.original_data['date'] < str(test_end))]) + 1

        self.training_data = self.original_data.copy()
        self.training_data = self.training_data[self.train_start_idx-window_size:self.train_end_idx]
        self.test_data = self.original_data.copy()
        self.test_data = self.test_data[self.test_start_idx-window_size:self.test_end_idx]

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
        for i in range(self.train_start_idx-self.window_size, self.train_end_idx):
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
class Cluster_Data(Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,*kwargs)
        self.train_tmp = self.training_data.copy()
        self.test_tmp = self.test_data.copy()
        self.n_cluster = 20
        self.pca_dim = 8

    def load_data(self):
        train_return, test_return = self.setting()
        return np.array(train_return), np.array(test_return)

    def setting(self):
        self.candle_stick()  #4
        self.overlay()  #5
        self.momentum() #5
        self.volume_indicator() #3
        self.volatility_indicator() #3
        del self.training_data['date']
        del self.test_data['date']

        train_windows_data = []
        test_windows_data = []
        for index in range(self.window_size,len(self.training_data)):
            data = self.training_data.iloc[index-self.window_size:index]
            train_windows_data.append(np.array(data))
        for index in range(self.window_size,len(self.test_data)):
            data = self.test_data.iloc[index-self.window_size:index]
            test_windows_data.append(np.array(data))

        return train_windows_data, test_windows_data

    def candle_stick(self):
        train_candle_data = self.train_tmp.copy()
        test_candle_data = self.test_tmp.copy()

        upper = []; lower = []; body= []; colors= []     #upper lenght,lower length,body length,body color
        for index in range(self.train_start_idx- self.window_size, self.train_end_idx):
            body.append([np.abs(self.close[index] - self.open[index])])
            if self.close[index] - self.open[index] > 0: 
                upper.append([self.high[index] - self.close[index]])
                lower.append([self.open[index] - self.low[index]])
                colors.append(0.0)        #body color is red
            else : 
                upper.append([self.high[index] - self.open[index]])
                lower.append([self.close[index] - self.low[index]])
                colors.append(1.0)                                             #body color is green


        train_candle_data['upper'] = self.winodw_scaler(upper); train_candle_data['lower'] = self.winodw_scaler(lower)
        train_candle_data['body'] = self.winodw_scaler(body); self.training_data['color'] = colors
        
        del train_candle_data['date']
        fcm = FCM(n_clusters=self.n_cluster); fcm.fit(np.array(train_candle_data))

        upper = []; lower = []; body = []; colors = []
        for index in range(self.test_start_idx - self.window_size, self.test_end_idx):
            body.append([np.abs(self.close[index] - self.open[index])])
            if self.close[index] - self.open[index] > 0:
                upper.append([self.high[index] - self.close[index]])
                lower.append([self.open[index] - self.low[index]])
                colors.append(0.0)  # body color is red
            else:
                upper.append([self.high[index] - self.open[index]])
                lower.append([self.close[index] - self.low[index]])
                colors.append(1.0)  # body color is green

        test_candle_data['upper'] = self.winodw_scaler(upper); test_candle_data['lower'] = self.winodw_scaler(lower)
        test_candle_data['body'] = self.winodw_scaler(body); self.test_data['color'] = colors

        del test_candle_data['date']

        cluster_n = fcm.predict(np.array(train_candle_data))
        cluster_center = fcm.centers; d_1 = []; d_2 = []; d_3 = []
        for current_center in zip(cluster_n):
            d_1.append(cluster_center[current_center][0])
            d_2.append(cluster_center[current_center][1])
            d_3.append(cluster_center[current_center][2])
        self.training_data["candlestick_center 1"] = d_1
        self.training_data["candlestick_center 2"] = d_2
        self.training_data["candlestick_center 3"] = d_3

        cluster_n = fcm.predict(np.array(test_candle_data))
        cluster_center = fcm.centers; d_1 = []; d_2 = []; d_3 = []
        for current_center in zip(cluster_n):
            d_1.append(cluster_center[current_center][0])
            d_2.append(cluster_center[current_center][1])
            d_3.append(cluster_center[current_center][2])
        self.test_data["candlestick_center 1"] = d_1
        self.test_data["candlestick_center 2"] = d_2
        self.test_data["candlestick_center 3"] = d_3

    def overlay(self):
        train_overlap_data = self.train_tmp.copy()
        test_overlap_data = self.test_tmp.copy()

        upperband, middleband, lowerband = talib.BBANDS(self.close)
        train_overlap_data['upperband'], test_overlap_data['upperband'] = self.cluster("upperband",upperband)
        train_overlap_data['middleband'], test_overlap_data['middleband'] = self.cluster("middleband",middleband)
        train_overlap_data['lowerband'], test_overlap_data['lowerband'] = self.cluster("lowerband",lowerband)
        
        mama, fama = talib.MAMA(self.close)
        train_overlap_data['mama'], test_overlap_data['mama'] = self.cluster("mama",mama)
        train_overlap_data['fama'], test_overlap_data['fama'] = self.cluster("fama",fama)

        train_overlap_data['DEMA'], test_overlap_data['DEMA'] = self.cluster("DEMA",talib.DEMA(self.close))
        train_overlap_data['EMA'], test_overlap_data['EMA'] = self.cluster("EMA",talib.EMA(self.close))
        train_overlap_data['HT_TRENDLINE'], test_overlap_data['HT_TRENDLINE'] = self.cluster("HT_TRENDLINE",talib.HT_TRENDLINE(self.close))
        train_overlap_data['KAMA'], test_overlap_data['KAMA'] = self.cluster("KAMA",talib.KAMA(self.close))
        train_overlap_data['MA'], test_overlap_data['MA'] = self.cluster("MA",talib.MA(self.close))
        train_overlap_data['SMA'], test_overlap_data['SMA'] = self.cluster("SMA",talib.SMA(self.close))
        train_overlap_data['T3'], test_overlap_data['T3']= self.cluster("T3",talib.T3(self.close))
        train_overlap_data['TEMA'], test_overlap_data['TEMA'] = self.cluster("TEMA",talib.TEMA(self.close))
        train_overlap_data['TRIMA'], test_overlap_data['TRIMA'] = self.cluster("TRIMA",talib.TRIMA(self.close))
        train_overlap_data['WMA'], test_overlap_data['WMA'] = self.cluster("WMA",talib.WMA(self.close))
        train_overlap_data['MIDPOINT'], test_overlap_data['MIDPOINT'] = self.cluster("MIDPOINT",talib.MIDPOINT(self.close))
        train_overlap_data['MIDPRICE'], test_overlap_data['MIDPRICE'] = self.cluster("MIDPRICE",talib.MIDPRICE(self.high, self.low))
        train_overlap_data['SAR'], test_overlap_data['SAR'] = self.cluster("SAR",talib.SAR(self.high, self.low))
        train_overlap_data['SAREXT'], test_overlap_data['SAREXT'] = self.cluster("SAREXT",talib.SAREXT(self.high, self.low))

        del train_overlap_data['date']
        del test_overlap_data['date']
        pca = PCA(n_components=self.pca_dim) # None -> all feature, 2 -> 2-dimension
        pca.fit(train_overlap_data)
        train_pca_data = pca.transform(train_overlap_data)
        test_pca_data = pca.transform(test_overlap_data)
        
        fcm = FCM(n_clusters=self.n_cluster); fcm.fit(train_pca_data)

        cluster_n = fcm.predict(train_pca_data)
        cluster_center = fcm.centers; d_1 = []; d_2 = []; d_3 = []; d_4 = []; d_5 = []
        d_6 = []; d_7 = []; d_8 = []
        for current_center in zip(cluster_n):
            d_1.append(cluster_center[current_center][0]); d_2.append(cluster_center[current_center][1])
            d_3.append(cluster_center[current_center][2]); d_4.append(cluster_center[current_center][3])
            d_5.append(cluster_center[current_center][4]); d_6.append(cluster_center[current_center][5])
            d_7.append(cluster_center[current_center][6]); d_8.append(cluster_center[current_center][7])
        self.training_data["overlap_center 1"] = d_1; self.training_data["overlap_center 2"] = d_2
        self.training_data["overlap_center 3"] = d_3; self.training_data["overlap_center 4"] = d_4
        self.training_data["overlap_center 5"] = d_5; self.training_data["overlap_center 6"] = d_6
        self.training_data["overlap_center 7"] = d_7; self.training_data["overlap_center 8"] = d_8

        cluster_n = fcm.predict(test_pca_data)
        cluster_center = fcm.centers; d_1 = []; d_2 = []; d_3 = []; d_4 = []; d_5 = []
        d_6 = []; d_7 = []; d_8 = []
        for current_center in zip(cluster_n):
            d_1.append(cluster_center[current_center][0]); d_2.append(cluster_center[current_center][1])
            d_3.append(cluster_center[current_center][2]); d_4.append(cluster_center[current_center][3])
            d_5.append(cluster_center[current_center][4]); d_6.append(cluster_center[current_center][5])
            d_7.append(cluster_center[current_center][6]); d_8.append(cluster_center[current_center][7])
        self.test_data["overlap_center 1"] = d_1; self.test_data["overlap_center 2"] = d_2
        self.test_data["overlap_center 3"] = d_3; self.test_data["overlap_center 4"] = d_4
        self.test_data["overlap_center 5"] = d_5; self.test_data["overlap_center 6"] = d_6
        self.test_data["overlap_center 7"] = d_7; self.test_data["overlap_center 8"] = d_8

    def volume_indicator(self):
        train_volume_data = self.train_tmp.copy()
        test_volume_data = self.test_tmp.copy()

        train_volume_data['AD'], test_volume_data['AD'] = self.cluster("AD", talib.AD(self.high, self.low, self.close, self.volume))
        train_volume_data['ADOSC'], test_volume_data['ADOSC'] = self.cluster("ADOSC", talib.ADOSC(self.high, self.low, self.close, self.volume))
        train_volume_data['AOBVD'], test_volume_data['AOBVD'] = self.cluster("OBV", talib.OBV(self.close, self.volume))
        del train_volume_data['date']
        del test_volume_data['date']

        fcm = FCM(n_clusters=self.n_cluster); fcm.fit(np.array(train_volume_data))

        cluster_n = fcm.predict(np.array(train_volume_data))
        cluster_center = fcm.centers;d_1 = [];d_2 = [];d_3 = []
        for current_center in zip(cluster_n):
            d_1.append(cluster_center[current_center][0])
            d_2.append(cluster_center[current_center][1])
            d_3.append(cluster_center[current_center][2])
        self.training_data["volume_indicator_center 1"] = d_1
        self.training_data["volume_indicator_center 2"] = d_2
        self.training_data["volume_indicator_center 3"] = d_3

        cluster_n = fcm.predict(np.array(test_volume_data))
        cluster_center = fcm.centers; d_1 = []; d_2 = []; d_3 = []
        for current_center in zip(cluster_n):
            d_1.append(cluster_center[current_center][0])
            d_2.append(cluster_center[current_center][1])
            d_3.append(cluster_center[current_center][2])
        self.test_data["volume_indicator_center 1"] = d_1
        self.test_data["volume_indicator_center 2"] = d_2
        self.test_data["volume_indicator_center 3"] = d_3

    def volatility_indicator(self):
        train_vol_data = self.train_tmp.copy()
        test_vol_data = self.test_tmp.copy()

        train_vol_data['ATR'], test_vol_data['ATR'] = self.cluster("ATR", talib.ATR(self.high, self.low, self.close))
        train_vol_data['NATR'], test_vol_data['NATR'] = self.cluster("NATR", talib.NATR(self.high, self.low, self.close))
        train_vol_data['TRANGE'], test_vol_data['TRANGE'] = self.cluster("TRANGE", talib.TRANGE(self.high, self.low, self.close))
        del train_vol_data['date']
        del test_vol_data['date']
        
        fcm = FCM(n_clusters=self.n_cluster); fcm.fit(np.array(train_vol_data))

        cluster_n = fcm.predict(np.array(train_vol_data))
        cluster_center = fcm.centers; d_1 = []; d_2 = []; d_3 = []
        for current_center in zip(cluster_n):
            d_1.append(cluster_center[current_center][0])
            d_2.append(cluster_center[current_center][1])
            d_3.append(cluster_center[current_center][2])
        self.training_data["volatility_indicator 1"] = d_1
        self.training_data["volatility_indicator 2"] = d_2
        self.training_data["volatility_indicator 3"] = d_3

        cluster_n = fcm.predict(np.array(test_vol_data))
        cluster_center = fcm.centers; d_1 = []; d_2 = []; d_3 = []
        for current_center in zip(cluster_n):
            d_1.append(cluster_center[current_center][0])
            d_2.append(cluster_center[current_center][1])
            d_3.append(cluster_center[current_center][2])
        self.test_data["volatility_indicator 1"] = d_1
        self.test_data["volatility_indicator 2"] = d_2
        self.test_data["volatility_indicator 3"] = d_3

    def momentum(self):
        train_momentum_data = self.train_tmp.copy()
        test_momentum_data = self.test_tmp.copy()

        fastk, fastd = talib.STOCHRSI(self.close)
        train_momentum_data['fastk_rsi'], test_momentum_data['fastk_rsi'] = self.cluster("fastk_rsi", fastk)
        train_momentum_data['fastd_rsi'], test_momentum_data['fastd_rsi'] = self.cluster("fastd_rsi", fastd)

        slowk, slowd = talib.STOCH(self.high, self.low, self.close)
        train_momentum_data['slowk'], test_momentum_data['slowk'] = self.cluster("slowk", slowk)
        train_momentum_data['slowd'], test_momentum_data['slowd'] = self.cluster("slowd", slowd)

        fastk2, fastd2 = talib.STOCHF(self.high, self.low, self.close)
        train_momentum_data['fastk'], test_momentum_data['fastk'] = self.cluster("fastk", fastk2)
        train_momentum_data['fastd'], test_momentum_data['fastd'] = self.cluster("fastd", fastd2)

        aroondown, aroonup = talib.AROON(self.high, self.low)
        train_momentum_data['aroondown'], test_momentum_data['aroondown'] = self.cluster("aroondown", aroondown)
        train_momentum_data['aroonup'], test_momentum_data['aroonup'] = self.cluster("aroonup", aroonup)

        macd, macdsignal, macdhist = talib.MACD(self.close)
        macdext, macdsignal2, macdhist2 = talib.MACDEXT(self.close)
        macdfix, macdsignal3, macdhist3 = talib.MACDFIX(self.close)
        train_momentum_data['macd'], test_momentum_data['macd'] = self.cluster("macd", macd)
        train_momentum_data['macdext'], test_momentum_data['macdext'] = self.cluster("macdext", macdext)
        train_momentum_data['macdfix'], test_momentum_data['macdfix'] = self.cluster("macdfix", macdfix)

        train_momentum_data['RSI'], test_momentum_data['RSI'] = self.cluster("RSI", talib.RSI(self.close))
        train_momentum_data['MOM'], test_momentum_data['MOM'] = self.cluster("MOM", talib.MOM(self.close))
        train_momentum_data['ROCR'], test_momentum_data['ROCR'] = self.cluster("ROCR", talib.ROCR(self.close))
        train_momentum_data['ROCP'], test_momentum_data['ROCP'] = self.cluster("ROCP", talib.ROCP(self.close))
        train_momentum_data['ROC'], test_momentum_data['ROC'] = self.cluster("ROC", talib.ROC(self.close))
        train_momentum_data['PPO'], test_momentum_data['PPO'] = self.cluster("PPO", talib.PPO(self.close))

        train_momentum_data['PLUS_DM'], test_momentum_data['PLUS_DM'] = self.cluster("PLUS_DM", talib.PLUS_DM(self.high, self.low))
        train_momentum_data['PLUS_DI'], test_momentum_data['PLUS_DI'] = self.cluster("PLUS_DI", talib.PLUS_DI(self.high, self.low, self.close))
        train_momentum_data['MINUS_DM'], test_momentum_data['MINUS_DM'] = self.cluster("MINUS_DM", talib.MINUS_DM(self.high, self.low))
        train_momentum_data['MINUS_DI'], test_momentum_data['MINUS_DI'] = self.cluster("MINUS_DI", talib.MINUS_DI(self.high, self.low, self.close))

        train_momentum_data['MFI'], test_momentum_data['MFI'] = self.cluster("MFI", talib.MFI(self.high, self.low, self.close, self.volume))
        train_momentum_data['DX'], test_momentum_data['DX'] = self.cluster("DX", talib.DX(self.high, self.low, self.close))
        train_momentum_data['CCI'], test_momentum_data['CCI'] = self.cluster("CCI", talib.CCI(self.high, self.low, self.close))
        train_momentum_data['BOP'], test_momentum_data['BOP'] = self.cluster("BOP", talib.BOP(self.open, self.high, self.low, self.close))
        train_momentum_data['AROONOSC'], test_momentum_data['AROONOSC'] = self.cluster("AROONOSC", talib.AROONOSC(self.high, self.low))
        train_momentum_data['ADXR'], test_momentum_data['ADXR'] = self.cluster("ADXR", talib.ADXR(self.high, self.low, self.close))
        train_momentum_data['ADX'], test_momentum_data['ADX'] = self.cluster("ADX", talib.ADX(self.high, self.low, self.close))
        train_momentum_data['APO'], test_momentum_data['APO'] = self.cluster("APO", talib.APO(self.close))
        train_momentum_data['CMO'], test_momentum_data['CMO'] = self.cluster("CMO", talib.CMO(self.close))
        train_momentum_data['TRIX'], test_momentum_data['TRIX'] = self.cluster("TRIX", talib.TRIX(self.close, timeperiod=30))
        train_momentum_data['ULTOSC'], test_momentum_data['ULTOSC'] = self.cluster("ULTOSC", talib.ULTOSC(self.high, self.low, self.close))
        train_momentum_data['WILLR'], test_momentum_data['WILLR'] = self.cluster("WILLR", talib.WILLR(self.high, self.low, self.close))

        del train_momentum_data['date']
        del test_momentum_data['date']

        pca = PCA(n_components=self.pca_dim) # None -> all feature, 2 -> 2-dimension
        pca.fit(train_momentum_data)
        train_pca_data = pca.transform(train_momentum_data)
        test_pca_data = pca.transform(test_momentum_data)
        
        fcm = FCM(n_clusters=self.n_cluster); fcm.fit(train_pca_data)

        cluster_n = fcm.predict(train_pca_data)
        cluster_center = fcm.centers; d_1 = []; d_2 = []; d_3 = []; d_4 = []; d_5 = []
        d_6 = []; d_7 = []; d_8 = []
        for current_center in zip(cluster_n):
            d_1.append(cluster_center[current_center][0]); d_2.append(cluster_center[current_center][1])
            d_3.append(cluster_center[current_center][2]); d_4.append(cluster_center[current_center][3])
            d_5.append(cluster_center[current_center][4]); d_6.append(cluster_center[current_center][5])
            d_7.append(cluster_center[current_center][6]); d_8.append(cluster_center[current_center][7])
        self.training_data["momentum_center 1"] = d_1; self.training_data["momentum_center 2"] = d_2
        self.training_data["momentum_center 3"] = d_3; self.training_data["momentum_center 4"] = d_4
        self.training_data["momentum_center 5"] = d_5; self.training_data["momentum_center 6"] = d_6
        self.training_data["momentum_center 7"] = d_7; self.training_data["momentum_center 8"] = d_8

        cluster_n = fcm.predict(test_pca_data)
        cluster_center = fcm.centers;
        d_1 = []; d_2 = []; d_3 = []; d_4 = []; d_5 = []
        d_6 = []; d_7 = []; d_8 = []
        for current_center in zip(cluster_n):
            d_1.append(cluster_center[current_center][0]); d_2.append(cluster_center[current_center][1])
            d_3.append(cluster_center[current_center][2]); d_4.append(cluster_center[current_center][3])
            d_5.append(cluster_center[current_center][4]); d_6.append(cluster_center[current_center][5])
            d_7.append(cluster_center[current_center][6]); d_8.append(cluster_center[current_center][7])
        self.test_data["momentum_center 1"] = d_1; self.test_data["momentum_center 2"] = d_2
        self.test_data["momentum_center 3"] = d_3; self.test_data["momentum_center 4"] = d_4
        self.test_data["momentum_center 5"] = d_5; self.test_data["momentum_center 6"] = d_6
        self.test_data["momentum_center 7"] = d_7; self.test_data["momentum_center 8"] = d_8

    def cluster(self,name,data):
        train_data = data[self.train_start_idx-self.window_size:self.train_end_idx]
        train_data = self.winodw_scaler(train_data)
        test_data = data[self.test_start_idx - self.window_size:self.test_end_idx]
        test_data = self.winodw_scaler(test_data)
        return train_data, test_data
    
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