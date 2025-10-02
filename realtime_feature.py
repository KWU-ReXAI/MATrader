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


class Data(metaclass=abc.ABCMeta):
    def __init__(self, stock, data, window_size=1, fmpath=None, feature_window=1, train=True):
        self.open = data['open']
        self.close = data['close']
        self.high = data['high']
        self.low = data['low']
        self.adj_close = data['adj close']
        self.volume = data['volume']
        self.original_data = pd.DataFrame({"date": data['date']})
        self.train = train
        self.stock = stock

        self.data = self.original_data.copy()

        self.scaler = StandardScaler()
        self.window_size = window_size
        self.feature_window = feature_window
        if feature_window < window_size: self.feature_window = window_size
        self.fmpath = fmpath

    @abc.abstractmethod
    # def load_data(self):
    def load_data(self, start, end, realtime=False):
        pass

    @abc.abstractmethod
    def setting(self):
        pass


from sklearn.decomposition import PCA


class Realtime_Data(Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tmp = self.data.copy()
        self.start_idx = None
        self.end_idx = None
        self.tmp = None
        self.n_cluster = 20
        self.pca_dim = 8

    def load_data(self, start, end):
        self.start_idx = len(self.data) - 1
        self.end_idx = len(self.data)
        self.data = self.data[self.start_idx - self.window_size + 1:self.end_idx]
        self.tmp = self.data.copy()
        data = self.setting()
        return np.array(data)

    def setting(self):
        self.candle_stick()  # 4
        self.overlay()  # 5
        self.momentum()  # 5
        self.volume_indicator()  # 3
        self.volatility_indicator()  # 3
        del self.data['date']

        windows_data = []
        for index in range(self.window_size, len(self.data) + 1):
            data = self.data.iloc[index - self.window_size:index]
            windows_data.append(np.array(data))

        return windows_data

    def candle_stick(self):
        candle_data = self.tmp.copy()

        upper = [];
        lower = [];
        body = [];
        colors = []  # upper lenght,lower length,body length,body color
        for index in range(self.start_idx - self.window_size + 1, self.end_idx):
            body.append([np.abs(self.close[index] - self.open[index])])
            if self.close[index] - self.open[index] > 0:
                upper.append([self.high[index] - self.close[index]])
                lower.append([self.open[index] - self.low[index]])
                colors.append(0.0)  # body color is red
            else:
                upper.append([self.high[index] - self.open[index]])
                lower.append([self.close[index] - self.low[index]])
                colors.append(1.0)  # body color is green

        candle_data['upper'] = self.window_scaler(upper);
        candle_data['lower'] = self.window_scaler(lower)
        candle_data['body'] = self.window_scaler(body);
        self.data['color'] = colors

        del candle_data['date']

        if self.train:
            fcm = FCM(n_clusters=self.n_cluster)
            fcm.fit(np.array(candle_data))
            joblib.dump(fcm, os.path.join(self.fmpath, f'fcm_candle_{self.stock}.joblib'))
        else:
            fcm = joblib.load(os.path.join(self.fmpath, f'fcm_candle_{self.stock}.joblib'))
        cluster_n = fcm.predict(np.array(candle_data))
        cluster_center = fcm.centers;
        d_1 = [];
        d_2 = [];
        d_3 = []
        for current_center in zip(cluster_n):
            d_1.append(cluster_center[current_center][0])
            d_2.append(cluster_center[current_center][1])
            d_3.append(cluster_center[current_center][2])
        self.data["candlestick_center 1"] = d_1
        self.data["candlestick_center 2"] = d_2
        self.data["candlestick_center 3"] = d_3

    def overlay(self):
        overlap_data = self.tmp.copy()

        upperband, middleband, lowerband = talib.BBANDS(self.close)
        overlap_data['upperband'] = self.cluster(upperband)
        overlap_data['middleband'] = self.cluster(middleband)
        overlap_data['lowerband'] = self.cluster(lowerband)

        mama, fama = talib.MAMA(self.close)
        overlap_data['mama'] = self.cluster(mama)
        overlap_data['fama'] = self.cluster(fama)

        overlap_data['DEMA'] = self.cluster(talib.DEMA(self.close))
        overlap_data['EMA'] = self.cluster(talib.EMA(self.close))
        overlap_data['HT_TRENDLINE'] = self.cluster(talib.HT_TRENDLINE(self.close))
        overlap_data['KAMA'] = self.cluster(talib.KAMA(self.close))
        overlap_data['MA'] = self.cluster(talib.MA(self.close))
        overlap_data['SMA'] = self.cluster(talib.SMA(self.close))
        overlap_data['T3'] = self.cluster(talib.T3(self.close))
        overlap_data['TEMA'] = self.cluster(talib.TEMA(self.close, timeperiod=20))
        overlap_data['TRIMA'] = self.cluster(talib.TRIMA(self.close))
        overlap_data['WMA'] = self.cluster(talib.WMA(self.close))
        overlap_data['MIDPOINT'] = self.cluster(talib.MIDPOINT(self.close))
        overlap_data['MIDPRICE'] = self.cluster(talib.MIDPRICE(self.high, self.low))
        overlap_data['SAR'] = self.cluster(talib.SAR(self.high, self.low))
        overlap_data['SAREXT'] = self.cluster(talib.SAREXT(self.high, self.low))

        del overlap_data['date']

        # PCA로 기술적 지표 차원 축소
        if self.train:
            pca = PCA(n_components=self.pca_dim)  # None -> all feature, 2 -> 2-dimension
            pca.fit(overlap_data)
            joblib.dump(pca, os.path.join(self.fmpath, f'pca_overlay_{self.stock}.joblib'))
        else:
            pca = joblib.load(os.path.join(self.fmpath, f'pca_overlay_{self.stock}.joblib'))
        pca_data = pca.transform(overlap_data)

        if self.train:
            fcm = FCM(n_clusters=self.n_cluster)
            fcm.fit(pca_data)
            joblib.dump(fcm, os.path.join(self.fmpath, f'fcm_overlay_{self.stock}.joblib'))
        else:
            fcm = joblib.load(os.path.join(self.fmpath, f'fcm_overlay_{self.stock}.joblib'))

        cluster_n = fcm.predict(pca_data)
        cluster_center = fcm.centers
        lists = [[] for _ in range(self.pca_dim)]
        for current_center in zip(cluster_n):
            for i in range(self.pca_dim):
                lists[i].append(cluster_center[current_center][i])
        for i in range(self.pca_dim):
            self.data[f"overlap_center {i + 1}"] = lists[i]

    def volume_indicator(self):
        volume_data = self.tmp.copy()

        volume_data['AD'] = self.cluster(talib.AD(self.high, self.low, self.close, self.volume))
        volume_data['ADOSC'] = self.cluster(talib.ADOSC(self.high, self.low, self.close, self.volume))
        volume_data['AOBVD'] = self.cluster(talib.OBV(self.close, self.volume))
        del volume_data['date']

        if self.train:
            fcm = FCM(n_clusters=self.n_cluster)
            fcm.fit(np.array(volume_data))
            joblib.dump(fcm, os.path.join(self.fmpath, f'fcm_volume_{self.stock}.joblib'))
        else:
            fcm = joblib.load(os.path.join(self.fmpath, f'fcm_volume_{self.stock}.joblib'))

        cluster_n = fcm.predict(np.array(volume_data))
        cluster_center = fcm.centers;
        d_1 = [];
        d_2 = [];
        d_3 = []
        for current_center in zip(cluster_n):
            d_1.append(cluster_center[current_center][0])
            d_2.append(cluster_center[current_center][1])
            d_3.append(cluster_center[current_center][2])
        self.data["volume_indicator_center 1"] = d_1
        self.data["volume_indicator_center 2"] = d_2
        self.data["volume_indicator_center 3"] = d_3

    def volatility_indicator(self):
        vol_data = self.tmp.copy()

        vol_data['ATR'] = self.cluster(talib.ATR(self.high, self.low, self.close))
        vol_data['NATR'] = self.cluster(talib.NATR(self.high, self.low, self.close))
        vol_data['TRANGE'] = self.cluster(talib.TRANGE(self.high, self.low, self.close))
        del vol_data['date']

        if self.train:
            fcm = FCM(n_clusters=self.n_cluster)
            fcm.fit(np.array(vol_data))
            joblib.dump(fcm, os.path.join(self.fmpath, f'fcm_volatility_{self.stock}.joblib'))
        else:
            fcm = joblib.load(os.path.join(self.fmpath, f'fcm_volatility_{self.stock}.joblib'))

        cluster_n = fcm.predict(np.array(vol_data))
        cluster_center = fcm.centers;
        d_1 = [];
        d_2 = [];
        d_3 = []
        for current_center in zip(cluster_n):
            d_1.append(cluster_center[current_center][0])
            d_2.append(cluster_center[current_center][1])
            d_3.append(cluster_center[current_center][2])
        self.data["volatility_indicator 1"] = d_1
        self.data["volatility_indicator 2"] = d_2
        self.data["volatility_indicator 3"] = d_3

    def momentum(self):
        momentum_data = self.tmp.copy()

        fastk, fastd = talib.STOCHRSI(self.close)
        momentum_data['fastk_rsi'] = self.cluster(fastk)
        momentum_data['fastd_rsi'] = self.cluster(fastd)

        slowk, slowd = talib.STOCH(self.high, self.low, self.close)
        momentum_data['slowk'] = self.cluster(slowk)
        momentum_data['slowd'] = self.cluster(slowd)

        fastk2, fastd2 = talib.STOCHF(self.high, self.low, self.close)
        momentum_data['fastk'] = self.cluster(fastk2)
        momentum_data['fastd'] = self.cluster(fastd2)

        aroondown, aroonup = talib.AROON(self.high, self.low)
        momentum_data['aroondown'] = self.cluster(aroondown)
        momentum_data['aroonup'] = self.cluster(aroonup)

        macd, macdsignal, macdhist = talib.MACD(self.close)
        macdext, macdsignal2, macdhist2 = talib.MACDEXT(self.close)
        macdfix, macdsignal3, macdhist3 = talib.MACDFIX(self.close)
        momentum_data['macd'] = self.cluster(macd)
        momentum_data['macdext'] = self.cluster(macdext)
        momentum_data['macdfix'] = self.cluster(macdfix)

        momentum_data['RSI'] = self.cluster(talib.RSI(self.close))
        momentum_data['MOM'] = self.cluster(talib.MOM(self.close))
        momentum_data['ROCR'] = self.cluster(talib.ROCR(self.close))
        momentum_data['ROCP'] = self.cluster(talib.ROCP(self.close))
        momentum_data['ROC'] = self.cluster(talib.ROC(self.close))
        momentum_data['PPO'] = self.cluster(talib.PPO(self.close))

        momentum_data['PLUS_DM'] = self.cluster(talib.PLUS_DM(self.high, self.low))
        momentum_data['PLUS_DI'] = self.cluster(talib.PLUS_DI(self.high, self.low, self.close))
        momentum_data['MINUS_DM'] = self.cluster(talib.MINUS_DM(self.high, self.low))
        momentum_data['MINUS_DI'] = self.cluster(talib.MINUS_DI(self.high, self.low, self.close))

        momentum_data['MFI'] = self.cluster(talib.MFI(self.high, self.low, self.close, self.volume))
        momentum_data['DX'] = self.cluster(talib.DX(self.high, self.low, self.close))
        momentum_data['CCI'] = self.cluster(talib.CCI(self.high, self.low, self.close))
        momentum_data['BOP'] = self.cluster(talib.BOP(self.open, self.high, self.low, self.close))
        momentum_data['AROONOSC'] = self.cluster(talib.AROONOSC(self.high, self.low))
        momentum_data['ADXR'] = self.cluster(talib.ADXR(self.high, self.low, self.close))
        momentum_data['ADX'] = self.cluster(talib.ADX(self.high, self.low, self.close))
        momentum_data['APO'] = self.cluster(talib.APO(self.close))
        momentum_data['CMO'] = self.cluster(talib.CMO(self.close))
        momentum_data['TRIX'] = self.cluster(talib.TRIX(self.close, timeperiod=20))
        momentum_data['ULTOSC'] = self.cluster(talib.ULTOSC(self.high, self.low, self.close))
        momentum_data['WILLR'] = self.cluster(talib.WILLR(self.high, self.low, self.close))

        del momentum_data['date']

        # PCA로 기술적 지표 차원 축소
        if self.train:
            pca = PCA(n_components=self.pca_dim)  # None -> all feature, 2 -> 2-dimension
            pca.fit(momentum_data)
            joblib.dump(pca, os.path.join(self.fmpath, f'pca_momentum_{self.stock}.joblib'))
        else:
            pca = joblib.load(os.path.join(self.fmpath, f'pca_momentum_{self.stock}.joblib'))
        pca_data = pca.transform(momentum_data)

        if self.train:
            fcm = FCM(n_clusters=self.n_cluster)
            fcm.fit(pca_data)
            joblib.dump(fcm, os.path.join(self.fmpath, f'fcm_momentum_{self.stock}.joblib'))
        else:
            fcm = joblib.load(os.path.join(self.fmpath, f'fcm_momentum_{self.stock}.joblib'))

        cluster_n = fcm.predict(pca_data)
        cluster_center = fcm.centers
        lists = [[] for _ in range(self.pca_dim)]
        for current_center in zip(cluster_n):
            for i in range(self.pca_dim):
                lists[i].append(cluster_center[current_center][i])
        for i in range(self.pca_dim):
            self.data[f"momentum_center {i + 1}"] = lists[i]

    def cluster(self, data):
        data = data[self.start_idx - self.window_size + 1:self.end_idx]
        data = self.window_scaler(data)
        return data

    def window_scaler(self, data):
        data = np.array(data)
        scaler_data = [];
        s_index = 0;
        e_index = 0
        while (e_index != len(data)):
            e_index = s_index + self.feature_window
            if e_index > len(data): e_index = len(data)
            temp = self.scaler.fit_transform(data.reshape(-1, 1)[s_index:e_index]).reshape(1, -1)[0]
            s_index += self.feature_window
            scaler_data.extend(temp)
        return scaler_data
