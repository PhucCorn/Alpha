import pandas as pd
import requests
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import warnings
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Normalizer, RobustScaler
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import *
from sklearn.metrics import *
from copy import copy
from itertools import combinations
import heapq
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# import telegram

time_enter = '13:45'
warnings.filterwarnings("ignore")

def numpy_2(a, n):
    M = a.shape[0]
    perc = (np.arange(M-n,M)+1.0)/M*100
    return np.percentile(a,perc)

def get_vn30f():
    def vn30f():
            return requests.get("https://services.entrade.com.vn/chart-api/chart?from=1701315223&resolution=1&symbol=VN30F1M&to=9999999999").json()
    vn30fm = pd.DataFrame(vn30f()).iloc[:,:6]
    vn30fm['t'] = vn30fm['t'].astype(int).apply(lambda x: datetime.utcfromtimestamp(x) + timedelta(hours = 7))
    vn30fm.columns = ['Date','Open','High','Low','Close','Volume']
    ohlc_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',}
    vn30fm = pd.DataFrame(vn30f()).iloc[:,:6]
    vn30fm['t'] = vn30fm['t'].astype(int).apply(lambda x: datetime.utcfromtimestamp(x) + timedelta(hours = 7))
    vn30fm.columns = ['Date','Open','High','Low','Close','Volume']
    return vn30fm.set_index('Date')

def get_vn30():
    def vn30():
            return requests.get("https://services.entrade.com.vn/chart-api/v2/ohlcs/index?from=1701315223&resolution=1&symbol=VN30&to=9999999999").json()
    vn30fm = pd.DataFrame(vn30()).iloc[:,:6]
    vn30fm['t'] = vn30fm['t'].astype(int).apply(lambda x: datetime.utcfromtimestamp(x) + timedelta(hours = 7))
    vn30fm.columns = ['Date','Open','High','Low','Close','Volume']
    ohlc_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',}
    vn30fm = pd.DataFrame(vn30()).iloc[:,:6]
    vn30fm['t'] = vn30fm['t'].astype(int).apply(lambda x: datetime.utcfromtimestamp(x) + timedelta(hours = 7))
    vn30fm.columns = ['Date','Open','High','Low','Close','Volume']
    return vn30fm.set_index('Date')

def zscore(x, window):
    r = x.rolling(window=window)
    m = r.mean().shift(1)
    s = r.std(ddof=0).shift(1)
    z = (x-m)/s
    return z

def backtest_position_ps(position, price):
    a = pd.DataFrame()
    pos = pd.Series(position)
    pr = pd.Series(price)
    pos_long = np.where(pos>0, pos, 0)
    pos_short = np.where(pos<0, pos, 0)
    pnl_long = (pr.shift(-1)-pr) * pos_long
    pnl_short = (pr.shift(-1)-pr) * pos_short
    fees = abs(pos.diff(1)).cummax()*0.037
    return pnl_long + pnl_short - fees

def transform_us_index(us_index):
    return pd.DataFrame(us_index, index= close_feature.index).shift(1).ffill()


def data_supplier():
    ohlc_dict = {
        'basis': 'mean',
        'Close': 'last',
            'Volume': 'sum',}
    data2 = pd.DataFrame()
    data2['Close'] = get_vn30f()['Close']
    data2['Volume'] = get_vn30f()['Volume']
    data2['Close_VN30'] = get_vn30()['Close']
    data2 = data2.ffill()
    data2['basis'] = data2['Close_VN30'] - data2['Close']
    dataset1 = data2.drop_duplicates().dropna()
    dataset2 = dataset1.reset_index()
    dataset2['Date'] = pd.to_datetime(dataset2['Date'])
    dataset2 = dataset2[['Date','Close', 'Volume', 'basis']]
    dataset = dataset2.resample('1Min', on='Date', label='left').apply(ohlc_dict).dropna()
    dataset['High'] = dataset2.resample('1Min', on='Date', label='left')['Close'].max().dropna()
    dataset['Low'] = dataset2.resample('1Min', on='Date', label='left')['Close'].min().dropna()
    dataset =dataset.reset_index()
    dataset2['time'] = [str(i)[11:16] for i in dataset2['Date']]
    dataset2['Date'] = [str(i)[:10] for i in dataset2['Date']]
    close_feature = dataset2.pivot(index = 'Date', columns = 'time', values = 'Close')
    close_feature.columns = [str(i) for i in close_feature.columns]
    close_feature = close_feature.dropna(axis =1, thresh = len(close_feature)-30)
    close_feature = close_feature[close_feature.index > '2018-01-05'].ffill(axis=1)
    data = close_feature.stack().reset_index()
    data.columns = ['Date','time','Close']
    data['datetime'] = pd.to_datetime(data['Date'] + ' ' + data['time'])
    return data