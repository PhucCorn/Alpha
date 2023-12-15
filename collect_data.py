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

def get_vn30f_clone():
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

def get_vn30_clone():
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

def numpy_2(a, n):
    M = a.shape[0]
    perc = (np.arange(M-n,M)+1.0)/M*100
    return np.percentile(a,perc)

def get_vn30f():
    def vn30f():
            return requests.get("https://services.entrade.com.vn/chart-api/chart?from=1667201340&resolution=1&symbol=VN30F1M&to=9999999999").json()
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
            return requests.get("https://services.entrade.com.vn/chart-api/v2/ohlcs/index?from=1667201340&resolution=1&symbol=VN30&to=9999999999").json()
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

def transform_us_index(us_index, close_feature):
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


def data_supplier_2():
    time_enter = '13:45'
    ohlc_dict = {
        'basis': 'mean',
        'Close': 'last',
            'Volume': 'sum',}
    data = pd.read_csv(r"C:\D\BKU\Intern\trade\Alpha\data_ps_1min.csv").set_index('Date')
    data2 = pd.DataFrame()
    data2['Close'] = get_vn30f()['Close']
    data2['Volume'] = get_vn30f()['Volume']
    data2['Close_VN30'] = get_vn30()['Close']
    data2 = data2.ffill()
    data2['basis'] = data2['Close_VN30'] - data2['Close']
    dataset1 = pd.concat([data,data2]).drop_duplicates().dropna()
    dataset2 = dataset1.reset_index()
    dataset2['Date'] = pd.to_datetime(dataset2['Date'])
    dataset2 = dataset2[['Date','Close', 'Volume', 'basis']]
    dataset = dataset2.resample('1Min', on='Date', label='left').apply(ohlc_dict).dropna()
    dataset['High'] = dataset2.resample('1Min', on='Date', label='left')['Close'].max().dropna()
    dataset['Low'] = dataset2.resample('1Min', on='Date', label='left')['Close'].min().dropna()
    dataset =dataset.reset_index()
    dataset2['time'] = [str(i)[11:16] for i in dataset2['Date']]
    dataset2['Date'] = [str(i)[:10] for i in dataset2['Date']]
    basis_feature = dataset2.pivot(index = 'Date', columns = 'time', values = 'basis')
    basis_feature.columns = [str(i) for i in basis_feature.columns]
    basis_feature = basis_feature.dropna(axis =1, thresh = len(basis_feature)-30)
    basis_feature = basis_feature[basis_feature.index > '2018-01-05'].ffill(axis=1)
    close_feature = dataset2.pivot(index = 'Date', columns = 'time', values = 'Close')
    close_feature.columns = [str(i) for i in close_feature.columns]
    close_feature = close_feature.dropna(axis =1, thresh = len(close_feature)-30)
    close_feature = close_feature[close_feature.index > '2018-01-05'].ffill(axis=1)
    volume_feature = dataset2.pivot(index = 'Date', columns = 'time', values = 'Volume')
    volume_feature.columns = [str(i) for i in volume_feature.columns]
    volume_feature = volume_feature.dropna(axis =1, thresh = len(close_feature)-30)
    volume_feature = volume_feature[volume_feature.index > '2018-01-05']

    sp1 = yf.Ticker("^GSPC").history(period="max")[['Open', 'High', 'Low', 'Close', 'Volume']]
    sp = transform_us_index(sp1, close_feature)
    model_data = pd.DataFrame()
    date_list = [str(i)[:10] for i in pd.date_range('2019-01-01','2022-01-01').tolist()]
    time = time_enter
    bt_list = []
    today_pos_list = []
    sharpe_list = []
    for n_trial in range(10):
        time_name = time[0:2] + time[-2:]
        dt = pd.read_csv('https://github.com/ngohienduong/finpros/blob/main/' + str(time_name) + '_retrain.csv?raw=True').sort_values('values_0')
        num_trial = n_trial
        for a in range(len([s1 for s1 in list(dt) if s1.startswith('params')])):
                i = [s1 for s1 in list(dt) if s1.startswith('params')][a]
                exec(str(i)[7:] + '=' +str(dt.iloc[:,6:][i].iloc[num_trial]))
        time_enter = time
        minute_index = list(close_feature)
        time_exit = '14:45'
        enter_index = minute_index.index(time_enter)
        model_data = pd.DataFrame()
        model_data['basis'] = basis_feature[time_enter]
        model_data['trend_yesterday'] = close_feature['14:45'].shift(1)/close_feature['09:00'].shift(1) - 1
        model_data['trend_b4_esterday'] = close_feature['14:45'].shift(2)/close_feature['09:00'].shift(2) - 1
        model_data['trend_3d'] = close_feature['14:45'].shift(3)/close_feature['09:00'].shift(3) - 1
        model_data['trend_yesterday_afternoon'] = close_feature['14:45'].shift(1)/close_feature['13:00'].shift(1) - 1
        model_data['trend_yesterday_atc'] = close_feature['14:45'].shift(1)/close_feature['14:29'].shift(1) - 1
        model_data['trend_yesterday_cat'] = pd.cut(model_data['trend_yesterday'], labels=[-2,-1,1,2], bins = [-1,-0.02,0, 0.02,1],right=False)
        model_data['trend_b4_esterday'] = pd.cut(model_data['trend_b4_esterday'], labels=[-2,-1,1,2], bins = [-1,-0.02,0, 0.02,1],right=False)
        model_data['basis_trend']= basis_feature[time_enter] - basis_feature[minute_index[enter_index-28]]
        model_data['us_trend_day'] = sp['Close']/sp['Close'].shift(1) - 1
        model_data['us_trend_day'] = model_data['us_trend_day'].replace(0, np.nan).ffill()
        model_data['us_trend_week'] = sp['Close']/sp['Close'].shift(5) - 1
        model_data['basis_diff'] = basis_feature[time_enter] - basis_feature.mean(axis=1).shift(1)
        model_data['trend_from_close_yesterday'] = close_feature[time_enter]/close_feature['14:45'].shift(1) - 1
        model_data['trend_from_ato'] = close_feature[time_enter]/close_feature['09:00'] - 1
        model_data['trend_start'] = close_feature['09:30']/close_feature['09:15'].shift(1) - 1
        model_data['vn30f_trend_30m']= close_feature[time_enter]/close_feature[minute_index[enter_index-30]] - 1
        model_data['vn30f_trend_15m']= close_feature[time_enter]/close_feature[minute_index[enter_index-15]] - 1
        model_data['vn30f_trend_45m']= close_feature[time_enter]/close_feature[minute_index[enter_index-45]] - 1
        model_data['vn30f_trend_1h']= close_feature[time_enter]/close_feature[minute_index[enter_index-60]] - 1
        model_data['vn30f_std_1h'] = close_feature[minute_index[enter_index-28:enter_index]].std(axis = 1)
        model_data['vn30f_std'] = close_feature[minute_index[:enter_index]].std(axis = 1)
        model_data['vn30f_trend_1d'] = close_feature[time_enter]/close_feature[time_enter].shift(1) - 1
        model_data['vn30f_trend_1w'] = close_feature[time_enter]/close_feature[time_enter].shift(5) - 1
        model_data['vn30f_trend_2w'] = close_feature[time_enter]/close_feature[time_enter].shift(15) - 1
        model_data['vn30f_gap']= close_feature['09:00']/close_feature['14:45'].shift(1) - 1
        model_data['trend_max'] = close_feature[time_enter]/close_feature.iloc[:,:enter_index].max(axis = 1) - 1
        model_data['trend_min'] = close_feature[time_enter]/close_feature.iloc[:,:enter_index].min(axis = 1) - 1
        model_data['vn30f_trend_zscore'+str(zscore_range)] = model_data['trend_from_ato']/model_data['trend_from_ato'].rolling(zscore_range, closed = 'left').mean()-1
        model_data['vn30f_std_zscore'+str(zscore_range)] = model_data['vn30f_std_1h']/model_data['vn30f_std_1h'].rolling(zscore_range, closed = 'left').mean()-1
        model_data['liquid_zscore'+str(zscore_range)] = (volume_feature.iloc[:,:enter_index]*close_feature.iloc[:,:enter_index]).sum(axis=1)/(volume_feature.iloc[:,:5]*close_feature.iloc[:,:5]).sum(axis=1).rolling(zscore_range, closed = 'left').mean() - 1
        model_data['volume_zscore'+str(zscore_range)] = volume_feature.iloc[:,:enter_index].sum(axis=1)/volume_feature.iloc[:,:5].sum(axis=1).rolling(zscore_range, closed = 'left').mean() - 1
        model_data['Target'] = close_feature[time_exit]/close_feature[time_enter] - 1
        f_names1 = list(model_data.drop('Target', axis =1))
        s = np.array([f_0, f_1, f_2, f_3, f_4, f_5, f_6, f_7, f_8, f_9, f_10, f_11, f_12, f_13, f_14, f_15, f_16, f_17, f_18, f_19, f_20, f_21, f_22, f_23, f_24, f_25, f_26, f_27, f_28, f_29])
        f_names = []
        for i in np.where(s>0)[0]:
            f_names.append(f_names1[i])
        model_train = model_data.dropna()
        model_train = model_train.dropna()
        model_test = model_train.dropna()
        model = make_pipeline(StandardScaler(), Ridge())
        bt = close_feature[close_feature.index.isin(model_test.index)][[time_exit,time_enter]]
        bt['predict'] = list(model.predict(model_test[f_names]))
        #bt = bt[bt['abs_predict']>bt['abs_rolling'] ]
        bt['signal'] = (bt['predict']>0).astype(int).replace(0,-1)
        bt['gain'] = (bt[time_exit] - bt[time_enter])*bt['signal'] - 0.075
        k = -(bt['gain']>0).astype(int).sum()/len(bt['gain'])
        bt_list.append(bt)
    return bt