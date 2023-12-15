# importing all required libraries
import telebot
from telethon.sync import TelegramClient
from telethon.tl import types
from telethon.tl.types import InputPeerUser, InputPeerChannel
from telethon import TelegramClient, sync, events
import telegram
import requests
import pandas as pd
from datetime import datetime, timedelta, time
import matplotlib.pyplot as plt
from collect_data import *
import time
from imgurpython import ImgurClient
import os
import pickle


def is_outside_trading_hours():
    trading_hours_1_start = datetime(datetime.now().year, datetime.now().month, datetime.now().day, 9, 0)
    trading_hours_1_end = datetime(datetime.now().year, datetime.now().month, datetime.now().day, 11, 30)

    trading_hours_2_start = datetime(datetime.now().year, datetime.now().month, datetime.now().day, 13, 0)
    trading_hours_2_end = datetime(datetime.now().year, datetime.now().month, datetime.now().day, 15, 47)

    trading_hours_3 = datetime(datetime.now().year, datetime.now().month, datetime.now().day, 14, 46)
    if (
        (datetime.now() < trading_hours_1_start or datetime.now() > trading_hours_1_end)
        and (datetime.now() < trading_hours_2_start or datetime.now() > trading_hours_2_end)
    ):
    # if (
    #     (datetime.now()-timedelta(minutes=180) < trading_hours_1_start or datetime.now()-timedelta(minutes=180) > trading_hours_1_end)
    #     and (datetime.now()-timedelta(minutes=180) < trading_hours_2_start or datetime.now()-timedelta(minutes=180) > trading_hours_2_end)
    #     and datetime.now()-timedelta(minutes=180) != trading_hours_3
    # ):
        return True
    return False

def alpha_run(front, condition, back, data):
    data['ret_'+str(front)] = data['Close'] - data['Close'].shift(int(front))
    data['bool'] = (data['ret_'+str(front)]>float(condition)).astype(int)
    data['gain_'+str(back)] = data['Close'].shift(-int(back)) - data['Close']
    data['gain'] = data['gain_'+str(back)]*data['bool']
    return data

def send_zalo_message(text, with_img):
    # URL của API bạn muốn gọi
    get_api_url = """https://openapi.zalo.me/v2.0/oa/getfollowers?data={"offset":0,"count":10}"""
    headers = {
        "access_token": "NStRA8eOF4DQnTiWZ1WuJLVBt2Q105vHDhU_CwTW03CSrQu7cNOEFpJIwWIpGNyL2fN_6QP4Q2zzcFDDpYXT9ttx-H2cJ4ik1ChEHOyQL4OzhCqawouj16wRmNEyBt5p59hSPOKuUrmhk-TSf101DIY4WHgfAsTj88253_e3260eawXtyoGcPdI6e5ZxJYLXNll30iTJ9ZGKzBqxWc8p5H3eamYPL3GWECBLJEXuGbXU-iL6XNOaQX_tm6QAPN5CBkRANRTqUsOPm-jEZX1fMrkGyNJX5cT8IhlgQySqK6HpeUzQmIrXCm-Pnn6aJYKO9Vs85eDE0tWft9nFibqTOI72YboeGWz3UioDTVnXAbrbveSsw6aBEsFJZWx6KHHbLl-IRCnKSbvyX_LlrpHL10ANomzwCUFbr5gA1Zr4"  # Thêm token authorization nếu cần thiết
    }
    # Thực hiện một GET request đơn giản
    response = requests.get(get_api_url, headers=headers)
    data = response.json()
    user_ids = [follower["user_id"] for follower in data["data"]["followers"]]
    for user_id in user_ids:
        post_api_url = "https://openapi.zalo.me/v3.0/oa/message/cs"
        # Dữ liệu bạn muốn gửi đi dưới dạng JSON
        data_to_send = { 
            "recipient":{
                "user_id":user_id
            },
            "message":{
                "text":text,
                "attachment": {
                    "type": "template",
                    "payload": {
                        "template_type": "media",
                        "elements": [{
                            "media_type": "image",
                            "url": with_img
                        }]
                    }
                }
            }
        }
        headers = {
            "Content-Type": "application/json",
            "access_token": "NStRA8eOF4DQnTiWZ1WuJLVBt2Q105vHDhU_CwTW03CSrQu7cNOEFpJIwWIpGNyL2fN_6QP4Q2zzcFDDpYXT9ttx-H2cJ4ik1ChEHOyQL4OzhCqawouj16wRmNEyBt5p59hSPOKuUrmhk-TSf101DIY4WHgfAsTj88253_e3260eawXtyoGcPdI6e5ZxJYLXNll30iTJ9ZGKzBqxWc8p5H3eamYPL3GWECBLJEXuGbXU-iL6XNOaQX_tm6QAPN5CBkRANRTqUsOPm-jEZX1fMrkGyNJX5cT8IhlgQySqK6HpeUzQmIrXCm-Pnn6aJYKO9Vs85eDE0tWft9nFibqTOI72YboeGWz3UioDTVnXAbrbveSsw6aBEsFJZWx6KHHbLl-IRCnKSbvyX_LlrpHL10ANomzwCUFbr5gA1Zr4"  # Thêm token authorization nếu cần thiết
        }
        # Thực hiện một POST request với dữ liệu gửi kèm dưới dạng JSON
        response = requests.post(post_api_url, json=data_to_send, headers=headers)
            
    if response.status_code != 200:
        print(f"Failed to call API. Status code: {response.status_code}")
        print(response.text)

def upload_imgur(path):
    client_id = '18cba0cd32f2305'
    client_secret = '47c7dc90bfec7fdefd3689d9b146535a347add5d'

    # Tạo đối tượng ImgurClient
    client = ImgurClient(client_id, client_secret)

    # Tải ảnh lên Imgur
    uploaded_image = client.upload_from_path(path, anon=True)

    # Lấy URL của ảnh
    image_url = uploaded_image['link']
    return image_url

def draw_5days_pnl(data):
    img = data['gain'].iloc[-1275:].cumsum().plot()
    plt.savefig('plot.png')
    
def six_months_ratio(data):
    return data['gain'].iloc[-180:].mean()/data['gain'].iloc[-180:].std()*np.sqrt(252)

def today_nav(data):
    # nav = round((data['14:45'].iloc[-1]-data['14:45'].iloc[-2])*100 / data['14:45'].iloc[-2], 4)
    nav = round((data['gain'].iloc[-1]/data['13:45'].max())*100, 4)
    return nav

def day_report(data):
    six_month_ratio = six_months_ratio(data)
    nav = today_nav(data)
    today = data.index[-1]
    message = f"""
    Financial Report
    ---------------------------------------
    Alpha name: HEPHAESTUS_TECH_BASE_TRADE
    Date: {today}
    
    DAILY_NAV: {nav:,.2f}%
    6_MONTHS_SHARPE_RATIO: {six_month_ratio:,.2f}
    
    ---------------------------------------
    End of Financial Report
    """
    # message = "DUONG_NGO_BASE   "+"   "+today+"_NAV="+str(nav)+"%    6_MONTHS_SHARPE_RATIO="+str(round(six_month_ratio,4))
    return message

def model_predict(bt, data):
    with open('weight.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
        bt['predict'] = list(loaded_model.predict(data))
        #bt = bt[bt['abs_predict']>bt['abs_rolling'] ]
        bt['signal'] = (bt['predict']>0).astype(int).replace(0,-1)
        bt['gain'] = (bt['14:45'] - bt['13:45'])*bt['signal'] - 0.075
        return bt

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

def transform_us_index(us_index):
    return pd.DataFrame(us_index, index= close_feature.index).shift(1).ffill()

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
sp = transform_us_index(sp1)
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
    model_data['trend_from_ato'] = (close_feature[time_enter]/close_feature['09:00'] - 1).ffill()
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
    # print("f_names: "+','.join(f_names))
    # print(model_train)
    model.fit(model_train.dropna()[f_names], model_train.dropna()['Target'])
    bt = close_feature[close_feature.index.isin(model_test.index)][[time_exit,time_enter]]
    bt['predict'] = list(model.predict(model_test[f_names]))
    #bt = bt[bt['abs_predict']>bt['abs_rolling'] ]
    bt['signal'] = (bt['predict']>0).astype(int).replace(0,-1)
    bt['gain'] = (bt[time_exit] - bt[time_enter])*bt['signal'] - 0.075
    k = -(bt['gain']>0).astype(int).sum()/len(bt['gain'])
    bt_list.append(bt)

api_id = '23426972'
api_hash = 'e161487f61d6c6931b58c69892050954'
token = '6839571824:AAHfZlOA0EhyAvq8-06lQyHT7FQg3kR6UGE'
phone = '+84905982163'
client = TelegramClient('session', api_id, api_hash)
client.connect()

if not client.is_user_authorized():
	client.send_code_request(phone)
	client.sign_in(phone, input('Enter the code: '))

try:
    receiver = InputPeerChannel(-1002060032542, 0)
    # bt = data_supplier_2()
    message = day_report(bt)
    img = bt['gain'].iloc[-30:].cumsum().plot()
    plt.savefig('plot.png')
    client.send_file('@hephaestus_trading', file='plot.png', caption=message)
    if os.path.exists('plot.png'):
        os.remove('plot.png')
except Exception as e:
	print(e);

client.disconnect()
