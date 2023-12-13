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
import altair as alt

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

def get_vn30_test(time):
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
    vn30fm = vn30fm[vn30fm['Date'] <= time]
    if vn30fm['Date'].iloc[-1] == datetime(2023, 11, 29, 13, 30, 0) or vn30fm['Date'].iloc[-1] == datetime(2023, 11, 29, 13, 35, 0) or vn30fm['Date'].iloc[-1] == datetime(2023, 11, 29, 13, 40, 0) or vn30fm['Date'].iloc[-1] == datetime(2023, 11, 29, 13, 45, 0) or vn30fm['Date'].iloc[-1] == datetime(2023, 11, 29, 13, 50, 0) or vn30fm['Date'].iloc[-1] == datetime(2023, 11, 29, 13, 55, 0) or vn30fm['Date'].iloc[-1] == datetime(2023, 11, 29, 15, 0, 0):
        vn30fm = vn30fm.drop(vn30fm.index[-1])
    return vn30fm.set_index('Date')

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
    return data['gain'].iloc[-45900:].mean()/data['gain'].iloc[-45900:].std()*np.sqrt(252)

def today_nav(data):
    last_day_data = data[data['datetime'].dt.date == data['datetime'].dt.date.max()]
    nav = round((last_day_data['Close'].iloc[-1]-last_day_data['Close'].iloc[0])*100 / last_day_data['Close'].iloc[0], 4)
    return nav

def day_report(data):
    six_month_ratio = six_months_ratio(data)
    nav = today_nav(data)
    message = "4-0.21-3_BASE   "+"   TODAY_NAV="+str(nav)+"%    6_MONTHS_SHARPE_RATIO="+str(round(six_month_ratio,4))
    return message



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
    current_time = 0
    missing_time = False
    last_day_report = True
    while True:
        # df = get_vn30_test(now-timedelta(minutes=181))
        df = get_vn30()
        now = datetime.now().replace(second=0, microsecond=0)
        df['Date'] = df.index
        df = df.reset_index(drop=True)
        current_time = df['Date'].iloc[-1] if current_time == 0 else current_time
        if is_outside_trading_hours():
            continue
        elif current_time == df['Date'].iloc[-1]:
            time.sleep(20)
            df = get_vn30()
            now = datetime.now().replace(second=0, microsecond=0)
            df['Date'] = df.index
            df = df.reset_index(drop=True)
            if current_time == df['Date'].iloc[-1]:
                # if (current_time + timedelta(minutes=2)) == (now-timedelta(minutes=180)) and missing_time:
                if now - current_time >= timedelta(minutes=2) and missing_time:
                    missing_time = False
                    message = "Missing value"
                    client.send_message('@hephaestus_trading', message, parse_mode='html')
                continue
            else:
                continue
        current_time = df['Date'].iloc[-1]
        print(current_time)
        print(str(current_time + timedelta(minutes=1)) + ' $ ' + str(now))
        missing_time = True
        data = data_supplier()
        data = data[data['datetime'] <= df['Date'].iloc[-1]]
        data = alpha_run(4,0.21,3, data)
        message = str('4-0.21-3_BASE   '+str(data['bool'].iloc[-1])+'   '+str(data['Close'].iloc[-1])+'   '+str(data['datetime'].iloc[-1]))
        client.send_message('@hephaestus_trading', message, parse_mode='html')
        if datetime.now() >= datetime(datetime.now().year, datetime.now().month, datetime.now().day, 14, 45) and last_day_report:
            last_day_report = False
            data['gain'] = data['gain'].fillna(method='ffill')
            # gain_data = data.groupby(pd.to_datetime(data['Date']).dt.date)['gain'].cumsum()
            message = day_report(data)
            img = data['gain'].iloc[-1275:].cumsum().plot()
            plt.savefig('plot.png')
            client.send_file('@hephaestus_trading', file='plot.png', caption=message)
            # image_url = upload_imgur('plot.png')
            # send_zalo_message(message, image_url)
            if os.path.exists('plot.png'):
                os.remove('plot.png')
except Exception as e:
	print(e);


# df = get_vn30()
# now = datetime.now().replace(second=0, microsecond=0)
# df['Date'] = df.index
# df = df.reset_index(drop=True)
# current_time = df['Date'].iloc[-1]
# data = data_supplier()
# data['datetime'] = pd.to_datetime(data['Date'] + ' ' + data['time'])
# data = data[data['datetime'] <= df['Date'].iloc[-1]]
# data = alpha_run(4,0.21,3, data)
# message = str('4-0.21-3_BASE   '+str(data['bool'].iloc[-1])+'   '+str(data['Close'].iloc[-1])+'   '+str(data['datetime'].iloc[-1]))
# client.send_message('@hephaestus_trading', message, parse_mode='html')
# last_day_report = False
# data['gain'] = data['gain'].fillna(method='ffill')
# gain_data = data.groupby(pd.to_datetime(data['Date']).dt.date)['gain'].cumsum()
# six_month_ratio = data['gain'].iloc[-45900:].mean()/data['gain'].iloc[-45900:].std()*np.sqrt(252)
# last_day_data = data[data['datetime'].dt.date == data['datetime'].dt.date.max()]
# nav = round((last_day_data['Close'].iloc[-1]-last_day_data['Close'].iloc[0])*100 / last_day_data['Close'].iloc[0], 4)
# message = day_report(data)
# img = data['gain'].iloc[-1275:].cumsum().plot()
# plt.savefig('plot.png')
# client.send_file('@hephaestus_trading', file='plot.png', caption=message)
# # image_url = upload_imgur('plot.png')
# # send_zalo_message(message, image_url)
# if os.path.exists('plot.png'):
#     # Delete the file
#     os.remove('plot.png')

client.disconnect()
