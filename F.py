import requests
import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
from datetime import timedelta, time
import matplotlib.pyplot as plt
import sqlalchemy

class Query_realtime():
    def __init__(self):
        self.db_realtime=sqlalchemy.create_engine('postgresql://client1:finpros2022@192.168.110.15:6431/db_data')
        self.db_history=sqlalchemy.create_engine('postgresql://client2:finpros2022@192.168.110.15:6431/db_ps')
        
    def history_realtime(self, day=datetime.datetime.now().strftime('%Y_%m_%d')):
        query=f'''select "Time",
                "Close_Price",
                "Total_Volume"
                from  realtime_{day}
                where "Open_Interest"=(select max("Open_Interest") from realtime_{day}) 
                order by "Time" desc '''
        df=pd.read_sql_query(query,self.db_realtime)
        df.set_index('Time',inplace=True)
        return df
    
    def query_his_real(self, list_day=[datetime.datetime.now().strftime('%Y_%m_%d')], duration=5):
        list_df = []
        for day in list_day:
            try:
                list_df.append(self.history_realtime(day))
            except:
                pass

        df3=pd.concat(list_df).dropna()
        df4 = df3.resample(f'{duration}Min',closed='right').last().dropna()
        df4['Open'] = df3.resample(f'{duration}Min',closed='right').Close_Price.first()
        df4['High'] = df3.resample(f'{duration}Min',closed='right').Close_Price.max()
        df4['Low'] = df3.resample(f'{duration}Min',closed='right').Close_Price.min()
        df4 = df4.reset_index().loc[~df4.reset_index()['Time'].dt.time.isin([time(14,31), time(14,32), time(14,33), time(14,34), \
                                                                            time(14,35), time(14,36), time(14,37), time(14,38), \
                                                                            time(14,39), time(14,40), time(14,41), time(14,42), \
                                                                            time(14,43), time(14,44), time(8,55), time(8, 50), \
                                                                            time(8, 39), time(8, 40), time(8, 41),time(8, 42),time(8, 43), \
                                                                            time(8, 44), time(8, 45), time(8, 46),time(8, 47),time(8, 48), \
                                                                            time(8, 49),time(8, 51),time(8, 52),time(8, 53),time(8, 54),\
                                                                            time(8, 56),time(8, 57),time(8, 58),time(8, 59),time(8, 0)])]
        df4['Volume'] = df4.Total_Volume.diff()
        df4.loc[df4['Time'].dt.time == time(9,0), 'Volume'] = df4.Total_Volume
        df4.loc[df4.index[0], 'Volume'] = df4.loc[df4.index[0], 'Total_Volume']
        return df4.rename(columns={'Time':'Datetime', 'Close_Price':'Close'}).set_index('Datetime')

def get_vn30(duration):
    def vn30():                                     
        return requests.get("https://services.entrade.com.vn/chart-api/v2/ohlcs/index?from=0&resolution=1&symbol=VN30&to=9999999999").json()
    vn30fm = pd.DataFrame(vn30()).iloc[:,:6]
    vn30fm['t'] = vn30fm['t'].astype(int).apply(lambda x: datetime.datetime.utcfromtimestamp(x) + timedelta(hours = 7))
    vn30fm.columns = ['Date','Open','High','Low','Close','Volume']
    s = pd.read_csv('../VN30.csv')
    s['Date'] = pd.to_datetime(s['Date']) + timedelta(hours=7)
    ohlc_dict = {                                                                                                             
        'Open': 'first',                                                                                                    
        'High': 'max',                                                                                                       
        'Low': 'min',                                                         
        'Close': 'last',                                                                                                    
        'Volume': 'sum',}
    vn30fm = pd.DataFrame(vn30()).iloc[:,:6]
    vn30fm['t'] = vn30fm['t'].astype(int).apply(lambda x: datetime.datetime.utcfromtimestamp(x) + timedelta(hours = 7))
    vn30fm.columns = ['Date','Open','High','Low','Close','Volume']
    def process_data(input_df):
        vn30train = pd.DataFrame(input_df.resample(str(duration)+'Min', on='Date', label='left').apply(ohlc_dict).dropna()).reset_index()
        vn30train['Date'] = [str(i)[:16] for i in vn30train['Date']]
        return vn30train
    vn30f_base = pd.concat([process_data(vn30fm), process_data(s)]).sort_values('Date').drop_duplicates('Date').sort_values('Date')
    return vn30f_base

def portfolio_pnl_future(position_long, position_short, Close):
    ''' tính PNL của một chiến thuật 
    position_long: series position long
    position_short: series position short'''
    intitial_capital_long = (position_long.iloc[0])*(Close.iloc[0])
    cash_long = (position_long.diff(1)*Close)
    cash_long[0] = intitial_capital_long
    cash_cs_long = cash_long.cumsum()
    portfolio_value_long = (position_long*Close)

    intitial_capital_short = (position_short.iloc[0])*(Close.iloc[0])
    cash_short = (position_short.diff(1)*Close)
    cash_short[0] = intitial_capital_short
    cash_cs_short = cash_short.cumsum()
    portfolio_value_short = (position_short*Close)

    backtest = (portfolio_value_long - cash_cs_long).iloc[1:] + (cash_cs_short - portfolio_value_short).iloc[1:]
    cash_max = (cash_long + cash_short).max()
    pnl =  backtest/cash_max
    
    ''' return PNL, lần vào lệnh lớn nhất, PNL tương đối theo % '''
    return backtest, cash_max, pnl

def Sharp(pnl):
    ''' Tính Sharp ratio '''
    r = pnl.diff(1)
    return r.mean()/r.std() * np.sqrt(252)

def maximum_drawdown_future(gain, cash_max):
    ''' Tính maximum drawdown theo điểm, theo % '''
    return (gain.cumsum().cummax() - gain.cumsum()).max(), (gain.cumsum().cummax() - gain.cumsum()).max()/cash_max

def Margin(test):
    ''' Tính Margin '''
    test = test.copy()
    try:
        test['signal_long'] = np.where(test['Position'] > 0, 1, 0)
        test['signal_short'] = np.where(test['Position'] < 0, 1, 0)
    except:
        pass
    test['total_gain'] = portfolio_pnl_future(test.signal_long, test.signal_short, test.Close)[0]
    test['inLong'] = test.signal_long.diff()[test.signal_long.diff() > 0].astype(int)
    test['inShort'] = test.signal_short.diff()[test.signal_short.diff() > 0].astype(int)
    test['outLong'] = -test.signal_long.diff()[test.signal_long.diff() < 0].astype(int)
    test['outShort'] = -test.signal_short.diff()[test.signal_short.diff() < 0].astype(int)
    test.loc[test.index[0], 'inLong'] = test.signal_long.iloc[0]
    test.loc[test.index[0], 'inShort'] = test.signal_short.iloc[0]
    test.fillna(0, inplace=True)

    ''' return dataframe chưa thêm các cột inLong, inShort, outLong, outShort và Margin '''
    return test, test.total_gain.iloc[-1]/(test.inLong * test.Close + test.inShort * test.Close + test.outLong * test.Close + test.outShort * test.Close).sum()*10000

def HitRate(test):
    ''' Tính Hit Rate '''
    test = test.copy()
    try:
        test['signal_long'] = np.where(test['Position'] > 0, 1, 0)
        test['signal_short'] = np.where(test['Position'] < 0, 1, 0)
    except:
        pass
    test['total_gain'] = portfolio_pnl_future(test.signal_long, test.signal_short, test.Close)[0]
    test = Margin(test)[0]
    test = test[((test.outLong == 1) | (test.outShort == 1) | (test.inLong == 1) | (test.inShort == 1))]
    test['total_gain'] = portfolio_pnl_future(test.signal_long, test.signal_short, test.Close)[0]
    test.fillna(0, inplace=True)
    test['gain'] = test.total_gain.diff()
    test.fillna(0, inplace=True)
    test['gain'] = np.where(np.abs(test.gain) < 0.00001, 0, test.gain)
    try:
        ''' return dataframe thu gọn và Hit Rate'''
        return test, len(test[test.gain > 0])/(len(test[test.inLong == 1]) + len(test[test.inShort == 1]))
    except:
        return 0

def test_live(duration):
    ''' Lấy dữ liệu từ API '''
    ''' Input: duration: sample dữ liệu theo phút '''
    def vn30f():
        return requests.get("https://services.entrade.com.vn/chart-api/chart?from=1651727820&resolution=1&symbol=VN30F1M&to=9999999999").json()
    vn30fm = pd.DataFrame(vn30f()).iloc[:,:6]
    vn30fm['t'] = vn30fm['t'].astype(int).apply(lambda x: datetime.datetime.utcfromtimestamp(x) + timedelta(hours = 7))
    vn30fm.columns = ['Date','Open','High','Low','Close','Volume']
    s = pd.read_csv('../Data/DataMinute/VN30F1M.csv')
    s['Date'] = pd.to_datetime(s['Date']) + timedelta(hours =7)
    ohlc_dict = {                                                                                                             
        'Open': 'first',                                                                                                    
        'High': 'max',                                                                                                       
        'Low': 'min',                                                                                                        
        'Close': 'last',                                                                                                    
        'Volume': 'sum',}
    vn30fm = pd.DataFrame(vn30f()).iloc[:,:6]
    vn30fm['t'] = vn30fm['t'].astype(int).apply(lambda x: datetime.datetime.utcfromtimestamp(x) + timedelta(hours = 7))
    vn30fm.columns = ['Date','Open','High','Low','Close','Volume']
    def process_data(input_df):
        vn30train = pd.DataFrame(input_df.resample(str(duration)+'Min', on='Date', label='left').apply(ohlc_dict).dropna()).reset_index()#change s
        vn30train['Date'] = [str(i)[:16] for i in vn30train['Date']]
        return vn30train
    vn30f_base = pd.concat([process_data(vn30fm), process_data(s)]).sort_values('Date').drop_duplicates('Date').sort_values('Date')
    return vn30f_base

def send_to_telegram(message, token='5683073192:AAHOAHjiRwk3pbNWI4dPFfURa4YaySvbfLY', id='-879820435'):
    ''' Gửi tin nhắn đến telegram '''
    ''' Input: message: tin nhắn muốn gửi
               token: token của bot
                id: id của chat group '''
    apiToken = token
    chatID = id
    try:
        apiURL = f"https://api.telegram.org/bot{apiToken}/sendMessage?chat_id={chatID}&text={message}"
        requests.get(apiURL).json()
    except Exception as e:
        print(e)

def position_input(position, path_Po='G:/alpha_live_pos/PHU/PS13_PHU.txt'):
    ''' ghi file position input (cho chiến thuật chạy live) '''
    ''' Input: position: vị thế của chiến thuật 
               path_Po: đường dẫn file position input'''
    f = open(path_Po, "w")
    if position != 0:
        info = "pos={}\ntime=5".format(position)
        f.write(info)
    else:
        info = "pos=0\ntime=0"
        f.write(info)

def position_report(position, path_CP='G:/alpha_live_pos/PHU/PS13_PHU_CP.txt'):
    ''' ghi file position report (vị thế hiện tại) (cho chiến thuật chạy live) 
        Input: position: vị thế của chiến thuật
               path_CP: đường dẫn file position report'''
    f = open(path_CP, "w")
    pos_rp = "pos={}".format(position)
    f.write(pos_rp)

def DumpCSV_and_MesToTele(name, path_csv_intraday, Position, Close, token, id, position_input=1):
    ''' Ghi file csv và gửi tin nhắn đến telegram 
        Input: name: tên của chiến thuật
               path_csv_intraday: đường dẫn file csv intraday
               Position: Series vị thế của chiến thuật 
               Close: Series giá khớp lệnh
               token: token của bot telegram
               id: id của chat group telegram
               position_input: số hợp đồng vào mỗi lệnh'''
    try:
        df = pd.read_csv(path_csv_intraday)
        dict_data = {
            'Datetime': df.Datetime.tolist(),
            'Position': df.Position.tolist(),
            'Close': df.Close.tolist(),
        }
    except:
        dict_data = {
            'Datetime': [(datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')],
            'Position': [0],
            'Close': [0],
        }
        df = pd.DataFrame(data=dict_data)
        df['total_gain'] = 0
        df['gain'] = 0
        df.to_csv(path_csv_intraday, index=False)

    Close = Close.iloc[-1]
    new_Pos = int(Position.iloc[-1])
    time_now = datetime.datetime.now()
    profit = 0
    profit_today = 0
    mes = f'{name}:'
    
    if new_Pos != dict_data['Position'][-1] or time_now.time() >= datetime.time(14, 45):

        inputPos = int(new_Pos - dict_data['Position'][-1])
        dict_data['Datetime'].append(time_now.strftime('%Y-%m-%d %H:%M:%S'))
        dict_data['Close'].append(Close)
        dict_data['Position'].append(new_Pos)

        try:
            dict_data['Datetime'] = pd.to_datetime(dict_data['Datetime'])
        except:
            for i in range(len(dict_data['Datetime'])):
                dict_data['Datetime'][i] = pd.to_datetime(dict_data['Datetime'][i])

        df = pd.DataFrame(data=dict_data)
        df['signal_long'] = np.where(df.Position > 0, df.Position, 0)
        df['signal_short'] = np.where(df.Position < 0, np.abs(df.Position), 0)
        df['total_gain'] = portfolio_pnl_future(df['signal_long'], df['signal_short'], df.Close)[0]
        df['gain'] = df.total_gain.diff()
        df.fillna(0, inplace=True)
        df['gain'] = np.where(np.abs(df.gain.to_numpy()) < 0.00001, 0, df.gain.to_numpy())
        profit = df.gain.iloc[-1]
        profit_today = df.loc[df.Datetime.dt.date == time_now.date(), 'gain'].sum()
        
        if inputPos > 0:
            mes = f'{name}:\nLong {inputPos*position_input} at {Close}, Current Pos: {new_Pos*position_input}'
        elif inputPos < 0:
            mes = f'{name}:\nShort {inputPos*position_input} at {Close}, Current Pos: {new_Pos*position_input}'
        else:
            mes = f'{name}:\nClose at {Close}, Current Pos: {new_Pos*position_input}'

        if np.round(profit*10)/10 != 0:
            mes += f'\nProfit: {np.round(profit*10)/10}'
        mes += f'\nProfit today: {np.round(profit_today*10)/10}'

        df.drop(columns=['signal_long', 'signal_short'], inplace=True)
        send_to_telegram(mes, token, id)
        df.to_csv(path_csv_intraday, index=False)

    else:
        inputPos = 0
        
    print(name)
    print(time_now)
    print(Close)
    print('Input Position:', inputPos*position_input)
    print('Current Position:', new_Pos*position_input)
    if np.round(profit*10)/10 != 0:
        print(f'Profit: {np.round(profit*10)/10}')
    print(f'Profit today: {np.round(profit_today*10)/10}')
    print('\n')

    ''' return dataframe intraday, input position, current position'''
    return df, inputPos, new_Pos

def PNL_per_day(path_csv_daily, total_gain):
    ''' Ghi file csv PNL theo ngày
        Input: path_csv_daily: đường dẫn file csv PNL theo ngày
                total_gain: Series PNL của chiến thuật'''
    try:
        df = pd.read_csv(path_csv_daily)
        dict_data = {
            'Datetime': df.Datetime.tolist(),
            'total_gain': df.total_gain.tolist(),
        }
    except:
        dict_data = {
            'Datetime': [(datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')],
            'total_gain': [0],
        }
        df = pd.DataFrame(data=dict_data)
        df.to_csv(path_csv_daily, index=False)

    total_gain = total_gain.iloc[-1]
    time_now = datetime.datetime.now()

    if time_now.strftime('%Y-%m-%d') != pd.to_datetime(dict_data['Datetime'][-1]).strftime('%Y-%m-%d'):
        if total_gain != dict_data['total_gain'][-1]:
            dict_data['Datetime'].append(time_now.strftime('%Y-%m-%d'))
            dict_data['total_gain'].append(total_gain)
            df = pd.DataFrame(data=dict_data)
    else:
        dict_data['total_gain'][-1] = total_gain
        df = pd.DataFrame(data=dict_data)

    df['gain'] = 0
    try:
        df['gain'] = df['total_gain'].diff()
    except:
        pass
    df['gain'] = np.where(np.abs(df.gain.to_numpy()) < 0.00001, 0, df.gain.to_numpy())
    df.fillna(0, inplace=True)

    df.to_csv(path_csv_daily, index=False)
    ''' return dataframe PNL theo ngày '''
    return df

def Review_paper_trade(Datetime, Position, Close):
    ''' Review chiến thuật chạy thử
        Input: Datetime: Series Datetime
                Position: Series Position
                Close: Series Close '''
    BacktestInfo = BacktestInformation(Datetime, Position, Close)
    return BacktestInfo.Plot_PNL()

class BacktestInformation:
    ''' Thông tin backtest của chiến thuật 
        Input: Datetime: Series Datetime
                Position: Series Position
                Close: Series Close '''
    ''' CHÚ Ý: Nên dùng class này để lấy được các thông tin của chiến thuật chứ không nên dùng các hàm riêng lẻ
               vì các hàm riêng lẻ phía trên có thể có định dạng position không đồng nhất với class này '''
    def __init__(self, Datetime, Position, Close):
        signal_long = np.where(Position >= 0, Position, 0)
        signal_short = np.where(Position <= 0, np.abs(Position), 0)
        try:
            Datetime = pd.to_datetime(Datetime)
        except:
            Datetime = Datetime.to_list()
            for i in range(len(Datetime)):
                Datetime[i] = pd.to_datetime(Datetime[i])
        self.df = pd.DataFrame(data={'Datetime': Datetime, 'signal_long': signal_long, 'signal_short': signal_short, 'Close': Close})
        self.df.set_index('Datetime', inplace=True)
        self.df.index = pd.to_datetime(self.df.index)
        self.df = HitRate(self.df)[0]
    
    def PNL(self):
        ''' Tính PNL của chiến thuật '''
        total_gain, cash_max, pnl = portfolio_pnl_future(self.df.signal_long, self.df.signal_short, self.df.Close) 

        ''' return Series PNL, cash_max '''
        return total_gain, cash_max, pnl
    
    def Sharp(self):
        ''' Tính Sharp của chiến thuật '''
        return Sharp(self.df.gain.cumsum().resample("1D").last().dropna())
    
    def Margin(self):
        ''' Tính Margin của chiến thuật '''
        return Margin(self.df)[1]
    
    def MDD(self):
        ''' Tính MDD của chiến thuật '''
        return maximum_drawdown_future(self.df.gain, self.PNL()[1])
    
    def Hitrate(self):
        ''' Tính Hitrate của chiến thuật '''
        return HitRate(self.df)[1]
    
    def Profit_per_trade(self):
        ''' Tính Profit trung bình của 1 giao dịch '''
        return self.df.total_gain.iloc[-1]/(len(self.df[self.df.inLong == 1]) + len(self.df[self.df.inShort == 1])) - 0.1
    
    def Number_of_trade(self):
        ''' Tính số lần giao dịch của chiến thuật '''
        return len(self.df[self.df.inLong == 1]) + len(self.df[self.df.inShort == 1])
    
    def Profit_after_fee(self):
        ''' Tính Profit sau khi trừ phí '''
        return np.round(self.Profit_per_trade() * self.Number_of_trade()*10)/10
    
    def Profit_per_day(self):
        ''' Tính Profit trung bình theo ngày '''
        return self.Profit_after_fee()/len(self.PNL()[0].resample("1D").last().dropna())
    
    def Hitrate_per_day(self):
        ''' Tính Hitrate theo ngày '''
        Profit = self.df.total_gain.resample("1D").last().dropna().diff()
        Profit[Profit.index[0]] = self.df.total_gain.resample("1D").last().dropna().iloc[0]
        return Profit, len(Profit[Profit > 0])/len(Profit)
    
    def Return(self):
        ''' Tính Return trung bình mỗi năm theo % của chiến thuật '''
        cash_max = self.PNL()[1]
        return self.Profit_after_fee()/(len(self.PNL()[0].resample("1D").last()) / 365)/cash_max    
    
    def Profit_per_year(self):
        ''' Tính Profit trung bình theo năm '''
        return self.Profit_after_fee()/(len(self.PNL()[0].resample("1D").last()) / 365)

    def Plot_PNL(self, after_fee=False):
        ''' Print thông tin và Vẽ biểu đồ PNL của chiến thuật 
            Input: after_fee: bool, True: plot có trừ phí, False: plot không trừ phí'''

        test = self.df.copy()
        if after_fee:
            test.loc[test.gain != 0, 'gain'] = test.gain - 0.1
        
        total_gain, cash_max, pnl = self.PNL()
        test['PNL'] = test.gain.cumsum()
        test['Return'] = test.gain.cumsum()/cash_max

        print('Margin:',Margin(test)[1])
        print(f'MDD: {maximum_drawdown_future(test.gain, cash_max)}\n')

        data = [('Total trading quantity', self.Number_of_trade()),
                ('Profit per trade',self.Profit_per_trade()),
                ('Total Profit', np.round(total_gain.iloc[-1]*10)/10),
                ('Profit after fee', self.Profit_after_fee()),
                ('Trading quantity per day', self.Number_of_trade()/len(total_gain.resample("1D").last().dropna())),
                ('Profit per day after fee', self.Profit_per_day()),
                ('Return', self.Return()),
                ('Profit per year', self.Profit_per_year()),
                ('HitRate', self.Hitrate()),
                ('HitRate per day', self.Hitrate_per_day()[1]),
                ]
        for row in data:
            print('{:>25}: {:>1}'.format(*row))

        test.reset_index(inplace=True)
        previous_day = pd.DataFrame(test.iloc[0].to_numpy(), index=test.columns).T
        previous_day.loc[previous_day.index[0], 'Datetime'] = pd.to_datetime(previous_day['Datetime'].iloc[0]) - timedelta(days = 1) 
        previous_day.loc[previous_day.index[0], ['signal_long', 'signal_short', 'gain']] = [0, 0, 0]
        test = pd.concat([previous_day, test]).set_index('Datetime')

        (test.gain.cumsum().resample("1D").last().dropna()).plot(figsize=(15, 4), label=f'{Sharp(test.gain.cumsum().resample("1D").last().dropna())}')
        plt.grid()
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('PNL')
        plt.show()

        plt.figure()
        ((1 + test['Return'])).plot(figsize=(15, 4), label=f'{Sharp(total_gain.resample("1D").last().dropna())}')
        plt.legend()
        plt.grid()
        plt.xlabel('Time')
        plt.ylabel('Return')
        plt.show()

        return test.drop(columns=['inLong', 'inShort', 'outLong', 'outShort']).fillna(0)