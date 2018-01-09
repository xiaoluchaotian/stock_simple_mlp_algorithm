

import pandas as pd
import MySQLdb
import tushare as ts
import datetime
import time
import tushare as ts
import os
import pickle
from sqlalchemy import create_engine
# from WindPy import *
from func_deeplearning import *
from WindPy import *
from sklearn import preprocessing


def get_trade_days():#获取所有的交易日
    w.start()
    trade_days=w.tdays("2010-05-01")#返回5 月1 日到6 月8 日之间的交易日序列
    engine=create_engine('mysql://root:0325xb@localhost/trade_days_list?charset=utf8')
    for i in range(len(trade_days.Times)):
        #trade_days.Times[i]=time.strftime('%Y-%m-%d',datetime.strptime(fdate,'%Y-%m-%d')(trade_days.Times[i]))
        trade_days.Times[i]=str(trade_days.Times[i])[0:19]
    trade_days.Times=pd.DataFrame(trade_days.Times)
    print(trade_days.Times)
    trade_days.Times.to_sql('trade_days_list',engine,if_exists='replace')
#factors = ['esp','roe','net_profits','profits_yoy','business_income','bips','cashflowratio','rateofreturn']
#每股收益，净资产收益率(%)，净利润(万元），净利润同比(%)，营业收入(百万元),每股主营业务收入(元),现金流量比率,资产的经营现金流量回报率
#get_trade_days()
def get_n_tradeday_price(stocks,fdate,n_days):#该日期前n天的收盘均价和增长（不含当天
    fdate=fdate+' 00:00:00'
    engine=create_engine('mysql://root:0325xb@localhost/trade_days_list?charset=utf8')
    trade_days_list=pd.read_sql('trade_days_list',engine)
    trade_days_list=trade_days_list.loc[:,'0'].values.tolist()
    trade_index=trade_days_list.index(fdate)#定位当前日期在所有交易日的序号
    print(type(trade_index))
    trade_n_days=trade_days_list[trade_index-n_days:trade_index]
    df_n_days_price_mean=pd.DataFrame(stocks,index=stocks,columns=['code'])
    df_n_days_price_mean['%d_days_price_mean'%n_days]=None
    df_n_days_price_mean['%d_days_price_increase'%n_days]=None
    #df_n_days_price_mean.set_index('code')

    for stock in stocks:
        try:
            engine=create_engine('mysql://root:0325xb@localhost/stock_day_k_history_data_qfq?charset=utf8')
            df=pd.read_sql('stock_%s'%stock,engine)
            df.index=df['datetime']
            price=df.ix[trade_n_days[n_days-1]:trade_n_days[0],'close'].values
            #print(stock)
            try:
                df_n_days_price_mean.loc[stock,'%d_days_price_increase'%n_days]=100*(price[0]-price[n_days-1])/price[n_days-1]
                df_n_days_price_mean.loc[stock,'%d_days_price_mean'%n_days]=price.mean()
            except:
                df_n_days_price_mean.drop(stock,inplace=True)
        except:
                df_n_days_price_mean.drop(stock,inplace=True)
    #print(df_n_days_price_mean)
    return df_n_days_price_mean

def formatDate(Date, formatType='YYYYMMDD'):#将整数转化为年月日
    formatType = formatType.replace('YYYY', Date[0:4])
    formatType = formatType.replace('MM', Date[4:6])
    formatType = formatType.replace('DD', Date[-2:])
    return formatType



def get_n_days_increase(stocks,fdate,n_days):#该日期后n天的收盘价增长，作为训练数据的label
    fdate=fdate+' 00:00:00'
    engine=create_engine('mysql://root:0325xb@localhost/trade_days_list?charset=utf8')
    trade_days_list=pd.read_sql('trade_days_list',engine)
    trade_days_list=trade_days_list.loc[:,'0'].values.tolist()
    trade_index=trade_days_list.index(fdate)
    trade_n_days=trade_days_list[trade_index:trade_index+n_days]
    print(trade_n_days)
    df_train_y=pd.DataFrame(stocks,index=stocks,columns=['code'])
    df_train_y['%d_days_price_increase'%n_days]=None
    for stock in stocks:
        try:
            engine=create_engine('mysql://root:0325xb@localhost/stock_day_k_history_data_qfq?charset=utf8')
            df=pd.read_sql('stock_%s'%stock,engine)
            df.index=df['datetime']
            try:
                price=df.ix[trade_n_days[n_days-1]:trade_n_days[0],'close'].values
                df_train_y.loc[stock,'%d_days_price_increase'%n_days]=100*(price[0]-price[n_days-1])/price[n_days-1]
                #print(stock)
            except:
                df_train_y.drop(stock,inplace=True)
        except:
                df_train_y.drop(stock,inplace=True)
    df_train_y.to_csv("y.csv",encoding = "GB18030")
    return df_train_y



#
# get_n_days_increase(['000001','000792'],'2017-12-25',2)


def get_train_x(fdate):

    n_pe_days=5#计算pe时采用的几日均价
    n_y_days=10#计算pe时采用的几日均价

    engine=create_engine('mysql://root:0325xb@localhost/stocks_financial_data?charset=utf8')
    #df.to_csv("financial_data_%s_%s.csv"%(year,quarter),encoding = "GB18030")
    #df.to_sql('financial_data_%s_%s'%(year,quarter),engine,if_exists='append')
    ptime=datetime.strptime(fdate,'%Y-%m-%d')
    year=ptime.year
    month=ptime.month
    quarter_list=[1,2,3,4]

    quarter=int((ptime.month+2)/3)
    i_quarter=quarter-2
    if month in [1,4,7,10]:
        i_quarter=quarter-3
        quarter=quarter_list[i_quarter]#财报在下季度的第一个月完成，当季第一个月需要使用上上季度数据
        if month in [1,4]:
            year=year-1
    else:
        if month in [2,3]:
            year=year-1
        quarter=quarter_list[i_quarter]

    i_previous_quarter=i_quarter-1
    previous_quarter=quarter_list[i_previous_quarter]
    previous_year=year
    if previous_quarter==4:
        previous_year=year-1
    #index_data.ix[i,'trade_date']=time.strftime('%Y%m%d',ptime)
    print(previous_year,previous_quarter,year,quarter)
    df=pd.read_sql('financial_data_%s_%s'%(year,quarter),engine)
    df.index=df['code']
    #eps取两个季度财报的平均值
    previous_df=pd.read_sql('financial_data_%s_%s'%(previous_year,previous_quarter),engine)
    previous_eps=previous_df.loc[:,['code','eps_x']]
    now_eps=df.loc[:,['code','eps_x']]
    now_eps.index=now_eps['code']
    previous_eps.index=previous_eps['code']
    eps=pd.merge(now_eps,previous_eps ,on='code')
    eps['eps']=(eps['eps_x_x']+eps['eps_x_y'])/2
    eps.to_csv("eps.csv",encoding = "GB18030")
    df_train_x=df.loc[:,['code','roe_x','net_profits_x','profits_yoy','business_income','bips','cashflowratio','rateofreturn','nprg','epsg']]
    df_train_x=pd.merge(df_train_x,eps,on='code')
    stocks=df_train_x['code'].values
    del df_train_x['eps_x_x']
    del df_train_x['eps_x_y']
    #取前五日股票均价，作为计算pe的股价pe=price/eps
    #stocks=df_data
    df_n_days_price=get_n_tradeday_price(stocks,fdate,n_pe_days)
    df_n_days_price.rename(columns={'%d_days_price_mean'%n_pe_days:'pe'}, inplace = True)
    df_train_x=pd.merge(df_train_x,df_n_days_price,on='code')
    df_train_x=df_train_x.dropna()

    stocks=df_train_x['code'].values
    df_train_x.index=df_train_x['code']
    df_time_to_market=ts.get_stock_basics().loc[:,['code','timeToMarket']]

    for stock in stocks:
        time_to_market=formatDate(str(df_time_to_market.loc[stock,'timeToMarket']),'YYYY-MM-DD')
        test_now_date=datetime.strptime(fdate, '%Y-%m-%d')
        time_to_market=datetime.strptime(time_to_market, '%Y-%m-%d')
        if (test_now_date-time_to_market).days<90:
            df_train_x.drop(stock,axis=0,inplace=True)
        else:
            if df_train_x.loc[stock,'eps']==0:
                df_train_x.drop(stock,axis=0,inplace=True)
            else:
                df_train_x.loc[stock,'pe']=df_train_x.loc[stock,'pe']/df_train_x.loc[stock,'eps']
    #df_train_x=pd.merge(df_train_x,df_n_days_price,on='code')
    #df_data=df_data.reset_index()
    #df_train_x.to_csv("financial_data_%s_%s.csv"%(year,quarter),encoding = "GB18030")
    #df_train_x.to_pickle('df_train_x.pkl')



    #df_train_x=pd.read_pickle('df_train_x.pkl')
    df_train_x.index=df_train_x['code']
    stocks=df_train_x['code'].values
    #df_train_x.to_csv("financial_data_%s_%s.csv"%(2018,3),encoding = "GB18030")
    df_train_y=get_n_days_increase(stocks,fdate,n_y_days)
    df_train=pd.merge(df_train_x,df_train_y,on='code').dropna()
    #df_train.to_csv("dftrain.csv",encoding = "GB18030")
    path=os.path.abspath('.')
    #os.mkdir(path+'/train_data');
    df_train.to_pickle(path+'/train_data/df_train%s.pkl'%fdate)
    return df_train
def get_predict_future_x(fdate):

    n_pe_days=5#计算pe时采用的几日均价
    n_y_days=9#计算pe时采用的几日均价

    engine=create_engine('mysql://root:0325xb@localhost/stocks_financial_data?charset=utf8')
    #df.to_csv("financial_data_%s_%s.csv"%(year,quarter),encoding = "GB18030")
    #df.to_sql('financial_data_%s_%s'%(year,quarter),engine,if_exists='append')
    ptime=datetime.strptime(fdate,'%Y-%m-%d')
    year=ptime.year
    month=ptime.month
    quarter_list=[1,2,3,4]

    quarter=int((ptime.month+2)/3)
    i_quarter=quarter-2
    if month in [1,4,7,10]:
        i_quarter=quarter-3
        quarter=quarter_list[i_quarter]#财报在下季度的第一个月完成，当季第一个月需要使用上上季度数据
        if month in [1,4]:
            year=year-1
    else:
        if month in [2,3]:
            year=year-1
        quarter=quarter_list[i_quarter]

    i_previous_quarter=i_quarter-1
    previous_quarter=quarter_list[i_previous_quarter]
    previous_year=year
    if previous_quarter==4:
        previous_year=year-1
    #index_data.ix[i,'trade_date']=time.strftime('%Y%m%d',ptime)
    print(previous_year,previous_quarter,year,quarter)
    df=pd.read_sql('financial_data_%s_%s'%(year,quarter),engine)
    df.index=df['code']
    #eps取两个季度财报的平均值
    previous_df=pd.read_sql('financial_data_%s_%s'%(previous_year,previous_quarter),engine)
    previous_eps=previous_df.loc[:,['code','eps_x']]
    now_eps=df.loc[:,['code','eps_x']]
    now_eps.index=now_eps['code']
    previous_eps.index=previous_eps['code']
    eps=pd.merge(now_eps,previous_eps ,on='code')
    eps['eps']=(eps['eps_x_x']+eps['eps_x_y'])/2
    eps.to_csv("eps.csv",encoding = "GB18030")
    df_train_x=df.loc[:,['code','roe_x','net_profits_x','profits_yoy','business_income','bips','cashflowratio','rateofreturn','nprg','epsg']]
    df_train_x=pd.merge(df_train_x,eps,on='code')
    stocks=df_train_x['code'].values
    del df_train_x['eps_x_x']
    del df_train_x['eps_x_y']
    #取前五日股票均价，作为计算pe的股价pe=price/eps
    #stocks=df_data
    df_n_days_price=get_n_tradeday_price(stocks,fdate,n_pe_days)
    df_n_days_price.rename(columns={'%d_days_price_mean'%n_pe_days:'pe'}, inplace = True)
    df_train_x=pd.merge(df_train_x,df_n_days_price,on='code')
    df_train_x=df_train_x.dropna()

    stocks=df_train_x['code'].values
    df_train_x.index=df_train_x['code']
    df_time_to_market=ts.get_stock_basics().loc[:,['code','timeToMarket']]

    for stock in stocks:
        time_to_market=formatDate(str(df_time_to_market.loc[stock,'timeToMarket']),'YYYY-MM-DD')
        test_now_date=datetime.strptime(fdate, '%Y-%m-%d')
        time_to_market=datetime.strptime(time_to_market, '%Y-%m-%d')
        if (test_now_date-time_to_market).days<90:
            df_train_x.drop(stock,axis=0,inplace=True)
        else:
            if df_train_x.loc[stock,'eps']==0:
                df_train_x.drop(stock,axis=0,inplace=True)
            else:
                df_train_x.loc[stock,'pe']=df_train_x.loc[stock,'pe']/df_train_x.loc[stock,'eps']
    #df_train_x=pd.merge(df_train_x,df_n_days_price,on='code')
    #df_data=df_data.reset_index()
    #df_train_x.to_csv("financial_data_%s_%s.csv"%(year,quarter),encoding = "GB18030")
    #df_train_x.to_pickle('df_train_x.pkl')



    #df_train_x=pd.read_pickle('df_train_x.pkl')
    df_train_x.index=df_train_x['code']
    stocks=df_train_x['code'].values
    #df_train_x.to_csv("financial_data_%s_%s.csv"%(2018,3),encoding = "GB18030")
    # df_train_y=get_n_days_increase(stocks,fdate,n_y_days)
    # df_train=pd.merge(df_train_x,df_train_y,on='code').dropna()
    #df_train.to_csv("dftrain.csv",encoding = "GB18030")
    df_train=df_train_x
    path=os.path.abspath('.')
    #os.mkdir(path+'/train_data');
    df_train.to_pickle(path+'/train_data/df_train%s.pkl'%fdate)
    return df_train

#get_trade_days()
#get_n_tradeday_price(['000001','300486'],'2017-12-15',5)
#for i in range(12):
#df_train=get_train_x('2016-11-15')
def train():

    engine=create_engine('mysql://root:0325xb@localhost/trade_days_list?charset=utf8')
    trade_days_list=pd.read_sql('trade_days_list',engine)
    trade_days_list=trade_days_list.loc[:,'0'].values.tolist()
    start_train_date='2017-10-09 00:00:00'
    date_index=trade_days_list.index(start_train_date)
    print(date_index)
    cash=100
    df_predict=pd.DataFrame()
    df_predict=df_predict.reset_index()
    while(date_index<1807):

        date=str(trade_days_list[date_index])[0:10]
        print(date)
        path=os.path.abspath('.')
        if os.access(path+'/train_data/df_train%s.pkl'%date, os.F_OK):
            df_train=pd.read_pickle(path+'/train_data/df_train%s.pkl'%date)
        else:
            df_train=get_train_x(date)
        df_train_x=df_train.iloc[:,-13:-1].values.T
        df_train_x=preprocessing.scale(df_train_x,axis=1)
        train_x=np.asarray(df_train_x)
        # min_max_scaler = preprocessing.MinMaxScaler()
        # df_train_x = min_max_scaler.fit_transform(df_train_x)
        #df_train_x=preprocessing.normalize(df_train_x,axis=1, norm='l2')
        #print(df_train_x)
        #x_normalize=np.asarray(np.matrix(df_train_x.T/x_norm).T)
        df_train_y=df_train.iloc[:,-1].values
        print(df_train_y.shape)
        qwer= np.zeros(df_train_y.shape[0])
        for i in range(0, df_train_y.shape[0]):
            if df_train_y[i] > 2:
                qwer[i] = 1
            else:
                qwer[i] = 0
        train_y=qwer

        #使用新一个时段的训练数据分析上次训练效果
        #try:
        if date_index!=trade_days_list.index(start_train_date):
            test_x=train_x
            test_y=train_y
            if date_index==trade_days_list.index(start_train_date):
                previous_date=-1
            else:
                previous_date=str(trade_days_list[date_index-10])[0:10]
            path=os.path.abspath('.')
            output = open(path+'/parameters/%s.pkl'%previous_date, 'rb')
            parameters = pickle.load(output)
            p,prob=predict(test_x,test_y,parameters)
            price=[]
            codes=[]
            prob=prob[0]
            b=np.argsort(-prob)

            print(len(prob))
            for i in range(20):
                codes.append(df_train.ix[b[i],'code'])
                #print(codes)
                price.append(df_train.ix[b[i],'10_days_price_increase'])
            cash=cash+sum(price)/20
            df_predict['code_%s'%date]=codes
            df_predict['10_days_price_increase_%s'%date]=price
            df_predict['cash_%s'%date]=cash
        #except:
        else:
            print('First train:No parameters exist')
        #使用新一个时段的训练数据分析上次训练效果


        ### 定义初始设定 ###
        n_x = 12    # feature数量
        n_h = 100
        n_y = 1
        layers_dims = (n_x, n_h, n_y)
        previous_date=None
        if date_index==trade_days_list.index(start_train_date):
            previous_date=-1
        else:
            previous_date=str(trade_days_list[date_index-10])[0:10]
        #previous_date=-1
        parameters = two_layer_model(train_x, train_y,  previous_date,layers_dims = (n_x, n_h, n_y), num_iterations = 6000, print_cost=True)
        path=os.path.abspath('.')
        #os.mkdir(path+'/parameters');
        #df_train.to_pickle(path+'/parameters/%s.pkl'%date)
        output = open(path+'/parameters/%s.pkl'%date, 'wb')
        # Pickle dictionary using protocol 0.
        pickle.dump(parameters, output)
        output.close()
        date_index=date_index+10
    df_predict.to_csv("dfpredict1.csv",encoding = "GB18030")

def  predict_future():
    engine=create_engine('mysql://root:0325xb@localhost/trade_days_list?charset=utf8')
    trade_days_list=pd.read_sql('trade_days_list',engine)
    trade_days_list=trade_days_list.loc[:,'0'].values.tolist()
    date_index=trade_days_list.index('2017-12-18 00:00:00')
    previous_date=str(trade_days_list[date_index])[0:10]
    date=str(trade_days_list[date_index+10])[0:10]
    path=os.path.abspath('.')
    if os.access(path+'/train_data/df_train%s.pkl'%date, os.F_OK):
        df_train=pd.read_pickle(path+'/train_data/df_train%s.pkl'%date)
    else:
        df_train=get_predict_future_x(date)
    code_num=len(df_train.index)
    #df_train=get_train_x(date)
    df_train_x=df_train.iloc[:,-12:].values.T
    df_train_x=preprocessing.scale(df_train_x,axis=1)
    test_x=np.asarray(df_train_x)
    # min_max_scaler = preprocessing.MinMaxScaler()
    # df_train_x = min_max_scaler.fit_transform(df_train_x)
    #df_train_x=preprocessing.normalize(df_train_x,axis=1, norm='l2')
    #print(df_train_x)
    #x_normalize=np.asarray(np.matrix(df_train_x.T/x_norm).T)
    #df_train_y=df_train.iloc[:,-1].values
    print(code_num)
    qwer= np.zeros((code_num))
    for i in range(0, code_num):
        qwer[i] = 1

    test_y=qwer
    #print(test_y)

    output = open(path+'/parameters/%s.pkl'%previous_date, 'rb')
    parameters = pickle.load(output)
    p,prob=predict(test_x,test_y,parameters)
    prob=prob[0]
    b=np.argsort(-prob)
    #print(prob[b])
    #b=sorted(xrange(len(prob)), key=prob.__getitem__)
    price=[]
    j=0
    for i in range(20):
        codes=df_train.ix[b[i],'code']
        print(codes)
        #print(df_train.ix[b[i],'10_days_price_increase'])



    print(j)


def  test():
    engine=create_engine('mysql://root:0325xb@localhost/trade_days_list?charset=utf8')
    trade_days_list=pd.read_sql('trade_days_list',engine)
    trade_days_list=trade_days_list.loc[:,'0'].values.tolist()
    date_index=trade_days_list.index('2017-09-18 00:00:00')
    previous_date=str(trade_days_list[date_index])[0:10]
    date=str(trade_days_list[date_index+10])[0:10]
    path=os.path.abspath('.')
    #os.mkdir(path+'/parameters');
    #df_train.to_pickle(path+'/parameters/%s.pkl'%date)
    if os.access('D:/Python/stock/test/train_data/df_train%s.pkl'%date, os.F_OK):
        df_train=pd.read_pickle('D:/Python/stock/test/train_data/df_train%s.pkl'%date)
    else:
        df_train=get_train_x(date)

    #df_train=get_train_x(date)
    df_train_x=df_train.iloc[:,-13:-1].values.T
    df_train_x=preprocessing.scale(df_train_x,axis=1)
    test_x=np.asarray(df_train_x)
    # min_max_scaler = preprocessing.MinMaxScaler()
    # df_train_x = min_max_scaler.fit_transform(df_train_x)
    #df_train_x=preprocessing.normalize(df_train_x,axis=1, norm='l2')
    #print(df_train_x)
    #x_normalize=np.asarray(np.matrix(df_train_x.T/x_norm).T)
    df_train_y=df_train.iloc[:,-1].values
    print(df_train_y.shape)
    qwer= np.zeros((df_train_y.shape[0]))
    for i in range(0, df_train_y.shape[0]):
        if df_train_y[i] > 2:
            qwer[i] = 1
        else:
            qwer[i] = 0
    test_y=qwer
    #print(test_y)

    output = open(path+'/parameters/%s.pkl'%date, 'rb')
    parameters = pickle.load(output)
    p,prob=predict(test_x,test_y,parameters)
    prob=prob[0]
    b=np.argsort(-prob)
    np.savetxt('f1.csv',prob, delimiter = ',')
    #print(prob[b])
    #b=sorted(xrange(len(prob)), key=prob.__getitem__)
    price=[]
    j=0
    for i in range(20):
        codes=df_train.ix[b[i],'code']
        # print(codes)
        # print(df_train.ix[b[i],'10_days_price_increase'])



    print(j)

# train()
# test()
#predict_future()