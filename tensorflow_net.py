#!/usr/bin/env python


import pandas as pd
import MySQLdb
import tushare as ts
import datetime
import time
import os
import pickle
from sqlalchemy import create_engine
# from WindPy import *
from func_deeplearning import *
from WindPy import *
from sklearn import preprocessing
import tensorflow as tf
import numpy as np
import get_data
n_x = 12    # feature数量
n_h_1 = 1500
n_h_2 = 1500
n_y = 1
#layers_dims = (n_x, n_h, n_y)
#按照高斯分布初始化权重矩阵
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape))

#定义神经网络模型
def model(X, w_h_1, w_h_2,w_o,b1,b_o):
    h1 = tf.nn.relu(tf.matmul(X, w_h_1)+b1) # 激活函数采用sigmoid函数
    #h2 = tf.matmul(h1, w_h_2) # 激活函数采用sigmoid函数
    out=tf.nn.sigmoid(tf.matmul(h1, w_o)+b_o)
    return out # note that we dont take the softmax at the end because our cost fn does that for us


#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)#读取数据
#mnist.train.images是一个55000 * 784维的矩阵, mnist.train.labels是一个55000 * 10维的矩阵
#trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

X = tf.placeholder("float", [None, 12])#创建占位符,在训练时传入图片的向量
Y = tf.placeholder("float")#图像的label用一个10维向量表示

w_h_1 = init_weights([n_x, n_h_1]) # 输入层到隐藏层的权重矩阵,隐藏层包含625个隐藏单元
b1=tf.Variable(tf.random_normal([n_h_1]))
w_h_2 = init_weights([n_h_1, n_h_2]) # 输入层到隐藏层的权重矩阵,隐藏层包含625个隐藏单元
b2=tf.Variable(tf.random_normal([n_h_2]))
w_o = init_weights([n_h_2, n_y])#隐藏层到输出层的权重矩阵
b_o=tf.Variable(tf.random_normal([n_y]))


py_x = model(X, w_h_1,w_h_2,w_o,b1,b_o)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y )) # 计算py_x与Y的交叉熵
#tf.train.GradientDescentOptimizer(0.06).minimize(cost)  #通过步长为0.05的梯度下降算法求参数
train_op =tf.train.AdamOptimizer().minimize(cost)
predict_op = py_x# 预测阶段,返回py_x中值最大的index作为预测结果

# Launch the graph in a session

def train_data(start_train_date):
    engine=create_engine('mysql://root:0325xb@localhost/trade_days_list?charset=utf8')
    trade_days_list=pd.read_sql('trade_days_list',engine)
    trade_days_list=trade_days_list.loc[:,'0'].values.tolist()
    #start_train_date='2017-10-09 00:00:00'
    date_index=trade_days_list.index(start_train_date)
    print(date_index)
    cash=100
    df_predict=pd.DataFrame()
    df_predict=df_predict.reset_index()
    date=str(trade_days_list[date_index])[0:10]
    print(date)
    path=os.path.abspath('.')
    if os.access(path+'/train_data/df_train%s.pkl'%date, os.F_OK):
        df_train=pd.read_pickle(path+'/train_data/df_train%s.pkl'%date)
    else:
        df_train=get_data.get_train_x(date)
    n=len(df_train.index)
    print(n)
    nn=0
    for i in range(n):
        if df_train.ix[i,'10_days_price_increase']>3:
            df_train.ix[n+nn]=df_train.ix[i]
            df_train.ix[n+nn+1]=df_train.ix[i]
            df_train.ix[n+nn+2]=df_train.ix[i]
            nn=nn+3
    print('调整后长度:',len(df_train.index))
    df_train_x=df_train.iloc[:,-13:-1].values.T
    df_train_x=preprocessing.scale(df_train_x,axis=1)
    train_x=np.asarray(df_train_x)
    df_train_y=df_train.iloc[:,-1].values

    qwer= np.zeros((df_train_y.shape[0],1))
    print(qwer.shape)
    for i in range(0, df_train_y.shape[0]):
        if df_train_y[i] > 0.5:
            qwer[i] = 1
        else:
            qwer[i] = 0
    train_y=qwer
    return train_x.T,train_y
'''
def predict():
    #使用新一个时段的训练数据分析上次训练效果
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
    else:
        print('First train:No parameters exist')
    #使用新一个时段的训练数据分析上次训练效果


    ### 定义初始设定 ###
    n_x = 12    # feature数量
    n_h = 35
    n_y = 1
    layers_dims = (n_x, n_h, n_y)
    previous_date=None
    if date_index==trade_days_list.index(start_train_date):
        previous_date=-1
    else:
        previous_date=str(trade_days_list[date_index-10])[0:10]
    #previous_date=-1
    parameters = two_layer_model(train_x, train_y,  previous_date,layers_dims = (n_x, n_h, n_y), num_iterations = 2000, print_cost=True)
    path=os.path.abspath('.')
    # output = open(path+'/parameters/%s.pkl'%date, 'wb')
    # # Pickle dictionary using protocol 0.
    # pickle.dump(parameters, output)
    # output.close()
    date_index=date_index+10
    df_predict.to_csv("dfpredict1.csv",encoding = "GB18030")
'''
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()
    start_train_date='2017-10-09 00:00:00'
    trX,trY=train_data(start_train_date)
    for i in range(1500):
        p,predict=sess.run([train_op,predict_op], feed_dict={X: trX, Y: trY})
        #=sess.run(predict_op, feed_dict={X: trX, Y: trY})
    #print(predict)
    p = np.zeros(len(predict))
    n=0
    for j in range(len(predict)):
        if predict[j]>0.5:
            p[j]=1
            n=n+1
    print(n)
    np.savetxt('f.csv',predict, delimiter = ',')
    #print(trY)
    print(i, np.mean(trY==p))