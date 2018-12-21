# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 21:08:05 2018

@author: lj
"""
import numpy as np
import tensorflow as tf
import random

tf.reset_default_graph() ##用于清除默认图形堆栈并重置全局默认图形

def build_data(feature_len):  #构造2000个序列长度为n的正弦序列，前1500个作为训练集，后500个作为测试集
    xs=[]
    ys=[]
    for i in range(2000):
        k=random.uniform(1,50)
        x=[[np.sin(k + j)] for j in range(0,feature_len)]
        y=[np.sin(k + feature_len)]
        xs.append(x)
        ys.append(y)
    train_x=np.array(xs[0:1500])
    train_y=np.array(ys[0:1500])
    test_x=np.array(xs[1500:])
    test_y=np.array(ys[1500:])
    return train_x,train_y,test_x,test_y

class RNN_Sine():
    '''定义RNN_Sine类
    '''
    def __init__(self,n_steps,n_inputs,batch_size,hidden_size,input_train_feature,input_train_label,input_test_feature,input_test_label,n_output):
        self.batch_size = batch_size # RNN模型中RNNcell的个数
        self.hidden_size = hidden_size # RNN cell 隐含层节点数
        self.n_steps = n_steps #RNN每个节点输入特征的长度
        self.n_inputs = n_inputs #特征中每个元素的长度
        self.input_train_feature = input_train_feature #数据特征
        self.input_train_label = input_train_label #数据标签
        self.input_test_feature = input_test_feature
        self.input_test_label = input_test_label
        self.n_output = n_output #CTC层输出节点数
    
    def seq_predict_model(self,X,W,b):
        ## 初始化RNN单元
        X = tf.transpose(X,[1,0,2])  ##交换batch_size和n_steps
        X = tf.reshape(X,[-1,self.n_inputs]) ##(n_steps*batch_size,n_inputs)  
        X = tf.split(X,self.n_steps,0)  ## n_steps * (batch_size, n_inputs)
        rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units = self.batch_size)
        init_state = rnn_cell.zero_state(self.batch_size,dtype = tf.float32)
        outputs,states = tf.contrib.rnn.static_rnn(rnn_cell,X,initial_state = init_state) ## outputs -- [n_steps,batch_size,hidden_size]
        y_pred = tf.matmul(outputs[-1],W) + b
        return y_pred
        
    def fit(self):
        '''RNN模型训练
        '''
        #1.声明输入输出的占位符
        X = tf.placeholder('float',[None,self.n_steps,self.n_inputs],name = 'X')
        Y = tf.placeholder('float',[None,self.n_output],name = 'Y')
        # 2.输出层参数
        W = tf.Variable(tf.random_normal([self.hidden_size,self.n_output]))
        b = tf.Variable(tf.random_normal([self.n_output]))
        # 3.构造RNN计算图
        y_pred = self.seq_predict_model(X,W,b)
        # 4.声明代价函数和优化算法
        loss = tf.sqrt(tf.square(tf.subtract(Y,y_pred)))
        train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
        # 5.构造迭代过程并预测结果
        trX = self.input_train_feature
        trY = self.input_train_label
        teX = self.input_test_feature
        teY = self.input_test_label
        
        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            for i in range(50):
                for end in range(batch_size,len(trX),batch_size):
                    begin = end - batch_size
                    x_value = trX[begin:end]
                    y_value = trY[begin:end]
                    #通过session.run接口触发执行
                    sess.run(train_op,feed_dict={X:x_value,Y:y_value})
                    test_indices=np.arange(len(teX)) #在训练的过程中开始测试
                    np.random.shuffle(test_indices)
                    test_indices=test_indices[0:self.batch_size]
                    x_value=teX[test_indices]
                    y_value=teY[test_indices]
                    val_loss=np.mean(sess.run(loss,feed_dict={X:x_value,Y:y_value}))    #使用均方差作为代价函数
                print ('Run %s'%i,val_loss)

                    
if __name__ == '__main__':
    print('--------------------1. Parameters Setting--------------------')
    n_steps = 10 ##每个RNN cell输入特征长度
    n_inputs = 1 ##特征中每个元素的大小
    batch_size = 10  ##rnn网络的节点数
    hidden_size = 10
    n_output = 1       
    print('-----------------------2. Load Data---------------------------')
    input_train_feature,input_train_label,input_test_feature,input_test_label = build_data(n_steps)
    print('-------------------3.Model Training------------------')
    rnn = RNN_Sine(n_steps,n_inputs,batch_size,hidden_size,input_train_feature,input_train_label,input_test_feature,input_test_label,n_output)
    rnn.fit()
