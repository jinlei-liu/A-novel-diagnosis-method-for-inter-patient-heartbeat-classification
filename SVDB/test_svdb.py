#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 19:36:47 2021

@author: lingang
"""
import tensorflow as tf
import numpy as np
import tensorflow.keras
from keras import backend as K
import glob
from tensorflow.keras.optimizers import Adam,SGD
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
 
config.gpu_options.per_process_gpu_memory_fraction = 0.7
#tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
#config.gpu_options.allow_growth=True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
import time
import os
#import numpy as np
import random
from sklearn.metrics import confusion_matrix#混淆矩阵
from sklearn.metrics import f1_score,accuracy_score
from sklearn.preprocessing import scale
from tensorflow.keras.layers import Input
from tensorflow.keras  import optimizers
from keras.utils import np_utils
import MITmodel 
from matplotlib import pyplot
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler  #选择保留最佳训练模型
from sklearn.preprocessing import MinMaxScaler, scale
from sklearn import metrics#模型评估
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model

from sklearn.manifold import TSNE

def categorical_focal_loss(gamma=2):
    """
        Categorical form of focal loss.
            FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
        References:
            https://arxiv.org/pdf/1708.02002.pdf
        Usage:
            model.compile(loss=categorical_focal_loss(gamma=2), optimizer="adam", metrics=["accuracy"])
            model.fit(class_weight={0:alpha0, 1:alpha1, ...}, ...)
        Notes:
           1. The alpha variable is the class_weight of keras.fit, so in implementation of the focal loss function
           we needn't define this variable.
           2. (important!!!) The output of the loss is the loss value of each training sample, not the total or average
            loss of each batch.
    """

    def focal_loss(y_true, y_pred):
        y_pred = K.constant(y_pred) if not K.is_tensor(y_pred) else y_pred
        y_true = K.cast(y_true, y_pred.dtype)

        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

        return K.sum(-y_true * K.pow(1 - y_pred, gamma) * K.log(y_pred), axis=-1)

    return focal_loss
start = time.time() #开始时间

data_f_ds2=np.load('/home/lingang/liujinlei/mit_classify/SVDB/data2/x2_test.npy')#2
data_f_ds2=data_f_ds2.astype(np.float64)

#data_f_ds1=np.load('/home/lingang/liujinlei/ECG-Classification-Using-CNN-and-CWT-master/data3/x2_train.npy')
#data_f_ds1=data_f_ds1.astype(np.float64)

#data_ds1=np.load('/home/lingang/liujinlei/ECG-Classification-Using-CNN-and-CWT-master/data3/x1_train.npy')
#data_ds1=np.squeeze(data_ds1, axis=1).astype(np.float64)
#label_ds1=np.load('/home/lingang/liujinlei/ECG-Classification-Using-CNN-and-CWT-master/data3/y_train.npy')
#label_ds1=label_ds1.astype(np.float64)
data_ds2=np.load('/home/lingang/liujinlei/mit_classify/SVDB/data2/x1_test.npy')#1
data_ds2=np.squeeze(data_ds2, axis=1).astype(np.float64)

#data111=data_ds2

label_ds2=np.load('/home/lingang/liujinlei/mit_classify/SVDB/data2/y_test.npy')#y
label_ds2=label_ds2.astype(np.float64)

#data1_ds1=np.load('/home/lingang/liujinlei/ECG-Classification-Using-CNN-and-CWT-master/data3/x0_train.npy')
#data1_ds1=np.squeeze(data1_ds1, axis=1).astype(np.float64)

#label1_ds1=np.load('/home/lingang/liujinlei/ECG-Classification-Using-CNN-and-CWT-master/data2/y_train.npy')
#label1_ds1=label1_ds1.astype(np.float64)
data1_ds2=np.load('/home/lingang/liujinlei/mit_classify/SVDB/data2/x0_test.npy')#x0
data1_ds2=np.squeeze(data1_ds2, axis=1).astype(np.float64)

#label1_ds2=np.load('/home/lingang/liujinlei/ECG-Classification-Using-CNN-and-CWT-master/data3/y_test.npy')

#label1_ds2=label1_ds2.astype(str)

#data_f_ds2=np.load('/home/lingang/liujinlei/mit_classify/data13/Data_f_DS2.npy')
#data_f_ds1=np.load('/home/lingang/liujinlei/mit_classify/data13/Data_f_DS1.npy')
#
#data_ds1=np.load('/home/lingang/liujinlei/mit_classify/data13/Data_DS1.npy')
#label_ds1=np.load('/home/lingang/liujinlei/mit_classify/data13/Label_DS1.npy')
#data_ds2=np.load('/home/lingang/liujinlei/mit_classify/data13/Data_DS2.npy')
#label_ds2=np.load('/home/lingang/liujinlei/mit_classify/data13/Label_DS2.npy')
#
#data1_ds1=np.load('/home/lingang/liujinlei/mit_classify/data13/Data1_DS1.npy')
#label1_ds1=np.load('/home/lingang/liujinlei/mit_classify/data13/Label1_DS1.npy')
#data1_ds2=np.load('/home/lingang/liujinlei/mit_classify/data13/Data1_DS2.npy')
#label1_ds2=np.load('/home/lingang/liujinlei/mit_classify/data13/Label1_DS2.npy')

#data_f_ds2=np.load('/home/lingang/liujinlei/mit_classify/data11/Data_f_DS2.npy')
#data_f_ds1=np.load('/home/lingang/liujinlei/mit_classify/data11/Data_f_DS1.npy')
#
#data_ds1=np.load('/home/lingang/liujinlei/mit_classify/data11/Data_DS1.npy')
#label_ds1=np.load('/home/lingang/liujinlei/mit_classify/data11/Label_DS1.npy')
#data_ds2=np.load('/home/lingang/liujinlei/mit_classify/data11/Data_DS2.npy')
#label_ds2=np.load('/home/lingang/liujinlei/mit_classify/data11/Label_DS2.npy')
#
#data1_ds1=np.load('/home/lingang/liujinlei/mit_classify/data11/Data1_DS1.npy')
#label1_ds1=np.load('/home/lingang/liujinlei/mit_classify/data11/Label1_DS1.npy')
#data1_ds2=np.load('/home/lingang/liujinlei/mit_classify/data11/Data1_DS2.npy')
#label1_ds2=np.load('/home/lingang/liujinlei/mit_classify/data11/Label1_DS2.npy')
#----------------------导入连续2.5S数据------------

#data_ds1 = np.expand_dims(data_ds1, axis=2)

#data_250ds1 = np.expand_dims(data_250ds1, axis=2)
#不用扩展特征
#data_f_ds1 = np.expand_dims(data_f_ds1, axis=2)

#label_ds1=np_utils.to_categorical(label_ds1,4)  #-----------转化为one-hot标签 四分类

data1_ds2=np.expand_dims(data1_ds2, axis=2)
data_ds2 = np.expand_dims(data_ds2, axis=2)
#data_250ds2 = np.expand_dims(data_250ds2, axis=2)
#
#data_f_ds2 = np.expand_dims(data_f_ds2, axis=2)

label_ds2=np_utils.to_categorical(label_ds2,4)  #-----------转化为one-hot标签 四分类

##------打乱数据------
#Data_DS1,Data_f_DS1,Label_DS1=Pmodel.shuffle_set1(data_ds1,data_f_ds1,label_ds1)
#Data_DS1,Label_DS1=Pmodel.shuffle_set(data_ds1,label_ds1)
Data1_DS2=data1_ds2
Data_DS2=data_ds2

#data1_ds1=np.expand_dims(data1_ds1, axis=2)

#data_ds1 = np.expand_dims(data_ds1, axis=2)

#Data1_DS1=data1_ds1
#Data_DS1=data_ds1
#Label_DS1=label_ds1
#Data_f_DS1=data_f_ds1


Label_DS2=label_ds2
Data_f_DS2=data_f_ds2

Con_Matr=[]  #存储每一折的混淆矩阵
F1=[]        #存储每一折的f1
Acc=[]        #存储每一折的acc
Loss=[]
SS_SEN_SEG=[]
#panth='/home/lingang/liujinlei/mit_classify/model20220907/'
#panth='/home/lingang/liujinlei/mit_classify/proposedmodel/'
#panth='/home/lingang/liujinlei/mit_classify/test20220907_2/'
panth='/home/lingang/liujinlei/mit_classify/SVDB/modelsaved/'
#panth='/home/lingang/liujinlei/mit_classify/test20220907_2_2/cross_loss/'
file = glob.glob(panth+'*.h5')

for f in range(len(file)):
#    inputs1=Input(shape=(200,1 ))
    inputs1=Input(shape=(90,1 ))
#    inputs3=Input(shape=(360,1))
    inputs2=Input(shape=(90, 1))
    inputs4=Input(shape=(9, ))
#    inputs4=Input(shape=(8,  ))

#model20220907_1    without features
#model20220907_2    main
#model20220907_3    SE---->CNN
#model20220907_4    one input+features

#    model = MITmodel.model360_1(inputs1,inputs2)
#    model = Pmodel.model(inputs1,inputs2,inputs3,inputs4)
#    model = MITmodel.model360(inputs1,inputs2,inputs4)
#    model = MITmodel.model360_revised_1(inputs1,inputs2)
    model = MITmodel.model360_revised_2(inputs1,inputs2,inputs4)
#    model = MITmodel.model360_revised_4(inputs1,inputs4)
#    plot_model(model, to_file='model_3.png')
#    model.summary()
    adam = Adam(lr=1e-3)
    model.load_weights(file[f])
    model.compile(
#        loss='categorical_crossentropy',
        loss=categorical_focal_loss(gamma=2),
#                  optimizer='rmsprop',
#        optimizer=SGD(lr=0.01, decay=0.001, momentum=0.99, nesterov=False),
        optimizer=adam,
        metrics=['categorical_accuracy']
                  )
    print('\ntesting.....'+str(f))

    #Evaluate the model with the metrics  we defined earlier
#    loss,accuracy=model.evaluate([X_test,X_test,X_test],y_test)
    #loss,accuracy=model.evaluate([Data_DS2,Data1_DS2,Data_f_DS2],Label_DS2)
#    loss,accuracy=model.evaluate([Data_DS2,Data1_DS2],Label_DS2)
#    loss,accuracy=model.evaluate([Data_DS2,Data_DS2,Data_DS2],Label_DS2)
#    loss,accuracy=model.evaluate(Data_DS2,Label_DS2)
    #Acc.append(accuracy)
    #Loss.append(loss)
    y_pred_4 = model.predict([Data_DS2,Data1_DS2,Data_f_DS2])
#    y_pred_4 = model.predict([Data_DS2,Data_f_DS2])
 #####################################################################################################################   
#    y_pred_4, feature=model.predict([Data_DS2,Data1_DS2,Data_f_DS2])
####    feature=data111
#    print(y_pred_4.shape)
#    print(feature.shape)
##    
##    feature = np.reshape(feature,(49507,-1))
##    print(feature.shape)
###    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, random_state=0)
###    tsne = TSNE(perplexity=30, n_components=2,  init='pca',metric='cosine', learning_rate=200,random_state=0)
###    plot_only = 500
#    
#    
#    rows = np.arange(49507)
#    np.random.shuffle(rows)
#    n_select = 10000
#    
#    tsne = TSNE(n_components=2)
##    tsne = TSNE(perplexity=30, n_components=2,  init='pca',metric='cosine', random_state=0)
#    
#    X = tsne.fit_transform(feature[rows[:n_select],:])
#    
#    import seaborn as sns
#    
#    palette = sns.color_palette("bright", 10)
#    
#
#    
##    label1_ds2=label1_ds2.replace('0', 'N')
##    label1_ds2=label1_ds2.replace('1', 'S')
##    label1_ds2=label1_ds2.replace('2', 'V')
##    label1_ds2=label1_ds2.replace('3', 'F')
#    
##    sns.scatterplot(x= X[:,0],y = X[:,1], hue=label1_ds2, legend='full', palette=palette)
#    
#    sns.scatterplot(x= X[:,0],y = X[:,1],hue=label1_ds2[rows[:n_select]], legend='full', palette=palette)
#    plt.show()
###################################################################################################################################    
#    y_pred_4=model.predict([Data_DS2,Data1_DS2])
#    y_pred_4=model.predict([Data_DS2,Data_DS2,Data_DS2])
#    y_pred_4=model.predict(Data_DS2)
    #f1_score和confusion_matrix不支持one_hot，只支持普通标签
    
    y_test=np.argmax(Label_DS2,axis=1)
    y_pred=np.argmax(y_pred_4,axis=1)
    f1=metrics.f1_score(y_test, y_pred, average='macro')
    F1.append(f1)
    
    acc=accuracy_score(y_test, y_pred)
    Acc.append(acc)
    
    con_matr=confusion_matrix(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    
#    NN_SEN = con_matr[0][0]/(con_matr[0][0]+con_matr[0][1]+con_matr[0][2]+con_matr[0][3])
#    NN_ppv = con_matr[0][0]/(con_matr[0][0]+con_matr[1][0]+con_matr[2][0]+con_matr[3][0])
#    
#    SS_SEN = con_matr[1][1]/(con_matr[1][1]+con_matr[1][0]+con_matr[1][2]+con_matr[1][3])
#    SS_ppv = con_matr[1][1]/(con_matr[1][1]+con_matr[0][1]+con_matr[2][1]+con_matr[3][1])
#    
#    VV_SEN = con_matr[2][2]/(con_matr[2][2]+con_matr[2][0]+con_matr[2][1]+con_matr[2][3])
#    VV_ppv = con_matr[2][2]/(con_matr[2][2]+con_matr[0][2]+con_matr[1][2]+con_matr[3][2])
#    
#    FF_SEN = con_matr[3][3]/(con_matr[3][3]+con_matr[3][0]+con_matr[3][1]+con_matr[3][2])
#    FF_ppv = con_matr[3][3]/(con_matr[3][3]+con_matr[0][3]+con_matr[1][3]+con_matr[2][3])
    
#    SS_SEN_SEG.append(SS_SEN)
    Con_Matr.append(con_matr)
end = time.time() #结束时间
print("run: %f s" % (end - start)) #输出用时