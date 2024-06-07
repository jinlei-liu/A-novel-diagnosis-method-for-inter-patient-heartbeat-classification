# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import tensorflow.keras
from keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD
from imblearn.over_sampling import BorderlineSMOTE
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
 
config.gpu_options.per_process_gpu_memory_fraction = 0.9
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
#config = tf.compat.v1.ConfigProto(allow_soft_placement=False, log_device_placement=False) 
config.gpu_options.allow_growth=True
#tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

import os
#import numpy as np
import random
from collections import Counter
from sklearn.metrics import confusion_matrix#混淆矩阵
from sklearn.metrics import f1_score
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
from sklearn.utils import class_weight

from tensorflow.keras.utils import plot_model
#from tensorflow.keras.utils import multi_gpu_model
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:1'

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

######################################################################################
data_f_ds2=np.load('/home/lingang/liujinlei/mit_classify/incartdata/data2/x2_test.npy')
data_f_ds2=data_f_ds2.astype(np.float64)
data_f_ds1=np.load('/home/lingang/liujinlei/mit_classify/incartdata/data2/x2_train.npy')
data_f_ds1=data_f_ds1.astype(np.float64)

data_ds1=np.load('/home/lingang/liujinlei/mit_classify/incartdata/data2/x1_train.npy')
data_ds1=np.squeeze(data_ds1, axis=1).astype(np.float64)
label_ds1=np.load('/home/lingang/liujinlei/mit_classify/incartdata/data2/y_train.npy')
label_ds1=label_ds1.astype(np.float64)
print(Counter(label_ds1.reshape(-1)))

data_ds2=np.load('/home/lingang/liujinlei/mit_classify/incartdata/data2/x1_test.npy')
data_ds2=np.squeeze(data_ds2, axis=1).astype(np.float64)
label_ds2=np.load('/home/lingang/liujinlei/mit_classify/incartdata/data2/y_test.npy')
label_ds2=label_ds2.astype(np.float64)
print(Counter(label_ds2.reshape(-1)))

data1_ds1=np.load('/home/lingang/liujinlei/mit_classify/incartdata/data2/x0_train.npy')
data1_ds1=np.squeeze(data1_ds1, axis=1).astype(np.float64)
data1_ds2=np.load('/home/lingang/liujinlei/mit_classify/incartdata/data2/x0_test.npy')
data1_ds2=np.squeeze(data1_ds2, axis=1).astype(np.float64)

#####################################################################################################

data_ds1 = np.expand_dims(data_ds1, axis=2)
data1_ds1 = np.expand_dims(data1_ds1, axis=2)

label_ds1=np_utils.to_categorical(label_ds1,4)  #-----------转化为one-hot标签 四分类

data_ds2 = np.expand_dims(data_ds2, axis=2)
data1_ds2 = np.expand_dims(data1_ds2, axis=2)



label_ds2=np_utils.to_categorical(label_ds2,4)  #-----------转化为one-hot标签 四分类
#label1_ds2=np_utils.to_categorical(label1_ds2,4)
#------打乱数据------

Data_DS1,Data1_DS1,Data_f_DS1,Label_DS1=MITmodel.shuffle_set2(data_ds1,data1_ds1,data_f_ds1,label_ds1)


Data_DS2=data_ds2
Data_250DS2=data1_ds2

Label_DS2=label_ds2
Data_f_DS2=data_f_ds2

#Data_DS2,Label_DS2=Pmodel.shuffle_set(data_ds2,label_ds2)

#---------------------5折交叉验证训练和测试------
folds=5
jiange=int(Data_DS1.shape[0]/folds)
Con_Matr=[]  #存储每一折的混淆矩阵
F1=[]        #存储每一折的f1
Acc=[]        #存储每一折的acc
Loss=[]      #存储每一折的loss
#for i in range(1,6):  
for i in range(1,2):

    X_train=Data_DS1
    X_250train=Data1_DS1
    X_train_f=Data_f_DS1
    y_train=Label_DS1
    inputs1=Input(shape=(180, 1 ))
    inputs2=Input(shape=(180, 1 ))
    inputs4=Input(shape=(9,  ))   

    model = MITmodel.model360_revised_2(inputs1,inputs2,inputs4)

    adam = Adam(lr=1e-3)
    
#    plot_model(model, to_file='model_3.png')
    
    model.summary()
    model.compile(
#                  loss='categorical_crossentropy',
                  loss=categorical_focal_loss(gamma=2),
#                  optimizer='rmsprop',
                  optimizer= adam,
#                  optimizer=SGD(lr=0.01, decay=0.001, momentum=0.99, nesterov=True),
                  metrics=['categorical_accuracy']
                  )
#    filepath="/home/lingang/liujinlei/mit_classify/modelfiles/classification_20211111_512batch.hdf5"#保存模型的路径
    filepath="/home/lingang/liujinlei/mit_classify/incartdata/model20230206/model_{epoch:02d}-{val_categorical_accuracy:.2f}.h5"
    checkpoint = ModelCheckpoint(filepath, verbose=1,
                              monitor='val_categorical_accuracy', 
                             save_weights_only='false',period=1)
    
    callback_lists = [checkpoint]

    classweight = {0:1, 1:50.094105025050613, 2:2, 3:1.9151821908106785}
#    classweight = {0:1, 1:4.094105025050613, 2:1, 3:2.9151821908106785}
#    classweight = {0:1, 1:6, 2:1, 3:3}
#    classweight = 'auto',

    history = model.fit([X_train,X_250train,X_train_f],y_train,validation_data=([Data_DS2,Data_250DS2,Data_f_DS2],Label_DS2),class_weight =classweight,
                    callbacks=callback_lists,epochs=200,batch_size=1024) 
#    history = model.fit([X_train,X_250train],y_train,validation_data=([Data_DS2,Data_250DS2],Label_DS2),class_weight =classweight,
#                    callbacks=callback_lists,epochs=100,batch_size=1024) 

    ## loss曲线
#    pyplot.plot(history.history['loss'], label='Training loss')
#    pyplot.plot(history.history['val_loss'], label='Validation loss')
#    pyplot.legend()
#    pyplot.xlabel(u'Epochs')
#    pyplot.ylabel(u'Loss')
#    pyplot.savefig('/home/lingang/liujinlei/QT_train/loss.svg', dpi=600 )
#    pyplot.show()
#    
#    loss,accuracy=model.evaluate([Data_DS2,Data_250DS2,Data_f_DS2],Label_DS2)

##----------------------------------------总体评估----------------------------------------
#print("%.2f%% (+/- %.2f%%)" % (np.mean(F1), np.std(F1)))
#print("%.2f%% (+/- %.2f%%)" % (np.mean(Acc), np.std(Acc)))









