# -*- coding: utf-8 -*-

import numpy as np
from tensorflow.keras import backend as K
#from keras.layers import MaxPooling1D,AveragePooling1D,Conv1D,MaxPool1D,add,Flatten,Dense,Concatenate,Activation
from tensorflow.keras.layers import Bidirectional,GlobalAveragePooling1D,BatchNormalization,Add,LSTM, Dropout,MaxPooling1D,AveragePooling1D,Conv1D,MaxPool1D,add,Flatten,Dense,Concatenate,Activation
from tensorflow.keras.layers import GlobalAveragePooling1D,Multiply
#from tf.compat.v1.keras.layers  CuDNNLSTM
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input,Layer
import random
from tensorflow.keras.models import Sequential,load_model,Model
from attention import Attention
import tensorflow as tf


#without FEATURES	
def model360_revised_1(inputs1,inputs2):
    #11111111111111111111
    conv11 = Conv1D(filters=64,kernel_size=8,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(inputs1)
    conv11=BatchNormalization()(conv11)
    conv11=Activation('relu')(conv11)
#    conv111=conv11

    conv12 = Conv1D(filters=64,kernel_size=5,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv11)
    conv12=BatchNormalization()(conv12)
    conv12=Activation('relu')(conv12)
    
    conv13 = Conv1D(filters=64,kernel_size=3,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv12)
    conv13=BatchNormalization()(conv13)
    conv13=Activation('relu')(conv13)
    
    shortcut= Conv1D(filters=64,kernel_size=1,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(inputs1)
    shortcut=BatchNormalization()(shortcut)

    se_net1=SEBlock(conv13)
    block_1= Add()([shortcut, se_net1]) 
    block_1=Activation('relu')(block_1)   
#    block_1 = Dropout(0.2)(block_1)

#    22222222222222222222222222
    conv11 = Conv1D(filters=128,kernel_size=8,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(block_1)
    conv11=BatchNormalization()(conv11)
    conv11=Activation('relu')(conv11)

    
    conv12 = Conv1D(filters=128,kernel_size=5,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv11)   
    conv12=BatchNormalization()(conv12)
    conv12=Activation('relu')(conv12)
    
    conv13 = Conv1D(filters=128,kernel_size=3,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv12)
    conv13=BatchNormalization()(conv13)
    conv13=Activation('relu')(conv13)

    
    shortcut= Conv1D(filters=128,kernel_size=1,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(block_1)
    shortcut=BatchNormalization()(shortcut)
    
    se_net=SEBlock(conv13)
    block_1= Add()([shortcut, se_net]) 
    block_1=Activation('relu')(block_1)
    
#    block_1 = Dropout(0.2)(block_1)
     #333333333333333333333333
    conv11 = Conv1D(filters=128,kernel_size=8,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(block_1)
    conv11=BatchNormalization()(conv11)
    conv11=Activation('relu')(conv11)
#    pool11=MaxPool1D(pool_size=2)(conv11)
    conv12 = Conv1D(filters=128,kernel_size=5,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv11)
    conv12=BatchNormalization()(conv12)
    conv12=Activation('relu')(conv12)

    
    conv13 = Conv1D(filters=128,kernel_size=3,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv12)
    conv13=BatchNormalization()(conv13)
    conv13=Activation('relu')(conv13)
#    
#    pool13=MaxPool1D(pool_size=2)(conv13)

#    shortcut= Conv1D(filters=128,kernel_size=1,kernel_initializer="he_uniform",strides=2,padding='same',
#                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(block_1)
    shortcut=BatchNormalization()(block_1)
    
    se_net=SEBlock(conv13)
    block_1= Add()([shortcut, se_net])
 
    block_1=Activation('relu')(block_1)  

 ########################################################################################################################################### 
 #11111111111111111111
    conv11 = Conv1D(filters=64,kernel_size=8,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(inputs2)
    conv11=BatchNormalization()(conv11)
    conv11=Activation('relu')(conv11)

    conv12 = Conv1D(filters=64,kernel_size=5,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv11)
    conv12=BatchNormalization()(conv12)
    conv12=Activation('relu')(conv12)
    
    conv13 = Conv1D(filters=64,kernel_size=3,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv12)
    conv13=BatchNormalization()(conv13)
    conv13=Activation('relu')(conv13)
    
    shortcut= Conv1D(filters=64,kernel_size=1,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(inputs2)
    shortcut=BatchNormalization()(shortcut)

    se_net=SEBlock(conv13)
    block_2= Add()([shortcut, se_net]) 
    block_2=Activation('relu')(block_2)   

#    22222222222222222222222222
    conv11 = Conv1D(filters=128,kernel_size=8,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(block_2)
    conv11=BatchNormalization()(conv11)
    conv11=Activation('relu')(conv11)

    
    conv12 = Conv1D(filters=128,kernel_size=5,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv11)
    conv12=BatchNormalization()(conv12)
    conv12=Activation('relu')(conv12)
    
    conv13 = Conv1D(filters=128,kernel_size=3,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv12)
    conv13=BatchNormalization()(conv13)
    conv13=Activation('relu')(conv13)
   #    90*64
#    pool13=MaxPool1D(pool_size=3)(conv13)

    
    shortcut= Conv1D(filters=128,kernel_size=1,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(block_2)
    shortcut=BatchNormalization()(shortcut)
    
    se_net=SEBlock(conv13)
    block_2= Add()([shortcut, se_net]) 
    block_2=Activation('relu')(block_2)
    
    
#    block_1 = Dropout(0.2)(block_1)
     #333333333333333333333333
    conv11 = Conv1D(filters=128,kernel_size=8,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(block_2)
    conv11=BatchNormalization()(conv11)
#    45*16
    conv11=Activation('relu')(conv11)
#    pool11=MaxPool1D(pool_size=2)(conv11)
    conv12 = Conv1D(filters=128,kernel_size=5,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv11)
    conv12=BatchNormalization()(conv12)
    #    90*32
    conv12=Activation('relu')(conv12)
    
#    pool12=MaxPool1D(pool_size=2)(conv12)#################################################
    
    conv13 = Conv1D(filters=128,kernel_size=3,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv12)
    conv13=BatchNormalization()(conv13)
    conv13=Activation('relu')(conv13)

#    
#    pool13=MaxPool1D(pool_size=3)(conv13)

#    shortcut= Conv1D(filters=128,kernel_size=1,kernel_initializer="he_uniform",strides=2,padding='same',
#                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(block_1)
    
    shortcut=BatchNormalization()(block_2)
    
    se_net=SEBlock(conv13)
    block_2= Add()([shortcut, se_net])
    block_2=Activation('relu')(block_2)
    
    
#    block_1 = Dropout(0.2)(block_1)
    #444444444444444444444444444444


#####################################################################################################################
#    gap_layer=Flatten()(inputs4)
#    Dense2=Concatenate(axis=1)([Dense1,inputs4])
 
    Dense1=Concatenate(axis=1)([block_1,block_2])
    
#    LSTM layer  CuDNNLSTM
#    gap_layer=Bidirectional(tf.compat.v1.keras.layers.CuDNNLSTM(64,return_sequences=True))(Dense2)
#    gap_layer=tf.compat.v1.keras.layers.CuDNNLSTM(32,return_sequences=True)(Dense1)
#    gap_layer = Dropout(0.5)(gap_layer)
    
#    gap_layer=attention()(gap_layer)
    
    gap_layer1=Flatten()(Dense1)
    
    
#    # split for attention
#    attention_data = tf.keras.layers.Lambda(lambda x: x[:, :, :32])(Dense2)
#    attention_softmax = tf.keras.layers.Lambda(lambda x: x[:, :, 32:])(Dense2)
#    # attention mechanism
#    attention_softmax = tf.keras.layers.Softmax()(attention_softmax)
#    multiply_layer = tf.keras.layers.Multiply()([attention_softmax, attention_data])
    
#    gap_layer=Flatten()(Dense2)
    
    Dense2=Dense(800)(gap_layer1)
    
    Dense2=Activation('relu')(Dense2)
    
    Dense2 = Dropout(0.5)(Dense2)
    
    Dense3=Dense(50)(Dense2)
#    Dense2=Dense(32)(gap_layer)
    
    Dense3=Activation('relu')(Dense3)
    
    Dense3 = Dropout(0.5)(Dense3)
    

    
    output_layer = Dense(4, activation='softmax')(Dense3)

#    res = Model(inputs=[inputs1,inputs2], outputs=[output_layer, Dense3], name="model360")
    res = Model(inputs=[inputs1,inputs2], outputs=[output_layer], name="model360")
    return res


#without LSTM
def model360_revised_2(inputs1,inputs2,inputs4):
    #11111111111111111111
    conv11 = Conv1D(filters=64,kernel_size=8,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(inputs1)
    conv11=BatchNormalization()(conv11)
    conv11=Activation('relu')(conv11)
#    conv111=conv11

    conv12 = Conv1D(filters=64,kernel_size=5,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv11)
    conv12=BatchNormalization()(conv12)
    conv12=Activation('relu')(conv12)
    
    conv13 = Conv1D(filters=64,kernel_size=3,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv12)
    conv13=BatchNormalization()(conv13)
    conv13=Activation('relu')(conv13)
    
    shortcut= Conv1D(filters=64,kernel_size=1,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(inputs1)
    shortcut=BatchNormalization()(shortcut)

    se_net1=SEBlock(conv13)
    block_1= Add()([shortcut, se_net1]) 
    block_1=Activation('relu')(block_1)   
#    block_1 = Dropout(0.2)(block_1)

#    22222222222222222222222222
    conv11 = Conv1D(filters=128,kernel_size=8,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(block_1)
    conv11=BatchNormalization()(conv11)
    conv11=Activation('relu')(conv11)

    
    conv12 = Conv1D(filters=128,kernel_size=5,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv11)   
    conv12=BatchNormalization()(conv12)
    conv12=Activation('relu')(conv12)
    
    conv13 = Conv1D(filters=128,kernel_size=3,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv12)
    conv13=BatchNormalization()(conv13)
    conv13=Activation('relu')(conv13)

    
    shortcut= Conv1D(filters=128,kernel_size=1,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(block_1)
    shortcut=BatchNormalization()(shortcut)
    
    se_net=SEBlock(conv13)
    block_1= Add()([shortcut, se_net]) 
    block_1=Activation('relu')(block_1)
    
#    block_1 = Dropout(0.2)(block_1)
     #333333333333333333333333
    conv11 = Conv1D(filters=128,kernel_size=8,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(block_1)
    conv11=BatchNormalization()(conv11)
    conv11=Activation('relu')(conv11)
#    pool11=MaxPool1D(pool_size=2)(conv11)
    conv12 = Conv1D(filters=128,kernel_size=5,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv11)
    conv12=BatchNormalization()(conv12)
    conv12=Activation('relu')(conv12)

    
    conv13 = Conv1D(filters=128,kernel_size=3,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv12)
    conv13=BatchNormalization()(conv13)
    conv13=Activation('relu')(conv13)
#    
#    pool13=MaxPool1D(pool_size=2)(conv13)

#    shortcut= Conv1D(filters=128,kernel_size=1,kernel_initializer="he_uniform",strides=2,padding='same',
#                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(block_1)
    shortcut=BatchNormalization()(block_1)
    
    se_net=SEBlock(conv13)
    block_1= Add()([shortcut, se_net])
 
    block_1=Activation('relu')(block_1)  



 ########################################################################################################################################### 
 #11111111111111111111
    conv11 = Conv1D(filters=64,kernel_size=8,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(inputs2)
    conv11=BatchNormalization()(conv11)
    conv11=Activation('relu')(conv11)

    conv12 = Conv1D(filters=64,kernel_size=5,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv11)
    conv12=BatchNormalization()(conv12)
    conv12=Activation('relu')(conv12)
    
    conv13 = Conv1D(filters=64,kernel_size=3,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv12)
    conv13=BatchNormalization()(conv13)
    conv13=Activation('relu')(conv13)
    
    shortcut= Conv1D(filters=64,kernel_size=1,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(inputs2)
    shortcut=BatchNormalization()(shortcut)

    se_net=SEBlock(conv13)
    block_2= Add()([shortcut, se_net]) 
    block_2=Activation('relu')(block_2)   

#    22222222222222222222222222
    conv11 = Conv1D(filters=128,kernel_size=8,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(block_2)
    conv11=BatchNormalization()(conv11)
    conv11=Activation('relu')(conv11)

    
    conv12 = Conv1D(filters=128,kernel_size=5,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv11)
    conv12=BatchNormalization()(conv12)
    conv12=Activation('relu')(conv12)
    
    conv13 = Conv1D(filters=128,kernel_size=3,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv12)
    conv13=BatchNormalization()(conv13)
    conv13=Activation('relu')(conv13)
   #    90*64
#    pool13=MaxPool1D(pool_size=3)(conv13)

    
    shortcut= Conv1D(filters=128,kernel_size=1,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(block_2)
    shortcut=BatchNormalization()(shortcut)
    
    se_net=SEBlock(conv13)
    block_2= Add()([shortcut, se_net]) 
    block_2=Activation('relu')(block_2)
    
    
#    block_1 = Dropout(0.2)(block_1)
     #333333333333333333333333
    conv11 = Conv1D(filters=128,kernel_size=8,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(block_2)
    conv11=BatchNormalization()(conv11)
#    45*16
    conv11=Activation('relu')(conv11)
#    pool11=MaxPool1D(pool_size=2)(conv11)
    conv12 = Conv1D(filters=128,kernel_size=5,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv11)
    conv12=BatchNormalization()(conv12)
    #    90*32
    conv12=Activation('relu')(conv12)
    
#    pool12=MaxPool1D(pool_size=2)(conv12)#################################################
    
    conv13 = Conv1D(filters=128,kernel_size=3,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv12)
    conv13=BatchNormalization()(conv13)
    conv13=Activation('relu')(conv13)


    
    shortcut=BatchNormalization()(block_2)
    
    se_net=SEBlock(conv13)
    block_2= Add()([shortcut, se_net])
    block_2=Activation('relu')(block_2)
    
    
#    block_1 = Dropout(0.2)(block_1)
    #444444444444444444444444444444


#####################################################################################################################
#    gap_layer=Flatten()(inputs4)
#    Dense2=Concatenate(axis=1)([Dense1,inputs4])
 
    Dense1=Concatenate(axis=1)([block_1,block_2])
    

    
    gap_layer1=Flatten()(Dense1)

    
#    gap_layer=Flatten()(Dense2)
    
    Dense2=Dense(800)(gap_layer1)
    
    Dense2=Activation('relu')(Dense2)
    
    Dense2 = Dropout(0.5)(Dense2)
    
    Dense3=Dense(50)(Dense2)
#    Dense2=Dense(32)(gap_layer)
    
    Dense3=Activation('relu')(Dense3)
    
    Dense3 = Dropout(0.5)(Dense3)
    

    Dense4 = Concatenate(axis=1)([Dense3, inputs4])

    
    output_layer = Dense(4, activation='softmax')(Dense4)
    
#    res = Model(inputs=[inputs1,inputs4], outputs=[output_layer], name="model360")
    #################################TSNE##################
#    res = Model(inputs=[inputs1,inputs2,inputs4], outputs=[output_layer, Dense4], name="model360")
    res = Model(inputs=[inputs1,inputs2,inputs4], outputs=[output_layer], name="model360")
    return res
    
#SE->cnn
def model360_revised_3(inputs1,inputs2,inputs4):
    #11111111111111111111
    conv11 = Conv1D(filters=64,kernel_size=8,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(inputs1)
    conv11=BatchNormalization()(conv11)
    conv11=Activation('relu')(conv11)
#    conv111=conv11

    conv12 = Conv1D(filters=64,kernel_size=5,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv11)
    conv12=BatchNormalization()(conv12)
    conv12=Activation('relu')(conv12)
    
    conv13 = Conv1D(filters=64,kernel_size=3,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv12)
    conv13=BatchNormalization()(conv13)
    conv13=Activation('relu')(conv13)
    
    shortcut= Conv1D(filters=64,kernel_size=1,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(inputs1)
    shortcut=BatchNormalization()(shortcut)

    se_net1=Conv1D(filters=64,kernel_size=3,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv13)
    se_net1=BatchNormalization()(se_net1)
    se_net1=Activation('relu')(se_net1)
    
    
    block_1= Add()([shortcut, se_net1]) 
    block_1=Activation('relu')(block_1)   
#    block_1 = Dropout(0.2)(block_1)

#    22222222222222222222222222
    conv11 = Conv1D(filters=128,kernel_size=8,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(block_1)
    conv11=BatchNormalization()(conv11)
    conv11=Activation('relu')(conv11)

    
    conv12 = Conv1D(filters=128,kernel_size=5,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv11)   
    conv12=BatchNormalization()(conv12)
    conv12=Activation('relu')(conv12)
    
    conv13 = Conv1D(filters=128,kernel_size=3,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv12)
    conv13=BatchNormalization()(conv13)
    conv13=Activation('relu')(conv13)

    
    shortcut= Conv1D(filters=128,kernel_size=1,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(block_1)
    shortcut=BatchNormalization()(shortcut)
    
    
    se_net1=Conv1D(filters=128,kernel_size=3,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv13)
    se_net1=BatchNormalization()(se_net1)
    se_net1=Activation('relu')(se_net1)
    
    block_1= Add()([shortcut, se_net1]) 
    block_1=Activation('relu')(block_1)
    
#    block_1 = Dropout(0.2)(block_1)
     #333333333333333333333333
    conv11 = Conv1D(filters=128,kernel_size=8,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(block_1)
    conv11=BatchNormalization()(conv11)
    conv11=Activation('relu')(conv11)
#    pool11=MaxPool1D(pool_size=2)(conv11)
    conv12 = Conv1D(filters=128,kernel_size=5,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv11)
    conv12=BatchNormalization()(conv12)
    conv12=Activation('relu')(conv12)

    
    conv13 = Conv1D(filters=128,kernel_size=3,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv12)
    conv13=BatchNormalization()(conv13)
    conv13=Activation('relu')(conv13)
#    
#    pool13=MaxPool1D(pool_size=2)(conv13)

#    shortcut= Conv1D(filters=128,kernel_size=1,kernel_initializer="he_uniform",strides=2,padding='same',
#                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(block_1)
    shortcut=BatchNormalization()(block_1)
    
    se_net1=Conv1D(filters=128,kernel_size=3,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv13)
    se_net1=BatchNormalization()(se_net1)
    se_net1=Activation('relu')(se_net1)
    
    block_1= Add()([shortcut, se_net1])
 
    block_1=Activation('relu')(block_1)  


 ########################################################################################################################################### 
 #11111111111111111111
    conv11 = Conv1D(filters=64,kernel_size=8,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(inputs2)
    conv11=BatchNormalization()(conv11)
    conv11=Activation('relu')(conv11)

    conv12 = Conv1D(filters=64,kernel_size=5,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv11)
    conv12=BatchNormalization()(conv12)
    conv12=Activation('relu')(conv12)
    
    conv13 = Conv1D(filters=64,kernel_size=3,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv12)
    conv13=BatchNormalization()(conv13)
    conv13=Activation('relu')(conv13)
    
    shortcut= Conv1D(filters=64,kernel_size=1,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(inputs2)
    shortcut=BatchNormalization()(shortcut)

    se_net1=Conv1D(filters=64,kernel_size=3,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv13)
    se_net1=BatchNormalization()(se_net1)
    se_net1=Activation('relu')(se_net1)
    
    block_2= Add()([shortcut, se_net1]) 
    block_2=Activation('relu')(block_2)   

#    22222222222222222222222222
    conv11 = Conv1D(filters=128,kernel_size=8,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(block_2)
    conv11=BatchNormalization()(conv11)
    conv11=Activation('relu')(conv11)

    
    conv12 = Conv1D(filters=128,kernel_size=5,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv11)
    conv12=BatchNormalization()(conv12)
    conv12=Activation('relu')(conv12)
    
    conv13 = Conv1D(filters=128,kernel_size=3,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv12)
    conv13=BatchNormalization()(conv13)
    conv13=Activation('relu')(conv13)
   #    90*64
#    pool13=MaxPool1D(pool_size=3)(conv13)

    
    shortcut= Conv1D(filters=128,kernel_size=1,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(block_2)
    shortcut=BatchNormalization()(shortcut)
    
    se_net1=Conv1D(filters=128,kernel_size=3,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv13)
    se_net1=BatchNormalization()(se_net1)
    se_net1=Activation('relu')(se_net1)
    
    block_2= Add()([shortcut, se_net1]) 
    block_2=Activation('relu')(block_2)
    
    
#    block_1 = Dropout(0.2)(block_1)
     #333333333333333333333333
    conv11 = Conv1D(filters=128,kernel_size=8,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(block_2)
    conv11=BatchNormalization()(conv11)
#    45*16
    conv11=Activation('relu')(conv11)
#    pool11=MaxPool1D(pool_size=2)(conv11)
    conv12 = Conv1D(filters=128,kernel_size=5,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv11)
    conv12=BatchNormalization()(conv12)
    #    90*32
    conv12=Activation('relu')(conv12)
    
#    pool12=MaxPool1D(pool_size=2)(conv12)#################################################
    
    conv13 = Conv1D(filters=128,kernel_size=3,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv12)
    conv13=BatchNormalization()(conv13)
    conv13=Activation('relu')(conv13)

#    
#    pool13=MaxPool1D(pool_size=3)(conv13)

#    shortcut= Conv1D(filters=128,kernel_size=1,kernel_initializer="he_uniform",strides=2,padding='same',
#                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(block_1)
    
    shortcut=BatchNormalization()(block_2)
    
    se_net1=Conv1D(filters=128,kernel_size=3,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv13)
    se_net1=BatchNormalization()(se_net1)
    se_net1=Activation('relu')(se_net1)
    
    block_2= Add()([shortcut, se_net1])
    block_2=Activation('relu')(block_2)
    
    
#    block_1 = Dropout(0.2)(block_1)
    #444444444444444444444444444444
  

#####################################################################################################################
#    gap_layer=Flatten()(inputs4)
#    Dense2=Concatenate(axis=1)([Dense1,inputs4])
 
    Dense1=Concatenate(axis=1)([block_1,block_2])
    
#    LSTM layer  CuDNNLSTM
#    gap_layer=Bidirectional(tf.compat.v1.keras.layers.CuDNNLSTM(64,return_sequences=True))(Dense2)
#    gap_layer=tf.compat.v1.keras.layers.CuDNNLSTM(32,return_sequences=True)(Dense1)
#    gap_layer = Dropout(0.5)(gap_layer)
    
#    gap_layer=attention()(gap_layer)
    
    gap_layer1=Flatten()(Dense1)
    
    
#    # split for attention
#    attention_data = tf.keras.layers.Lambda(lambda x: x[:, :, :32])(Dense2)
#    attention_softmax = tf.keras.layers.Lambda(lambda x: x[:, :, 32:])(Dense2)
#    # attention mechanism
#    attention_softmax = tf.keras.layers.Softmax()(attention_softmax)
#    multiply_layer = tf.keras.layers.Multiply()([attention_softmax, attention_data])
    
#    gap_layer=Flatten()(Dense2)
    
    Dense2=Dense(800)(gap_layer1)
    
    Dense2=Activation('relu')(Dense2)
    
    Dense2 = Dropout(0.5)(Dense2)
    
    Dense3=Dense(50)(Dense2)
#    Dense2=Dense(32)(gap_layer)
    
    Dense3=Activation('relu')(Dense3)
    
    Dense3 = Dropout(0.5)(Dense3)
    
#    Dense2=attention()(Dense2)
#    Dense2=Dense(30)(Dense2)
#    
#    Dense2=Activation('relu')(Dense2)
#    
#    Dense2 = Dropout(0.5)(Dense2)
    
    Dense4 = Concatenate(axis=1)([Dense3, inputs4])
    
#    Dense2 = Dropout(0.1)(Dense1)
    
#    Dense1=Dense(32)(Dense1)
#    
#    Dense1=Activation('relu')(Dense1)
#    
#    Dense2=Dense(20)(Dense2)
#    fc2 = Dropout(0.1)(Dense2)
      
    
    
    output_layer = Dense(4, activation='softmax')(Dense4)
    
#    res = Model(inputs=[inputs1,inputs4], outputs=[output_layer], name="model360")
    #################################TSNE##################
#    res = Model(inputs=[inputs1,inputs2,inputs4], outputs=[output_layer, Dense3], name="model360")
    res = Model(inputs=[inputs1,inputs2,inputs4], outputs=[output_layer], name="model360")
    return res

def model360_revised_4(inputs1,inputs4):
    #11111111111111111111
    conv11 = Conv1D(filters=64,kernel_size=8,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(inputs1)
    conv11=BatchNormalization()(conv11)
    conv11=Activation('relu')(conv11)
#    conv111=conv11

    conv12 = Conv1D(filters=64,kernel_size=5,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv11)
    conv12=BatchNormalization()(conv12)
    conv12=Activation('relu')(conv12)
    
    conv13 = Conv1D(filters=64,kernel_size=3,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv12)
    conv13=BatchNormalization()(conv13)
    conv13=Activation('relu')(conv13)
    
    shortcut= Conv1D(filters=64,kernel_size=1,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(inputs1)
    shortcut=BatchNormalization()(shortcut)

    se_net1=SEBlock(conv13)
    block_1= Add()([shortcut, se_net1]) 
    block_1=Activation('relu')(block_1)   
#    block_1 = Dropout(0.2)(block_1)

#    22222222222222222222222222
    conv11 = Conv1D(filters=128,kernel_size=8,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(block_1)
    conv11=BatchNormalization()(conv11)
    conv11=Activation('relu')(conv11)

    
    conv12 = Conv1D(filters=128,kernel_size=5,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv11)   
    conv12=BatchNormalization()(conv12)
    conv12=Activation('relu')(conv12)
    
    conv13 = Conv1D(filters=128,kernel_size=3,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv12)
    conv13=BatchNormalization()(conv13)
    conv13=Activation('relu')(conv13)

    
    shortcut= Conv1D(filters=128,kernel_size=1,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(block_1)
    shortcut=BatchNormalization()(shortcut)
    
    se_net=SEBlock(conv13)
    block_1= Add()([shortcut, se_net]) 
    block_1=Activation('relu')(block_1)
    
#    block_1 = Dropout(0.2)(block_1)
     #333333333333333333333333
    conv11 = Conv1D(filters=128,kernel_size=8,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(block_1)
    conv11=BatchNormalization()(conv11)
    conv11=Activation('relu')(conv11)
#    pool11=MaxPool1D(pool_size=2)(conv11)
    conv12 = Conv1D(filters=128,kernel_size=5,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv11)
    conv12=BatchNormalization()(conv12)
    conv12=Activation('relu')(conv12)

    
    conv13 = Conv1D(filters=128,kernel_size=3,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(conv12)
    conv13=BatchNormalization()(conv13)
    conv13=Activation('relu')(conv13)
#    
#    pool13=MaxPool1D(pool_size=2)(conv13)

#    shortcut= Conv1D(filters=128,kernel_size=1,kernel_initializer="he_uniform",strides=2,padding='same',
#                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(block_1)
    shortcut=BatchNormalization()(block_1)
    
    se_net=SEBlock(conv13)
    block_1= Add()([shortcut, se_net])
 
    block_1=Activation('relu')(block_1)  


#####################################################################################################################

    
    gap_layer1=Flatten()(block_1)
    

    Dense2=Dense(800)(gap_layer1)
    
    Dense2=Activation('relu')(Dense2)
    
    Dense2 = Dropout(0.5)(Dense2)
    
    Dense3=Dense(50)(Dense2)
#    Dense2=Dense(32)(gap_layer)
    
    Dense3=Activation('relu')(Dense3)
    
    Dense3 = Dropout(0.5)(Dense3)
    
    Dense4 = Concatenate(axis=1)([Dense3, inputs4])
    
    
    
    output_layer = Dense(4, activation='softmax')(Dense4)
    
#    res = Model(inputs=[inputs1,inputs4], outputs=[output_layer], name="model360")
    #################################TSNE##################
#    res = Model(inputs=[inputs1,inputs2,inputs4], outputs=[output_layer, Dense3], name="model360")
    res = Model(inputs=[inputs1,inputs4], outputs=[output_layer], name="model360")
    return res



def cross_Kfolds(data1,data2,folds,jiange,start_index,end_index):
    #分割数据集;data1,data2分别为数据和标签
    df_Xtest=data1[start_index*jiange:end_index*jiange] #数据刚好可以做k折交叉验证。
    if start_index==0:
        df_Xtrain=data1[end_index*jiange:len(data1[:,0,0])*jiange]
    else:
        df_Xtrain00=data1[0*jiange:start_index*jiange]
        df_Xtrain01=data1[end_index*jiange:len(data1[:,0,0])*jiange]
        df_Xtrain=np.concatenate((df_Xtrain00,df_Xtrain01),axis=0)
    #分割标签
    df_ytest=data2[start_index*jiange:end_index*jiange] #数据刚好可以做k折交叉验证。
    if start_index==0:
        df_ytrain=data2[end_index*jiange:len(data2[:,0])*jiange]
    else:
        df_ytrain00=data2[0*jiange:start_index*jiange]
        df_ytrain01=data2[end_index*jiange:len(data2[:,0])*jiange]
        df_ytrain=np.concatenate((df_ytrain00,df_ytrain01),axis=0)
        
    return df_Xtrain,df_Xtest,df_ytrain,df_ytest  #每一折的训练和测试
def cross_Kfolds_f(data1,data1_f,data2,folds,jiange,start_index,end_index):
    #分割数据集;data1,data2分别为数据和标签
    df_Xtest=data1[start_index*jiange:end_index*jiange] #数据刚好可以做k折交叉验证。
    df_Xtest_f=data1_f[start_index*jiange:end_index*jiange] #数据刚好可以做k折交叉验证。
    if start_index==0:
        df_Xtrain=data1[end_index*jiange:len(data1[:,0,0])*jiange]
        df_Xtrain_f=data1_f[end_index*jiange:len(data1[:,0,0])*jiange]
    else:
        df_Xtrain00=data1[0*jiange:start_index*jiange]
        df_Xtrain01=data1[end_index*jiange:len(data1[:,0,0])*jiange]
        df_Xtrain=np.concatenate((df_Xtrain00,df_Xtrain01),axis=0)
        
        df_Xtrain00=data1_f[0*jiange:start_index*jiange]
        df_Xtrain01=data1_f[end_index*jiange:len(data1[:,0,0])*jiange]
        df_Xtrain_f=np.concatenate((df_Xtrain00,df_Xtrain01),axis=0)
    #分割标签
    df_ytest=data2[start_index*jiange:end_index*jiange] #数据刚好可以做k折交叉验证。
    if start_index==0:
        df_ytrain=data2[end_index*jiange:len(data2[:,0])*jiange]
    else:
        df_ytrain00=data2[0*jiange:start_index*jiange]
        df_ytrain01=data2[end_index*jiange:len(data2[:,0])*jiange]
        df_ytrain=np.concatenate((df_ytrain00,df_ytrain01),axis=0)
        
        df_ytrain00=data2[0*jiange:start_index*jiange]
        df_ytrain01=data2[end_index*jiange:len(data2[:,0])*jiange]
        df_ytrain=np.concatenate((df_ytrain00,df_ytrain01),axis=0)
        
    return df_Xtrain,df_Xtrain_f,df_Xtest,df_Xtest_f,df_ytrain,df_ytest  #每一折的训练和测试
#-----------------------------打乱数据集----------------------------------
def shuffle_set1(data,data1,label):
    train_row = list(range(len(label)))
    random.shuffle(train_row)
    Data = data[train_row]
    Data_f = data1[train_row]
    Label = label[train_row]
    return Data,Data_f,Label

def shuffle_set2(data,data1,data2,label):
    train_row = list(range(len(label)))
    random.shuffle(train_row)
    Data = data[train_row]
    Data1 = data1[train_row]
    Data_f = data2[train_row]
    Label = label[train_row]
    return Data,Data1,Data_f,Label

def shuffle_set3(data,data1,label):
    train_row = list(range(len(label)))
    random.shuffle(train_row)
    Data = data[train_row]
    Data1 = data1[train_row]
    Label = label[train_row]
    return Data,Data1,Label
def shuffle_set(data,label):
    train_row = list(range(len(label)))
    random.shuffle(train_row)
    Data = data[train_row]
    Label = label[train_row]
    return Data,Label

# Add attention layer to the deep learning network
class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1), 
                               initializer='zeros', trainable=True)        
        super(attention, self).build(input_shape)

    def call(self,x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)   
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context   
