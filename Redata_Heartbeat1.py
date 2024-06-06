# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 18:42:38 2020

@author: Aiyun
"""
import numpy as np
import wfdb
import matplotlib.pyplot as plt
from scipy.signal import resample
import glob
import pywt
from scipy import stats
#标准化：零均值，除以标准差
def get_mean_std(in_x):	
    return (in_x - np.mean(in_x,axis=0))/np.std(in_x,axis=0)
#--------------------小波去噪-----------------
def WTfilt_1d(sig):
    """
    对信号进行小波变换滤波
    :param sig: 输入信号，1-d array
    :return: 小波滤波后的信号，1-d array
    """
    coeffs = pywt.wavedec(sig, 'db6', level=9)
    coeffs[-1] = np.zeros(len(coeffs[-1]))
    coeffs[-2] = np.zeros(len(coeffs[-2]))
    coeffs[0] = np.zeros(len(coeffs[0]))
    sig_filt = pywt.waverec(coeffs, 'db6')
    return sig_filt    


#-------------------------心拍截取-------------------
def heartbeat(file0):
    '''
    file0:下载的MITAB数据
    
    '''
    N_Seg=[]; SVEB_Seg=[];  VEB_Seg=[]; F_Seg=[] ; RRtime_seg=[];
    N_RR_Seg=[] ;SVEB_RR_Seg=[] ;VEB_RR_Seg=[] ;F_RR_Seg=[] ;
#    Q_Seg=[];
    #--------去掉指定的四个导联的头文件---------
    De_file=[panth[:-1]+'\\102.hea',panth[:-1]+'\\104.hea',panth[:-1]+'\\107.hea',panth[:-1]+'\\217.hea']
    file=list(set(file0).difference(set(De_file)))
    
    for f in range(len(file)) :
        annotation= wfdb.rdann(panth+file[f][-7:-4],'atr')
        record_name=annotation.record_name    #读取记录名称
#        if record_name == 114:
#            Record=wfdb.rdsamp(panth+record_name)[0][:,1] #一般只取一个导联
#        else:
#            Record=wfdb.rdsamp(panth+record_name)[0][:,0] #一般只取一个导联
        
        if record_name == 114:
            Record=wfdb.rdsamp(panth+record_name)[0][:,0] #一般只取一个导联
        else:
            Record=wfdb.rdsamp(panth+record_name)[0][:,1] #一般只取一个导联
#        record=WTfilt_1d(Record)         #小波去噪
        record=Record
        label=annotation.symbol  #心拍标签列表
        label_index=annotation.sample   #标签索引列表
        RRtime_seg=[];
        for i in range(len(label)):#去除其他标签再求均值
            if label[i] == '+' or label[i] == 'x'or label[i] == '|'or label[i] == '~':
                label_index[i]=0
                label[i]='0'
        while '0' in label:
            label.remove('0')
        label_index1 = label_index!=0
        label_index = label_index[label_index1]
        
        for i in range(len(label_index)-1):
            RRtime=(label_index[i+1]-label_index[i])/360
            RRtime_seg.append(RRtime)
        RRtime_mean=np.mean(RRtime_seg)
        
        for j in range(len(label_index)-2):
            if j>10 and label_index[j]>=144  and (label_index[j]+180)<=650000:
#                if label[j]=='N' or label[j]=='.' or label[j]=='L' or label[j]=='R' or label[j]=='e' or label[j]=='j':
                if label[j]=='N' or label[j]=='L' or label[j]=='R' or label[j]=='e' or label[j]=='j':
                    if j<11:
                        f_near_pre_RR=np.mean(RRtime_seg[0:10])*360
                    else:
                        f_near_pre_RR=np.mean(RRtime_seg[j-11:j-1])*360
#                    
#                    Seg=record[label_index[j]-int(1/3*f_near_pre_RR):label_index[j]+int(2/3*f_near_pre_RR)]#R峰的前0.4s和后0.5s
                    Seg=record[label_index[j]-90:label_index[j]+110]
                    Seg=get_mean_std(Seg)
#                    segment=resample(Seg,300, axis=0)  #重采样到251
                    N_Seg.append(Seg)
#                    N_Seg.append(Seg)
                    
                    f1=RRtime_seg[j-1]
                    f2=RRtime_seg[j]
                    if j<5:
                        f3=np.mean(RRtime_seg[0:10])
                    elif j>len(label_index)-5:
                        f3=np.mean(RRtime_seg[len(label_index)-10:len(label_index)])
                    else:
                        f3=np.mean(RRtime_seg[j-5:j+5])
                    f4=RRtime_mean
                    
#                    f5=f1/f4
#                    f6=f2/f4
#                    f7=f3/f4
#                    
#                    #pre_RR_ratio
#                    f_pre_RR=np.mean(RRtime_seg[0:j])
#                    f8=f1/f_pre_RR
#                    
#                    #near_pre_RR_ratio
#                    if j<11:
#                        f9=np.mean(RRtime_seg[0:10])
#                    else:
#                        f_near_pre_RR=np.mean(RRtime_seg[j-11:j-1])
#                        f9=f1/f_near_pre_RR
                        
                    f5=f1-f4
                    f6=f2-f4
                    f7=f1/f2
                    
                    if j<11:
                        f9=np.mean(RRtime_seg[0:10])
                    else:
                        f_near_pre_RR=np.mean(RRtime_seg[j-11:j-1])
                        f9=f_near_pre_RR-f4
                        
                    f10=np.max(Seg)  
                    f11=np.min(Seg)
#                    f12=f10/f11
                    f13=np.var(Seg)
                    
                    f14=stats.skew(Seg)#偏度
                    f15=stats.kurtosis(Seg,fisher=False)#峰度
#                    RRSeg=[f5,f6,f7,f8,f9,f10,f11,f13,f14,f15]
#                    RRSeg=[f5,f6,f7,f8,f9,f10,f11,f14,f15]
                    RRSeg=[f5,f6,f7,f9,f10,f11,f14,f15]

#                    RRSeg=[f5,f6,f7,f10,f11,f13,f14]
                    
#                    RRSeg=[f1,f2,f3,f4,f5,f6,f7,f8,f9]
#                    RRSeg=[f1,f2,f3,f4,f5,f6,f7]
                    N_RR_Seg.append(np.array(RRSeg))
                        
                    
                if label[j]=='A' or label[j]=='a' or label[j]=='J' or label[j]=='S':
                    if j<11:
                        f_near_pre_RR=np.mean(RRtime_seg[0:10])*360
                    else:
                        f_near_pre_RR=np.mean(RRtime_seg[j-11:j-1])*360
                    
#                    Seg=record[label_index[j]-int(1/3*f_near_pre_RR):label_index[j]+int(2/3*f_near_pre_RR)]
                    Seg=record[label_index[j]-90:label_index[j]+110]
                    
#                    Seg=record[label_index[j-1]-72:label_index[j]+144]
                    Seg=get_mean_std(Seg)
#                    segment=resample(Seg,300, axis=0) 
                    SVEB_Seg.append(Seg)
#                    SVEB_Seg.append(Seg)
                    
                    f1=RRtime_seg[j-1]
                    f2=RRtime_seg[j]
                    if j<5:
                        f3=np.mean(RRtime_seg[0:10])
                    elif j>len(label_index)-5:
                        f3=np.mean(RRtime_seg[len(label_index)-10:len(label_index)])
                    else:
                        f3=np.mean(RRtime_seg[j-5:j+5])
                    f4=RRtime_mean
#                    f5=f1/f4
#                    f6=f2/f4
#                    f7=f3/f4
#                    
#                    f_pre_RR=np.mean(RRtime_seg[0:j])
#                    f8=f1/f_pre_RR
#                    #near_pre_RR_ratio
#                    if j<11:
#                        f9=np.mean(RRtime_seg[0:10])
#                    else:
#                        f_near_pre_RR=np.mean(RRtime_seg[j-11:j-1])
#                        f9=f1/f_near_pre_RR
                    
                    f5=f1-f4
                    f6=f2-f4
                    f7=f1/f2
                    
                    if j<11:
                        f9=np.mean(RRtime_seg[0:10])
                    else:
                        f_near_pre_RR=np.mean(RRtime_seg[j-11:j-1])
                        f9=f_near_pre_RR-f4
                        
                    f10=np.max(Seg)  
                    f11=np.min(Seg)
#                    f12=f10/f11
                    f13=np.var(Seg)
                    
                    f14=stats.skew(Seg)#偏度
                    f15=stats.kurtosis(Seg,fisher=False)#峰度
#                    RRSeg=[f5,f6,f7,f8,f9,f10,f11,f14,f15]
                    RRSeg=[f5,f6,f7,f9,f10,f11,f14,f15]
#                    RRSeg=[f5,f6,f7,f10,f11,f13,f14]
#                    RRSeg=[f1,f2,f3,f4,f5,f6,f7,f8,f9]
#                    RRSeg=[f1,f2,f3,f4,f5,f6,f7]
                    SVEB_RR_Seg.append(np.array(RRSeg))
                    
                if label[j]=='V' or label[j]=='E':
                    if j<11:
                        f_near_pre_RR=np.mean(RRtime_seg[0:10])*360
                    else:
                        f_near_pre_RR=np.mean(RRtime_seg[j-11:j-1])*360
                    
#                    Seg=record[label_index[j]-int(1/3*f_near_pre_RR):label_index[j]+int(2/3*f_near_pre_RR)]
                    Seg=record[label_index[j]-90:label_index[j]+110]
                    Seg=get_mean_std(Seg)
#                    segment=resample(Seg,300, axis=0)  
                    VEB_Seg.append(Seg)
#                    VEB_Seg.append(Seg)
                    
                    f1=RRtime_seg[j-1]
                    f2=RRtime_seg[j]
                    if j<5:
                        f3=np.mean(RRtime_seg[0:10])
                    elif j>len(label_index)-5:
                        f3=np.mean(RRtime_seg[len(label_index)-10:len(label_index)])
                    else:
                        f3=np.mean(RRtime_seg[j-5:j+5])
                    f4=RRtime_mean
#                    f5=f1/f4
#                    f6=f2/f4
#                    f7=f3/f4
#                    f_pre_RR=np.mean(RRtime_seg[0:j])
#                    f8=f1/f_pre_RR
#                    #near_pre_RR_ratio
#                    if j<11:
#                        f9=np.mean(RRtime_seg[0:10])
#                    else:
#                        f_near_pre_RR=np.mean(RRtime_seg[j-11:j-1])
#                        f9=f1/f_near_pre_RR
                    
                    f5=f1-f4
                    f6=f2-f4
                    f7=f1/f2
                    
                    if j<11:
                        f9=np.mean(RRtime_seg[0:10])
                    else:
                        f_near_pre_RR=np.mean(RRtime_seg[j-11:j-1])
                        f9=f_near_pre_RR-f4
                        
                    f10=np.max(Seg)  
                    f11=np.min(Seg)
#                    f12=f10/f11
                    f13=np.var(Seg)
                    
                    f14=stats.skew(Seg)#偏度
                    f15=stats.kurtosis(Seg,fisher=False)#峰度
#                    RRSeg=[f5,f6,f7,f8,f9,f10,f11,f14,f15]
                    RRSeg=[f5,f6,f7,f9,f10,f11,f14,f15]
#                    RRSeg=[f5,f6,f7,f10,f11,f13,f14]
#                    RRSeg=[f1,f2,f3,f4,f5,f6,f7,f8,f9]
#                    RRSeg=[f1,f2,f3,f4,f5,f6,f7]
                    VEB_RR_Seg.append(np.array(RRSeg))
                    
                if label[j]=='F':
                    if j<11:
                        f_near_pre_RR=np.mean(RRtime_seg[0:10])*360
                    else:
                        f_near_pre_RR=np.mean(RRtime_seg[j-11:j-1])*360
                    
#                    Seg=record[label_index[j]-int(1/3*f_near_pre_RR):label_index[j]+int(2/3*f_near_pre_RR)]
                    Seg=record[label_index[j]-90:label_index[j]+110]
                    Seg=get_mean_std(Seg)
#                    segment=resample(Seg,300, axis=0)  
                    F_Seg.append(Seg)
#                    F_Seg.append(Seg)
                    
                    f1=RRtime_seg[j-1]
                    f2=RRtime_seg[j]
                    if j<5:
                        f3=np.mean(RRtime_seg[0:10])
                    elif j>len(label_index)-5:
                        f3=np.mean(RRtime_seg[len(label_index)-10:len(label_index)])
                    else:
                        f3=np.mean(RRtime_seg[j-5:j+5])
                    f4=RRtime_mean
#                    f5=f1/f4
#                    f6=f2/f4
#                    f7=f3/f4
#                    f_pre_RR=np.mean(RRtime_seg[0:j])
#                    f8=f1/f_pre_RR
#                    #near_pre_RR_ratio
#                    if j<11:
#                        f9=np.mean(RRtime_seg[0:10])
#                    else:
#                        f_near_pre_RR=np.mean(RRtime_seg[j-11:j-1])
#                        f9=f1/f_near_pre_RR
                    
                    f5=f1-f4
                    f6=f2-f4
                    f7=f1/f2
                    
                    if j<11:
                        f9=np.mean(RRtime_seg[0:10])
                    else:
                        f_near_pre_RR=np.mean(RRtime_seg[j-11:j-1])
                        f9=f_near_pre_RR-f4
                    
                    f10=np.max(Seg)  
                    f11=np.min(Seg)
#                    f12=f10/f11
                    f13=np.var(Seg)
                    
                    f14=stats.skew(Seg)#偏度
                    f15=stats.kurtosis(Seg,fisher=False)#峰度
                    
#                    RRSeg=[f5,f6,f7,f8,f9,f10,f11,f14,f15]
                    RRSeg=[f5,f6,f7,f9,f10,f11,f14,f15]
#                    RRSeg=[f1,f2,f3,f4,f5,f6,f7,f8,f9]
#                    RRSeg=[f1,f2,f3,f4,f5,f6,f7]
                    F_RR_Seg.append(np.array(RRSeg))
#                if  label[j]=='/' or label[j]=='f' or label[j]=='Q':
#                    
#                    Seg=record[label_index[j]-144:label_index[j]+180]
#                    segment=resample(Seg,251, axis=0)  
#                    Q_Seg.append(segment)
                    
    N_segement=np.array(N_Seg)
    SVEB_segement=np.array(SVEB_Seg)
    VEB_segement=np.array(VEB_Seg)
    F_segement=np.array(F_Seg)
#    Q_segement=np.array(Q_Seg)
    
    N_RR_segement=np.array(N_RR_Seg)
    SVEB_RR_segement=np.array(SVEB_RR_Seg)
    VEB_RR_segement=np.array(VEB_RR_Seg)
    F_RR_segement=np.array(F_RR_Seg)
    
    label_N=np.zeros(N_segement.shape[0])
    label_SVEB=np.ones(SVEB_segement.shape[0])
    label_VEB=np.ones(VEB_segement.shape[0])*2
    label_F=np.ones(F_segement.shape[0])*3
#    label_Q=np.ones(Q_segement.shape[0])*4
                    
    Data=np.concatenate((N_segement,SVEB_segement,VEB_segement,F_segement),axis=0)
    Data_f=np.concatenate((N_RR_segement,SVEB_RR_segement,VEB_RR_segement,F_RR_segement),axis=0)

    Label=np.concatenate((label_N,label_SVEB,label_VEB,label_F,),axis=0) #四分类
    
    return  Data, Data_f, Label, N_segement, SVEB_segement, VEB_segement, F_segement

#-----------------------心拍截取和保存---------------------
#建议一次性截取和保存，不需要重复操作，下次训练和测试的时候，直接load
panth='/home/lingang/liujinlei/mit_classify/DS2/'
file = glob.glob(panth+'*.hea')
Data, Data_f, Label, N, S, V, F=heartbeat(file)

#Data1=np.save('/home/lingang/liujinlei/mit_classify/data11/'+'N_rawsegement',N)
#Data2=np.save('/home/lingang/liujinlei/mit_classify/data11/'+'S_rawsegement',S)
#Data3=np.save('/home/lingang/liujinlei/mit_classify/data11/'+'V_rawsegement',V)
#Data4=np.save('/home/lingang/liujinlei/mit_classify/data11/'+'F_rawsegement',F)

Data=np.save('/home/lingang/liujinlei/mit_classify/data13/'+'Data1_DS2',Data)
#Data_f=np.save('/home/lingang/liujinlei/mit_classify/data13/'+'Data_f_DS1',Data_f)
Label=np.save('/home/lingang/liujinlei/mit_classify/data13/'+'Label1_DS2',Label)

#data7双通道+9特征，无滤波    1/3D+2/3D
#data11是双通道+9特征，滤波   1/3D+2/3D

#data12是双通道+8特征，滤波   90,110,CWT文章特征

#data12是双通道+8特征，滤波 归一化  90,110,CWT文章特征

