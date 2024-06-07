from concurrent.futures import ProcessPoolExecutor

import cv2
import joblib
import numpy as np
import pywt
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from functools import partial
from sklearn.metrics import classification_report, confusion_matrix, f1_score, make_scorer
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring, Initializer, LRScheduler, TensorBoard,Checkpoint, TrainEndCheckpoint
from skorch.dataset import Dataset
from skorch.helper import predefined_split
from torch.backends import cudnn
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from tensorflow.keras.utils import plot_model

from scipy import stats

cudnn.benchmark = False
cudnn.deterministic = True

torch.manual_seed(0)

def worker1(data, wavelet, scales, sampling_period):
    # heartbeat segmentation interval
    before, after = 90, 160

    
#    coeffs, frequencies = pywt.cwt(data["signal_2"], scales, wavelet, sampling_period)
    r_peaks, categories, ecgsignal = data["r_peaks"], data["categories"], data["signal_2"]

    # for remove inter-patient variation
    avg_rri = np.mean(np.diff(r_peaks))

    x1, x2, y, groups = [], [], [], []
    for i in range(len(r_peaks)):
#        if i == 0 or i == len(r_peaks) - 1:
#            continue
        if i < 5 or i > len(r_peaks) - 5:
            continue
        if categories[i] == 4:  # remove AAMI Q class
            continue

        # cv2.resize is used to sampling the scalogram to (100 x100)
#        x1.append(coeffs[:, r_peaks[i] - before: r_peaks[i] + after])
#        if r_peaks[i] > 1800 and r_peaks[i] < data["signal"].shape[1]-1800:
            
#            coeffs, frequencies = pywt.cwt(data["signal"][r_peaks[i] - 1800: r_peaks[i] + 1800], scales, wavelet, sampling_period)
#        x0.append(ecgsignal[r_peaks[i] - before: r_peaks[i] + after])
        
        Seg=ecgsignal[r_peaks[i] - before: r_peaks[i] + after]
        
#        x0.append(cv2.resize(coeffs1[:, r_peaks[i] - before: r_peaks[i] + after], (100, 100)))
#        x1.append(cv2.resize(coeffs[:, r_peaks[i] - before: r_peaks[i] + after], (100, 100)))
        x1.append(Seg)
        x2.append([
            r_peaks[i] - r_peaks[i - 1] - avg_rri,  # previous RR Interval
            r_peaks[i + 1] - r_peaks[i] - avg_rri,  # post RR Interval
            (r_peaks[i] - r_peaks[i - 1]) / (r_peaks[i + 1] - r_peaks[i]),  # ratio RR Interval
            np.mean(np.diff(r_peaks[np.maximum(i - 10, 0):i + 1])) - avg_rri,  # local RR Interval
            
#            np.max(Seg),
#            np.min(Seg),
#            stats.skew(Seg),#偏度
#            stats.kurtosis(Seg,fisher=False)#峰度
            
            
            
        ])
        y.append(categories[i])
        groups.append(data["record"])

    return x1, x2, y, groups

def worker(data, wavelet, scales, sampling_period):
    # heartbeat segmentation interval
    before, after = 90, 160

#    if(data == 'train_data' or data == 'test_data'):
#        coeffs, frequencies = pywt.cwt(data["signal"], scales, wavelet, sampling_period)
#        r_peaks, categories, ecgsignal = data["r_peaks"], data["categories"], data["signal"]
#        
#    if(data == 'train_data2' or data == 'test_data2'):
#        coeffs1, frequencies = pywt.cwt(data["signal_2"], scales, wavelet, sampling_period)
#        r_peaks, categories, ecgsignal = data["r_peaks"], data["categories"], data["signal_2"]
#    
#    coeffs, frequencies = pywt.cwt(data["signal"], scales, wavelet, sampling_period)
    r_peaks, categories, ecgsignal = data["r_peaks"], data["categories"], data["signal"]
    # for remove inter-patient variation
    avg_rri = np.mean(np.diff(r_peaks))

    x1, x2, y, groups = [], [], [], []
    for i in range(len(r_peaks)):
#        if i == 0 or i == len(r_peaks) - 1:
#            continue
        if i < 5 or i > len(r_peaks) - 5:
            continue

        if categories[i] == 4:  # remove AAMI Q class
            continue

        # cv2.resize is used to sampling the scalogram to (100 x100)
#        x1.append(coeffs[:, r_peaks[i] - before: r_peaks[i] + after])
#        if r_peaks[i] > 1800 and r_peaks[i] < data["signal"].shape[1]-1800:
            
#            coeffs, frequencies = pywt.cwt(data["signal"][r_peaks[i] - 1800: r_peaks[i] + 1800], scales, wavelet, sampling_period)
#        x0.append(ecgsignal[r_peaks[i] - before: r_peaks[i] + after])
        
        Seg=ecgsignal[r_peaks[i] - before: r_peaks[i] + after]
        
#        x0.append(cv2.resize(coeffs1[:, r_peaks[i] - before: r_peaks[i] + after], (100, 100)))
#        x1.append(cv2.resize(coeffs[:, r_peaks[i] - before: r_peaks[i] + after], (100, 100)))
        x1.append(Seg)
#        x1.append(cv2.resize(coeffs[:, r_peaks[i] - before: r_peaks[i] + after], (100, 100)))
        x2.append([
            r_peaks[i] - r_peaks[i - 1] - avg_rri,  # previous RR Interval
            r_peaks[i + 1] - r_peaks[i] - avg_rri,  # post RR Interval
            (r_peaks[i] - r_peaks[i - 1]) / (r_peaks[i + 1] - r_peaks[i]),  # ratio RR Interval
            np.mean(np.diff(r_peaks[np.maximum(i - 10, 0):i + 1])) - avg_rri,  # local RR Interval
            np.max(Seg),
            np.min(Seg),
            stats.skew(Seg),#偏度
            stats.kurtosis(Seg,fisher=False)#峰度
            
            
            
        ])
        y.append(categories[i])
        groups.append(data["record"])

    return x1, x2, y, groups


def load_data(wavelet, scales, sampling_rate, filename="./data2/mitdb.pkl",filename2="./data2/mitdb_2.pkl"):
    import pickle
    from sklearn.preprocessing import RobustScaler

    with open(filename, "rb") as f:
        train_data, test_data = pickle.load(f)
        
    with open(filename2, "rb") as f:
        train_data2, test_data2 = pickle.load(f)

    cpus = 22 if joblib.cpu_count() > 22 else joblib.cpu_count() - 1  # for multi-process

    # for training
    x1_train, x2_train, y_train, groups_train = [], [], [], []
    with ProcessPoolExecutor(max_workers=cpus) as executor:
        for x1, x2, y, groups in executor.map(
                partial(worker, wavelet=wavelet, scales=scales, sampling_period=1. / sampling_rate), train_data):
            
            x1_train.append(x1)
            x2_train.append(x2)
            y_train.append(y)
            groups_train.append(groups)

    x1_train = np.expand_dims(np.concatenate(x1_train, axis=0), axis=1).astype(np.float32)
    x2_train = np.concatenate(x2_train, axis=0).astype(np.float32)
    y_train = np.concatenate(y_train, axis=0).astype(np.int64)
    groups_train = np.concatenate(groups_train, axis=0)

    # for test
    x1_test, x2_test, y_test, groups_test = [], [], [], []
    with ProcessPoolExecutor(max_workers=cpus) as executor:
        for x1, x2, y, groups in executor.map(
                partial(worker, wavelet=wavelet, scales=scales, sampling_period=1. / sampling_rate), test_data):
            x1_test.append(x1)
            x2_test.append(x2)
            y_test.append(y)
            groups_test.append(groups)

    x1_test = np.expand_dims(np.concatenate(x1_test, axis=0), axis=1).astype(np.float32)
    x2_test = np.concatenate(x2_test, axis=0).astype(np.float32)
    y_test = np.concatenate(y_test, axis=0).astype(np.int64)
    groups_test = np.concatenate(groups_test, axis=0)
    
    
    # for training2
    x0_train, x2_train1, y_train1, groups_train1 = [], [], [], []
    with ProcessPoolExecutor(max_workers=cpus) as executor:
        for x1, x2, y, groups in executor.map(
                partial(worker1, wavelet=wavelet, scales=scales, sampling_period=1. / sampling_rate), train_data2):
            
            x0_train.append(x1)
#            x2_train.append(x2)
#            y_train.append(y)
#            groups_train.append(groups)

    x0_train = np.expand_dims(np.concatenate(x0_train, axis=0), axis=1).astype(np.float32)
#    x2_train = np.concatenate(x2_train, axis=0).astype(np.float32)
#    y_train = np.concatenate(y_train, axis=0).astype(np.int64)
#    groups_train = np.concatenate(groups_train, axis=0)

    # for test
    x0_test, x2_test2, y_test2, groups_test2 = [], [], [], []
    with ProcessPoolExecutor(max_workers=cpus) as executor:
        for x1, x2, y, groups in executor.map(
                partial(worker1, wavelet=wavelet, scales=scales, sampling_period=1. / sampling_rate), test_data2):
            x0_test.append(x1)
#            x2_test.append(x2)
#            y_test.append(y)
#            groups_test.append(groups)

    x0_test = np.expand_dims(np.concatenate(x0_test, axis=0), axis=1).astype(np.float32)
#    x2_test = np.concatenate(x2_test, axis=0).astype(np.float32)
#    y_test = np.concatenate(y_test, axis=0).astype(np.int64)
#    groups_test = np.concatenate(groups_test, axis=0)
    
    

    # normalization
    scaler = RobustScaler()
    x2_train = scaler.fit_transform(x2_train)
    x2_test = scaler.transform(x2_test)

    return (x0_train,x1_train, x2_train, y_train, groups_train), (x0_test,x1_test, x2_test, y_test, groups_test)


def main():
    sampling_rate = 360

    wavelet = "mexh"  # mexh, morl, gaus8, gaus4

   (x0_train,x1_train, x2_train, y_train, groups_train), (x0_test,x1_test, x2_test, y_test, groups_test) = load_data(
       wavelet=wavelet, scales=scales, sampling_rate=sampling_rate)
   
   np.save('./data3/x0_train.npy', x0_train)
   np.save('./data3/x1_train.npy', x1_train)
   np.save('./data3/x2_train.npy', x2_train)
   np.save('./data3/y_train.npy', y_train)
   
   np.save('./data3/x0_test.npy', x0_test)
   np.save('./data3/x1_test.npy', x1_test)
   np.save('./data3/x2_test.npy', x2_test)
   np.save('./data3/y_test.npy', y_test)    

   print("Data Saved successfully!")


if __name__ == "__main__":
    main()
   




