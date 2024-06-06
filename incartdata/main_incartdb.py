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
from scipy.signal import resample

from scipy import stats

cudnn.benchmark = False
cudnn.deterministic = True

torch.manual_seed(0)




class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
 
class ResidualBlock(nn.Module):
    """
    实现子module: Residual Block
    """
 
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.se = SELayer(outchannel, 16)
        self.right = shortcut
        
 
    def forward(self, x):
        out = self.left(x)
        out= self.se(out)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)
 
 
class ResNet(nn.Module):
    """
    实现主module：ResNet34
    ResNet34包含多个layer，每个layer又包含多个Residual block
    用子module来实现Residual block，用_make_layer函数来实现layer
    """
 
    def __init__(self, blocks, num_classes=4):
        super(ResNet, self).__init__()
        self.model_name = 'resnet34'
 
        # 前几层: 图像转换
        self.pre = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1))
 
        # 重复的layer，分别有3，4，6，3个residual block
#        self.layer1 = self._make_layer(64, 64, blocks[0])
        self.layer1 = self._make_layer(1, 32, blocks[0])
#        self.layer2 = self._make_layer(64, 128, blocks[1], stride=2)
#        self.layer3 = self._make_layer(128, 256, blocks[2], stride=2)
#        self.layer4 = self._make_layer(256, 512, blocks[3], stride=2)
 
        # 分类用的全连接
#        self.fc1 = nn.Linear(260, 64)
#        self.fc = nn.Linear(64, num_classes)
        
        self.drop = nn.Dropout(p=0.3)
        
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
#        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
#        self.pooling1 = nn.MaxPool2d(5)
        self.pooling2 = nn.MaxPool2d(5)
        self.pooling3 = nn.AdaptiveMaxPool2d((1, 1))
        self.fc1 = nn.Linear(132, 32)
        self.fc = nn.Linear(32, 4)
 
    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        """
        构建layer,包含多个residual block
        """
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU()
        )
 
        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))
 
        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)
 
    def forward(self, x,x2):
#        x = self.pre(x)
        
        x = self.layer1(x)
#        model = Residual(1,1)
#        x = model(x)
#        x = self.layer2(x)
#        x = self.layer3(x)
#        x = self.layer4(x)
#        x = F.avg_pool2d(x, 7)
        print(x.shape)
#        x = x.view(x.size(0), -1)
        
        x = F.relu(self.bn2(self.conv2(x)))  # (32 x 16 x 16)
        
        x = self.pooling2(x)  # (32 x 5 x 5)
        
        x = F.relu(self.bn3(self.conv3(x)))  # (64 x 3 x 3)
        
        x = self.pooling3(x)  # (64 x 1 x 1)
        
        
        x = x.view((-1, 128)) 
        
        
        
#        x0 = self.pre(x0)
#    
#        x0 = self.layer1(x0)
#        x0 = self.layer2(x0)
#        x0 = self.layer3(x0)
##        x = self.layer4(x)
#        x0 = F.avg_pool2d(x0, 7)
##        print(x.shape)
#        x0 = x0.view(x0.size(0), -1)
        
        
#        x = torch.cat((x,x0,x2), dim=1)
        x = torch.cat((x,x2), dim=1)
#        print(x.shape)
        x = self.fc1(x)
        
        return self.fc(x)
    
class Residual(nn.Module):  #@save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        
        self.conv4 = nn.Conv2d(1, 64, 3)
        self.conv5 = nn.Conv2d(64, 128, 3)
#        self.bn1 = nn.BatchNorm2d(16)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(128)
#        self.pooling1 = nn.MaxPool2d(5)
        self.pooling4 = nn.MaxPool2d(5)
        self.pooling5 = nn.AdaptiveMaxPool2d((1, 1))
        self.fc1 = nn.Linear(132, 32)
        self.fc = nn.Linear(32, 4)
 
    def forward(self, x,x2):
        
        x = F.relu(self.bn1(self.conv1(x)))
        Y = self.bn2(self.conv2(x))
        if self.conv3:
            x = self.conv3(x)
        Y += x
        
#        print(Y.shape)
        
        x = F.relu(self.bn4(self.conv4(Y)))  # (32 x 16 x 16)
#        print(Y.shape)
        x = self.pooling4(x)  # (32 x 5 x 5)
#        print(Y.shape)
        x = F.relu(self.bn5(self.conv5(x)))  # (64 x 3 x 3)
#        print(Y.shape)
        x = self.pooling5(x)  # (64 x 1 x 1)
#        print(Y.shape)
        
        x = x.view((-1, 128)) 
        
        
        
#        x0 = self.pre(x0)
#    
#        x0 = self.layer1(x0)
#        x0 = self.layer2(x0)
#        x0 = self.layer3(x0)
##        x = self.layer4(x)
#        x0 = F.avg_pool2d(x0, 7)
##        print(x.shape)
#        x0 = x0.view(x0.size(0), -1)
        
        
#        x = torch.cat((x,x0,x2), dim=1)
        x = torch.cat((x,x2), dim=1)
#        print(x.shape)
        x = self.fc1(x)
        
        return self.fc(x)
        
 
def Se_ResNet18():
    return ResNet([1])


class MyModule1(nn.Module):
    def __init__(self):
        super(MyModule1, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 7)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.pooling1 = nn.MaxPool2d(5)
        self.pooling2 = nn.MaxPool2d(3)
        self.pooling3 = nn.AdaptiveMaxPool2d((1, 1))
        self.fc1 = nn.Linear(136, 32)
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x, x0, x2):
        x = F.relu(self.bn1(self.conv1(x)))  # (16 x 94 x 94)
        x = self.pooling1(x)  # (16 x 18 x 18)
        x = F.relu(self.bn2(self.conv2(x)))  # (32 x 16 x 16)
        x = self.pooling2(x)  # (32 x 5 x 5)
        x = F.relu(self.bn3(self.conv3(x)))  # (64 x 3 x 3)
        x = self.pooling3(x)  # (64 x 1 x 1)
        x = x.view((-1, 64))  # (64,)
        
        
        x0 = F.relu(self.bn1(self.conv1(x0)))  # (16 x 94 x 94)
        x0 = self.pooling1(x0)  # (16 x 18 x 18)
        x0 = F.relu(self.bn2(self.conv2(x0)))  # (32 x 16 x 16)
        x0 = self.pooling2(x0)  # (32 x 5 x 5)
        x0 = F.relu(self.bn3(self.conv3(x0)))  # (64 x 3 x 3)
        x0 = self.pooling3(x0)  # (64 x 1 x 1)
        x0 = x0.view((-1, 64))  # (64,)
        
        
        x = torch.cat((x, x0, x2), dim=1)  # (68,)
        x = F.relu(self.fc1(x))  # (32,)
        x = self.fc2(x)  # (4,)
        return x

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 7)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.pooling1 = nn.MaxPool2d(5)
        self.pooling2 = nn.MaxPool2d(3)
        self.pooling3 = nn.AdaptiveMaxPool2d((1, 1))
        self.fc1 = nn.Linear(68, 32)
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x, x2):
        x = F.relu(self.bn1(self.conv1(x)))  # (16 x 94 x 94)
        x = self.pooling1(x)  # (16 x 18 x 18)
        x = F.relu(self.bn2(self.conv2(x)))  # (32 x 16 x 16)
        x = self.pooling2(x)  # (32 x 5 x 5)
        x = F.relu(self.bn3(self.conv3(x)))  # (64 x 3 x 3)
        x = self.pooling3(x)  # (64 x 1 x 1)
        x = x.view((-1, 64))  # (64,)
    
        x = torch.cat((x, x2), dim=1)  # (68,)
        x = F.relu(self.fc1(x))  # (32,)
        x = self.fc2(x)  # (4,)
        return x
    
class MyModule2(nn.Module):
    def __init__(self):
        super(MyModule2, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 7)
        self.conv2 = nn.Conv2d(8, 16, 2)
        self.conv3 = nn.Conv2d(16, 32, 2)
        self.conv4 = nn.Conv2d(32, 64, 2)
        self.conv5 = nn.Conv2d(64, 128, 2)
        
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(128)
        
        self.pooling1 = nn.MaxPool2d(2)
        self.pooling2 = nn.MaxPool2d(2)
        self.pooling3 = nn.MaxPool2d(2)
        self.pooling4 = nn.MaxPool2d(2)
        
        self.pooling5 = nn.AdaptiveMaxPool2d((1, 1))
        
        self.fc1 = nn.Linear(132, 32)
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x, x2):
        x = F.relu(self.bn1(self.conv1(x)))  # (16 x 94 x 94)
        
        x = self.pooling1(x)  # (16 x 18 x 18)
        
        x = F.relu(self.bn2(self.conv2(x)))  # (32 x 16 x 16)
        
        x = self.pooling2(x)  # (32 x 5 x 5)
        
        x = F.relu(self.bn3(self.conv3(x)))  # (64 x 3 x 3)
        
        x = self.pooling3(x)  # (64 x 1 x 1)
        
        x = F.relu(self.bn4(self.conv4(x)))  # (64 x 3 x 3)
        
        x = self.pooling4(x)  # (64 x 1 x 1)
        
        x = F.relu(self.bn5(self.conv5(x)))  # (64 x 3 x 3)
        
        x = self.pooling5(x)  # (64 x 1 x 1)
        
        x = x.view((-1, 128))  # (64,)
    
        x = torch.cat((x, x2), dim=1)  # (68,)
        x = F.relu(self.fc1(x))  # (32,)
        x = self.fc2(x)  # (4,)
        return x    
    
def worker1(data, wavelet, scales, sampling_period):
    # heartbeat segmentation interval
    before, after = 65, 115

    
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
#        Seg=resample(Seg,250,axis=0)
        
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
#    before, after = 64, 114
    before, after = 65, 115

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
#        Seg=resample(Seg,250,axis=0)
        
#        x0.append(cv2.resize(coeffs1[:, r_peaks[i] - before: r_peaks[i] + after], (100, 100)))
#        x1.append(cv2.resize(coeffs[:, r_peaks[i] - before: r_peaks[i] + after], (100, 100)))
        x1.append(Seg)
#        x1.append(cv2.resize(coeffs[:, r_peaks[i] - before: r_peaks[i] + after], (100, 100)))
        x2.append([
            r_peaks[i] - r_peaks[i - 1] - avg_rri,  # previous RR Interval
            r_peaks[i + 1] - r_peaks[i] - avg_rri,  # post RR Interval
            (r_peaks[i] - r_peaks[i - 1]) / (r_peaks[i + 1] - r_peaks[i]),  # ratio RR Interval
            np.mean(np.diff(r_peaks[np.maximum(i - 10, 0):i + 1])) - avg_rri,  # local RR Interval
            
#            (r_peaks[i] - r_peaks[i - 1]) / avg_rri,  # previous RR Interval
#            (r_peaks[i + 1] - r_peaks[i]) / avg_rri,  # post RR Interval
            np.mean(np.diff(r_peaks[i-5:i+5])) - avg_rri, 
#            
            np.max(Seg),
            np.min(Seg),
            stats.skew(Seg),#偏度
            stats.kurtosis(Seg,fisher=False)#峰度
            
            
            
        ])
        y.append(categories[i])
        groups.append(data["record"])

    return x1, x2, y, groups


def load_data(wavelet, scales, sampling_rate, filename="./data1/mitdb.pkl",filename2="./data1/mitdb_2.pkl"):
    import pickle
    from sklearn.preprocessing import RobustScaler

    with open(filename, "rb") as f:
        train_data,test_data = pickle.load(f)
        
    with open(filename2, "rb") as f:
        train_data2,test_data2 = pickle.load(f)

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
#
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
#    x1,x2,y,groups=worker1(train_data2,wavelet,scales,1./sampling_rate)    
#    x0_train.append(x1)   
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
#
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
    sampling_rate = 257

    wavelet = "mexh"  # mexh, morl, gaus8, gaus4
    scales = pywt.central_frequency(wavelet) * sampling_rate / np.arange(1, 101, 1)
    
#    x0_train=np.load('./data1/x0_train.npy') 
#    
#    x1_train=np.load('./data/x1_train.npy')
#    x2_train=np.load('./data/x2_train.npy')
#    y_train=np.load('./data/y_train.npy')
##
##    x0_test=np.load('./data1/x0_test.npy')
#    
#    x1_test=np.load('./data/x1_test.npy')
#    x2_test=np.load('./data/x2_test.npy')
#    y_test=np.load('./data/y_test.npy')
    
    
#    x1_train=np.load('/home/lingang/liujinlei/mit_classify03/data_cwt/Data_DS1.npy')
#    x1_train=np.expand_dims(x1_train,axis=1).astype(np.float32)
#    
#    x2_train=np.load('/home/lingang/liujinlei/mit_classify03/data_cwt/Data_f_DS1.npy')
#    x2_train = x2_train.astype(np.float32);
#    
#    y_train=np.load('/home/lingang/liujinlei/mit_classify03/data_cwt/Label_DS1.npy')
#    y_train=y_train.astype(np.int64)
#    
#    #x0_test=np.load('/home/lingang/liujinlei/mit_classify03/data_cwt/xtime_test.npy')
#    x1_test=np.load('/home/lingang/liujinlei/mit_classify03/data_cwt/Data_DS2.npy')
#    x1_test=np.expand_dims(x1_test,axis=1).astype(np.float32)
#    
#    x2_test=np.load('/home/lingang/liujinlei/mit_classify03/data_cwt/Data_f_DS2.npy')
#    x2_test = x2_test.astype(np.float32);
#    
#    y_test=np.load('/home/lingang/liujinlei/mit_classify03/data_cwt/Label_DS2.npy')
#    y_test=y_test.astype(np.int64)
#    
    
    (x0_train,x1_train, x2_train, y_train, groups_train), (x0_test,x1_test, x2_test, y_test, groups_test) = load_data(
        wavelet=wavelet, scales=scales, sampling_rate=sampling_rate)
#    
    np.save('./data2/x0_train.npy', x0_train)
    np.save('./data2/x1_train.npy', x1_train)
    np.save('./data2/x2_train.npy', x2_train)
    np.save('./data2/y_train.npy', y_train)
    
    np.save('./data2/x0_test.npy', x0_test)
    np.save('./data2/x1_test.npy', x1_test)
    np.save('./data2/x2_test.npy', x2_test)
    np.save('./data2/y_test.npy', y_test)    

    print("Data loaded successfully!")

    log_dir = "./logs/{}".format(wavelet)
    shutil.rmtree(log_dir, ignore_errors=True)

    callbacks = [
        Initializer("[conv|fc]*.weight", fn=torch.nn.init.kaiming_normal_),
        Initializer("[conv|fc]*.bias", fn=partial(torch.nn.init.constant_, val=0.0)),
        LRScheduler(policy=StepLR, step_size=5, gamma=0.1),
        EpochScoring(scoring=make_scorer(f1_score, average="macro"), lower_is_better=False, name="valid_f1"),
        TensorBoard(SummaryWriter(log_dir)),
        
        Checkpoint(dirname='exp1'),
        TrainEndCheckpoint(dirname='exp1')
    ]
    net = NeuralNetClassifier(  # skorch is extensive package of pytorch for compatible with scikit-learn
        Se_ResNet18(),
#        ResNet18(),
#        MyModule2,
#        Residual(1, 1),
#        MyModule,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        lr=0.001,
        max_epochs=10,
        batch_size=1024,
#        train_split=predefined_split(Dataset({"x": x1_test,"x0": x0_test, "x2": x2_test}, y_test)),
        train_split=predefined_split(Dataset({"x": x1_test, "x2": x2_test}, y_test)),
        verbose=1,
        device="cuda",
        callbacks=callbacks,
        iterator_train__shuffle=True,
        optimizer__weight_decay=0,
    )
#    print(summary(ResNet18(),[x1_train.shape,x2_train.shape]))
    
#    net.fit({"x": x1_train, "x0": x0_train, "x2": x2_train}, y_train)
#    y_true, y_pred = y_test, net.predict({"x": x1_test,"x0": x0_test, "x2": x2_test})
    net.fit({"x": x1_train,  "x2": x2_train}, y_train)
    y_true, y_pred = y_test, net.predict({"x": x1_test, "x2": x2_test})
    

    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=4))

    net.save_params(f_params="./models20220917/model_{}.pkl".format(wavelet))


if __name__ == "__main__":
    

    
    main()
   
#    
    #a = torch.randn((1,1,100,100))
    #b = torch.randn((1,4))
    #net.initialize()
    #print(net.predict({"x": a, "x2": b}))
############################################################################################ 
    
#    x0_train=np.load('./data1/x0_train.npy') 
##    
#    x1_train=np.load('./data/x1_train.npy')
#    x2_train=np.load('./data/x2_train.npy')
#    y_train=np.load('./data/y_train.npy')
##
##    x0_test=np.load('./data1/x0_test.npy')
#    
#    x1_test=np.load('./data/x1_test.npy')
#    x2_test=np.load('./data/x2_test.npy')
#    y_test=np.load('./data/y_test.npy')
##    
#    callbacks = [
#        Initializer("[conv|fc]*.weight", fn=torch.nn.init.kaiming_normal_),
#        Initializer("[conv|fc]*.bias", fn=partial(torch.nn.init.constant_, val=0.0)),
#        LRScheduler(policy=StepLR, step_size=5, gamma=0.1),
#        EpochScoring(scoring=make_scorer(f1_score, average="macro"), lower_is_better=False, name="valid_f1"),
##        TensorBoard(SummaryWriter(log_dir)),
#        
#        Checkpoint(dirname='exp1'),
##        TrainEndCheckpoint(dirname='exp1')
#    ]
#    
#    model_weight = torch.load("./exp1/params.pt")
#    n = MyModule()
##    print(n( a,b))
#    n.load_state_dict(model_weight)
##    print(n( a,b))
#
#    
#    net = NeuralNetClassifier(  # skorch is extensive package of pytorch for compatible with scikit-learn
##        Se_ResNet18(),
##        ResNet18(),
##        MyModule2,
##        Residual(1, 1),
#        n,
#        criterion=torch.nn.CrossEntropyLoss,
#        optimizer=torch.optim.Adam,
#        lr=0.001,
#        max_epochs=20,
#        batch_size=1024,
##        train_split=predefined_split(Dataset({"x": x1_test,"x0": x0_test, "x2": x2_test}, y_test)),
#        train_split=predefined_split(Dataset({"x": x1_test, "x2": x2_test}, y_test)),
#        verbose=1,
#        device="cuda",
#        callbacks=callbacks,
#        iterator_train__shuffle=True,
#        optimizer__weight_decay=0,
#    )
#    
#    
#    
#    net.initialize()
#    
#    net.predict({"x": x1_test, "x2": x2_test})
#    
#    y_true, y_pred = y_test, net.predict({"x": x1_test, "x2": x2_test})
##    
##
#    print(confusion_matrix(y_true, y_pred))
#    print(classification_report(y_true, y_pred, digits=4))
    
##########################################################################    
#    
##    net.load_params(f_params="./exp1/train_end_params.pt")
##    net.fit({"x": x1_train,  "x2": x2_train}, y_train)
#    y_true, y_pred = y_test, net.predict({"x": x1_test, "x2": x2_test})
#    
#
#    print(confusion_matrix(y_true, y_pred))
#    print(classification_report(y_true, y_pred, digits=4))


