import torch
from torch.nn import Conv2d, MaxPool2d, LSTM
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import sampler
from torchvision import datasets, transforms
from SimpleCNNModel import SCNN


class CNNLSTM(nn.Module):
    def __init__(self,num_outputs = 13,fconv=[3,1,1],stop = False):
        super(CNNLSTM,self).__init__()
        self.num_outputs = num_outputs

        self.conv1 = self.conv_layer(3, 64)
        self.conv2 = self.conv_layer(64, 128)
        self.conv3 = self.conv_layer(128, 256)


        self.out = nn.Linear(128,self.num_outputs)

        self.fc = nn.Linear(256*8*8,256)
        self.rnn = nn.LSTM(
            input_size = 256,
            hidden_size= 128,
            num_layers= 1,
            batch_first=True
        )
        self.linear = nn.Linear(64,num_outputs)

        self.h, self.c = None,None
    def conv_layer(
        self,
        chIN,
        chOUT,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        pool_kernel=3,
        pool_stride=2,
        pool_padding=1
    ):
        conv = nn.Sequential(nn.Conv2d(chIN, chOUT, kernel_size, stride, padding, bias=bias),
                             nn.BatchNorm2d(chOUT),
                             nn.MaxPool2d(pool_kernel, pool_stride, pool_padding),
                             nn.ReLU(inplace=True))
        return conv
    def start_newSeq(self):
        self.h = torch.zeros((1,128))
        self.c = torch.zeros((1,128))
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.training:
            x = x.view(x.size(0),-1)
            x = self.fc(x)
            x, _ = self.rnn(x)

        else:
            x = x.view(x.size(0),-1)
            x = self.fc(x)
            x, (self.h, self.c) = self.rnn(x, (self.h,self.c))

        return self.out(x)

