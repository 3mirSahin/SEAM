import torch
from torch.nn import Conv2d, MaxPool2d, LSTM
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import sampler
from torchvision import datasets, transforms
from SimpleCNNModel import SCNN

class CNN(nn.Module):
    def __init__(self):
        super(SCNN).__init__()
        self.fc = nn.Linear(256*8*8,320)
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
class CNNLSTM(nn.Module):
    def __init__(self,num_outputs = 13):
        super(CNNLSTM,self).__init__()
        self.cnn = SCNN()
        self.rnn = nn.LSTM(
            input_size = 320,
            hidden_size= 64,
            num_layers= 1,
            batch_first=True
        )
        self.linear = nn.Linear(64,num_outputs)
    def forward(self,x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, (h_n, h_c) = self.rnn(r_in)
        r_out2 = self.linear(r_out[:, -1, :])

        return r_out2