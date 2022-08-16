import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import sampler
from torchvision import datasets, transforms
from e2cnn import gspaces
from e2cnn import nn

    #basically, we want to make a network that takes in the image and outputs the translational components of the end effector
    #regular for hidden layers and trivial for the final layer

class equCNNTest(torch.nn.Module):
    def __init__(self,num_outputs = 11, stop = False):
        super(equCNNTest, self).__init__()
        self.stop = stop

        if stop:
            self.num_outputs = num_outputs + 1
        else:
            self.num_outputs = num_outputs

        self.num_outputs = num_outputs
        self.r2_act = gspaces.Rot2dOnR2(N=8)
        in_type = nn.FieldType(self.r2_act,3*[self.r2_act.trivial_repr])
        self.in_type = in_type
        out_type = nn.FieldType(self.r2_act,32*[self.r2_act.regular_repr])

        self.block0 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool0 = nn.PointwiseMaxPool(out_type, kernel_size=3,stride=2, padding=1)

        in_type = self.block0.out_type
        out_type = nn.FieldType(self.r2_act,64*[self.r2_act.regular_repr])
        self.block1 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool1 = nn.PointwiseMaxPool(out_type, kernel_size=3, stride=2, padding=1)

        in_type = self.block1.out_type
        out_type = nn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr])
        self.block2 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool2 = nn.PointwiseMaxPool(out_type, kernel_size=3, stride=2, padding=1)

        self.gpool = nn.GroupPooling(out_type)

        c = self.gpool.out_type.size

        self.fc = torch.nn.Linear(c*64, self.num_outputs)

    def forward(self,x):
        x = nn.GeometricTensor(x, self.in_type)
        x=self.block0(x)
        x=self.pool0(x)
        x=self.block1(x)
        x=self.pool1(x)
        x=self.block2(x)
        x=self.pool2(x)
        x=self.gpool(x)
        x=x.tensor
        out=self.fc(x.reshape(x.shape[0], -1))

        if self.stop:
            out[:,-1] = torch.sigmoid(out[:,-1])

        return out

class seperate_stop_eCNN_GRU(torch.nn.Module):
    def __init__(self,num_outputs = 11, stop = False):
        super(seperate_stop_eCNN_GRU, self).__init__()
        self.stop = stop

        self.num_outputs = num_outputs
        self.r2_act = gspaces.Rot2dOnR2(N=8)
        in_type = nn.FieldType(self.r2_act,3*[self.r2_act.trivial_repr])
        self.in_type = in_type
        out_type = nn.FieldType(self.r2_act,32*[self.r2_act.regular_repr])

        self.block0 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool0 = nn.PointwiseMaxPool(out_type, kernel_size=3,stride=2, padding=1)

        in_type = self.block0.out_type
        out_type = nn.FieldType(self.r2_act,64*[self.r2_act.regular_repr])
        self.block1 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool1 = nn.PointwiseMaxPool(out_type, kernel_size=3, stride=2, padding=1)

        in_type = self.block1.out_type
        out_type = nn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr])
        self.block2 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool2 = nn.PointwiseMaxPool(out_type, kernel_size=3, stride=2, padding=1)

        c = self.pool2.out_type.size

        self.fc = torch.nn.Linear(c*64, 128)

        self.rnn = torch.nn.GRU(
            input_size = 128,
            hidden_size= 64,
            num_layers= 1,
            batch_first=True
        )
        self.out = torch.nn.Linear(64,self.num_outputs)
        self.stop = torch.nn.Linear(64,1)

    def start_newSeq(self):
        self.h = torch.zeros((1,64))
    def forward(self,x):
        x = nn.GeometricTensor(x, self.in_type)
        x=self.block0(x)
        x=self.pool0(x)
        x = self.block1(x)
        x=self.pool1(x)
        x=self.block2(x)
        x=self.pool2(x)
        x = x.tensor
        if self.training:
            x = x.view(x.size(0),-1)
            x = self.fc(x)
            x, _ = self.rnn(x)

        else:
            x = x.view(x.size(0),-1)
            x = self.fc(x)
            x, self.h = self.rnn(x, self.h)
        out = self.out(x)
        stop = torch.sigmoid(self.stop(x))



        return out, stop
class equCNNLSTM(torch.nn.Module):
    def __init__(self,num_outputs = 12, stop = True):
        super(equCNNLSTM, self).__init__()
        self.stop = stop

        if stop:
            self.num_outputs = num_outputs + 1
        else:
            self.num_outputs = num_outputs

        self.r2_act = gspaces.Rot2dOnR2(N=8)
        in_type = nn.FieldType(self.r2_act,3*[self.r2_act.trivial_repr])
        self.in_type = in_type
        out_type = nn.FieldType(self.r2_act,32*[self.r2_act.regular_repr])

        self.block0 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool0 = nn.PointwiseMaxPool(out_type, kernel_size=3,stride=2, padding=1)

        in_type = self.block0.out_type
        out_type = nn.FieldType(self.r2_act,64*[self.r2_act.regular_repr])
        self.block1 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool1 = nn.PointwiseMaxPool(out_type, kernel_size=3, stride=2, padding=1)

        in_type = self.block1.out_type
        out_type = nn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr])
        self.block2 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool2 = nn.PointwiseMaxPool(out_type, kernel_size=3, stride=2, padding=1)

        c = self.pool2.out_type.size

        self.fc = torch.nn.Linear(c*64, 128)

        self.rnn = torch.nn.LSTM(
            input_size = 128,
            hidden_size= 64,
            num_layers= 1,
            batch_first=True
        )
        self.out = torch.nn.Linear(64,self.num_outputs)
        print(self.num_outputs)
    def start_newSeq(self):
        self.h = torch.zeros((1,64))
        self.c = torch.zeros((1,64))
    def forward(self,x):
        x = nn.GeometricTensor(x, self.in_type)
        x=self.block0(x)
        x=self.pool0(x)
        x = self.block1(x)
        x=self.pool1(x)
        x=self.block2(x)
        x=self.pool2(x)
        x = x.tensor
        if self.training:
            x = x.view(x.size(0),-1)
            x = self.fc(x)
            x, _ = self.rnn(x)

        else:
            x = x.view(x.size(0),-1)
            x = self.fc(x)
            x, (self.h, self.c) = self.rnn(x, (self.h,self.c))
        out = self.out(x)
        if self.stop:
            out[:,-1] = torch.sigmoid(out[:,-1])

        return out

class dihCNNLSTM(torch.nn.Module):
    def __init__(self,num_outputs = 12, stop = True):
        super(dihCNNLSTM, self).__init__()
        self.stop = stop

        if stop:
            self.num_outputs = num_outputs + 1
        else:
            self.num_outputs = num_outputs

        self.r2_act = gspaces.FlipRot2dOnR2(N=4)
        in_type = nn.FieldType(self.r2_act,3*[self.r2_act.trivial_repr])
        self.in_type = in_type
        out_type = nn.FieldType(self.r2_act,32*[self.r2_act.regular_repr])

        self.block0 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool0 = nn.PointwiseMaxPool(out_type, kernel_size=3,stride=2, padding=1)

        in_type = self.block0.out_type
        out_type = nn.FieldType(self.r2_act,64*[self.r2_act.regular_repr])
        self.block1 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool1 = nn.PointwiseMaxPool(out_type, kernel_size=3, stride=2, padding=1)

        in_type = self.block1.out_type
        out_type = nn.FieldType(self.r2_act, 128 * [self.r2_act.trivial_repr])
        self.block2 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool2 = nn.PointwiseMaxPool(out_type, kernel_size=3, stride=2, padding=1)

        c = self.pool2.out_type.size

        self.fc = torch.nn.Linear(c*64, 128)

        self.rnn = torch.nn.LSTM(
            input_size = 128,
            hidden_size= 64,
            num_layers= 1,
            batch_first=True
        )
        self.out = torch.nn.Linear(64,self.num_outputs)
        print(self.num_outputs)
    def start_newSeq(self):
        self.h = torch.zeros((1,64))
        self.c = torch.zeros((1,64))
    def forward(self,x):
        x = nn.GeometricTensor(x, self.in_type)
        x=self.block0(x)
        x=self.pool0(x)
        x = self.block1(x)
        x=self.pool1(x)
        x=self.block2(x)
        x=self.pool2(x)
        x = x.tensor
        if self.training:
            x = x.view(x.size(0),-1)
            x = self.fc(x)
            x, _ = self.rnn(x)

        else:
            x = x.view(x.size(0),-1)
            x = self.fc(x)
            x, (self.h, self.c) = self.rnn(x, (self.h,self.c))
        out = self.out(x)
        if self.stop:
            out[:,-1] = torch.sigmoid(out[:,-1])

        return out

class EquivariantCNNCom(torch.nn.Module):
    def __init__(self,num_outputs = 12, stop = True):
        super(EquivariantCNNCom, self).__init__()
        self.stop = stop

        if stop:
            self.num_outputs = num_outputs + 1
        else:
            self.num_outputs = num_outputs

        self.r2_act = gspaces.FlipRot2dOnR2(N=4)
        in_type = nn.FieldType(self.r2_act,3*[self.r2_act.trivial_repr])
        self.in_type = in_type
        out_type = nn.FieldType(self.r2_act,32*[self.r2_act.regular_repr])

        self.block0 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool0 = nn.PointwiseMaxPool(out_type, kernel_size=3,stride=2, padding=1)

        in_type = self.block0.out_type
        out_type = nn.FieldType(self.r2_act,64*[self.r2_act.regular_repr])
        self.block1 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool1 = nn.PointwiseMaxPool(out_type, kernel_size=3, stride=2, padding=1)

        in_type = self.block1.out_type
        out_type = nn.FieldType(self.r2_act, 128 * [self.r2_act.trivial_repr])
        self.block2 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool2 = nn.PointwiseMaxPool(out_type, kernel_size=3, stride=2, padding=1)

        c = self.pool2.out_type.size

        self.fc = torch.nn.Linear(c*64, self.num_outputs)

    def forward(self,x):
        x = nn.GeometricTensor(x, self.in_type)
        x=self.block0(x)
        x=self.pool0(x)
        x = self.block1(x)
        x=self.pool1(x)
        x=self.block2(x)
        x=self.pool2(x)
        x = x.tensor
        x = x.view(x.size(0),-1)
        out = self.fc(x)
        if self.stop:
            out[:,-1] = torch.sigmoid(out[:,-1])

        return out