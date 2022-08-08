import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import sampler
from torchvision import datasets, transforms
from e2cnn import gspaces
from e2cnn import nn

class EquiResBlock(torch.nn.Module): #have to use torch here to make sure it's not e2cnn nn.Module
    #Equivariant Residual Block
    def __init__(self,in_channels, hidden_dim, kernel_size, N, flip=False,quotient=False,initialize = True):
        super(EquiResBlock, self).__init__()

        #Getting the symmetry group action on the image plane R^2. Flip flips it, otherwise, we directly turn it into R62
        if flip:
            r2_act = gspaces.FlipRot2dOnR2(N=N) #N represents the cyclic group that will be used.
        else:
            r2_act = gspaces.Rot2dOnR2(N=N)

        if quotient:
            if flip:
                rep = r2_act.quotient_repr((None,2))
            else:
                rep = r2_act.quotient_repr(2)
        else:
            rep = r2_act.regular_repr #without any changes as explained in the paper

        feat_type_in = nn.FieldType(r2_act, in_channels *[rep])
        feat_type_hid = nn.FieldType(r2_act, hidden_dim * [rep])

        self.layer1 = nn.SequentialModule(
            nn.R2Conv(feat_type_in,feat_type_hid,kernel_size=kernel_size,padding=(kernel_size-1)//2,
                      initialize=initialize),
            nn.ReLU(feat_type_hid)
        )

        self.layer2 = nn.SequentialModule(
            nn.R2Conv(feat_type_hid,feat_type_hid,kernel_size=kernel_size, padding = (kernel_size-1)//2,
                      initialize=initialize)
        )

        self.relu = nn.ReLU(feat_type_hid)

        self.upscale = None

        if in_channels != hidden_dim:
            self.upscale = nn.SequentialModule(
                nn.R2Conv(feat_type_in,feat_type_hid,kernel_size=kernel_size,padding=(kernel_size-1)//2,
                          initialize=initialize)
            )

    def forward(self,x):
        residual = x
        out = self.layer1(x)
        out = self.layer2(out)
        if self.upscale:
            out += self.upscale(residual)
        else:
            out += residual
        out = self.relu(out)

        return out

class conv2d(torch.nn.Module):
    #copy of conv2d but for equivariant models
    def __init__(self, in_channels, out_channels, kernel_size, stride, N, activation=True, triv=False,
                 flip=False, quotient=False, initialize=True):
        super(conv2d, self).__init__()

        #copy of block
        if flip:
            r2_act = gspaces.FlipRot2dOnR2(N=N)  # N represents the cyclic group that will be used.
        else:
            r2_act = gspaces.Rot2dOnR2(N=N)

        if quotient:
            if flip:
                rep = r2_act.quotient_repr((None, 2))
            else:
                rep = r2_act.quotient_repr(2)
        else:
            rep = r2_act.regular_repr  # without any changes as explained in the paper

        feat_type_in = nn.FieldType(r2_act, in_channels * [rep])
        if triv:
            feat_type_hid = nn.FieldType(r2_act, out_channels * [r2_act.trivial_repr]) #if it's the last layer, we want to use the trivial representation
        else:
            feat_type_hid = nn.FieldType(r2_act, out_channels * [rep])

        if activation:
            self.layer = nn.SequentialModule(
                nn.R2Conv(feat_type_in, feat_type_hid, kernel_size=kernel_size, stride=stride,
                          padding=(kernel_size - 1) // 2, initialize=initialize),
                nn.ReLU(feat_type_hid)
            )
        else:
            self.layer = nn.R2Conv(feat_type_in, feat_type_hid, kernel_size=kernel_size, stride=stride,
                                   padding=(kernel_size - 1) // 2, initialize=initialize)

    def forward(self, x):
        return self.layer(x)

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
        x = self.block1(x)
        x=self.pool1(x)
        x=self.block2(x)
        x=self.pool2(x)
        x= self.gpool(x)
        x = x.tensor
        out = self.fc(x.reshape(x.shape[0], -1))

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
    def __init__(self, initialize=True, n_p=2, n_theta=1):
        super().__init__()
        self.n_inv = 3 * n_theta * n_p
        self.r2_act = gspaces.Rot2dOnR2(4)
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.r2_act, 2*[self.r2_act.trivial_repr]),
                      nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr]), 2),
            # 64x64
            nn.R2Conv(nn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr]), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr]), 2),
            # 8x8
            nn.R2Conv(nn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 256 * [self.r2_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.r2_act, 256 * [self.r2_act.regular_repr]), inplace=True),

            nn.R2Conv(nn.FieldType(self.r2_act, 256 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 256 * [self.r2_act.regular_repr]),
                      kernel_size=3, padding=0),
            nn.ReLU(nn.FieldType(self.r2_act, 256 * [self.r2_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.r2_act, 256 * [self.r2_act.regular_repr]), 2),
            # 3x3
            nn.R2Conv(nn.FieldType(self.r2_act, 256 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, self.n_inv * [self.r2_act.trivial_repr]),
                      kernel_size=1, padding=0),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = nn.GeometricTensor(x, nn.FieldType(self.r2_act, 2 * [self.r2_act.trivial_repr]))
        out = self.conv(x).tensor.reshape(batch_size, self.n_inv, 9).permute(0, 2, 1)
        return out