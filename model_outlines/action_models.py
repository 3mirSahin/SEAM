import torch
import torch.nn
import torch.nn.functional as F
from e2cnn import gspaces
from e2cnn import nn

#This here is based off of the UNet code that can be found at https://github.com/milesial/Pytorch-UNet
class DoubleConv(torch.nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None,kernel_size = 3):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=1, bias=False),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(torch.nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(torch.nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = torch.nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
class UNet(torch.nn.Module):
    def __init__(self, n_channels, out_channels, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 8)
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)
        factor = 2 if bilinear else 1
        self.down3 = Down(32, 64 // factor)
        self.up1 = Up(64, 32 // factor, bilinear)
        self.up2 = Up(32, 16 // factor, bilinear)
        self.up3 = Up(16, 8, bilinear)
        self.out = OutConv(8,out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        out = self.out(x)
        return out

#My addition to the above code with equivariant CNNs.
class EqDoubleConv(torch.nn.Module):
    def __init__(self, input_channels, out_channels, mid_channels = None, N=4, flip=False, trivial=False):
        super(EqDoubleConv, self).__init__()

        if flip:
            r2_act = gspaces.FlipRot2dOnR2(N=N)
        else:
            r2_act = gspaces.Rot2dOnR2(N=N)
        rep = r2_act.regular_repr

        if trivial:
            in_type = nn.FieldType(r2_act,input_channels*[r2_act.trivial_repr])
        else:
            in_type = nn.FieldType(r2_act,input_channels*[rep])

        if mid_channels:
            out_type = nn.FieldType(r2_act,mid_channels*[rep])
        else:
            out_type = nn.FieldType(r2_act,out_channels*[rep])

        self.block0 = nn.SequentialModule(
            nn.R2Conv(in_type,out_type,kernel_size=3,padding=1,initialize=True),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type,inplace=True)
        )
        in_type = self.block0.out_type
        out_type = nn.FieldType(r2_act,out_channels*[rep])
        self.block1 = nn.SequentialModule(
            nn.R2Conv(in_type,out_type,kernel_size=3,padding=1,initialize=True),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type,inplace=True)
        )

    def forward(self,x):
        x = self.block0(x)
        return self.block1(x)

class EqDown(torch.nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels,N=4,flip=False):
        super(EqDown,self).__init__()
        if flip:
            r2_act = gspaces.FlipRot2dOnR2(N=N)
        else:
            r2_act = gspaces.Rot2dOnR2(N=N)
        rep = r2_act.regular_repr

        in_type = nn.FieldType(r2_act,in_channels*[rep])

        self.pool = nn.PointwiseMaxPool(in_type,kernel_size=2)
        self.conv = EqDoubleConv(in_channels,out_channels,N=N,flip=flip)

    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)

class EqUp(torch.nn.Module):
    def __init__(self,in_channels,out_channels,N=4,flip=False):
        super(EqUp,self).__init__()
        if flip:
            self.r2_act = gspaces.FlipRot2dOnR2(N=N)
        else:
            self.r2_act = gspaces.Rot2dOnR2(N=N)
        self.rep = self.r2_act.regular_repr
        self.in_channels = in_channels
        self.out_channels = out_channels
        in_type = nn.FieldType(self.r2_act,in_channels*[self.rep])
        out_type = nn.FieldType(self.r2_act,(out_channels)*[self.rep])
        self.upscale1 = nn.R2Upsampling(in_type,2,mode="bilinear")#Get the upscaled version
        # print(c)
        self.upscale2 = nn.R2ConvTransposed(in_type,out_type,kernel_size=3,padding=1,initialize=True) #Too small to work
        self.conv = EqDoubleConv(out_channels*2,out_channels,flip=flip,N=N)

    def forward(self,x2,x1): #x1 is larger
        #given the previous feature map, we want to upscale it to the size of the next one, then concatenate, then conv.
        x1 = x1.tensor
        # print(x1.shape)
        # print(x2.shape)
        x2 = self.upscale1(x2)
        # print(x2.shape)
        x2 = self.upscale2(x2).tensor
        # print(x2.shape)
        concat = torch.cat((x2,x1), dim=1)
        concat = nn.GeometricTensor(concat,nn.FieldType(self.r2_act, self.out_channels*2*[self.rep]))

        return self.conv(concat)
class EqOutConv(torch.nn.Module):
    def __init__(self,in_channels,out_channels,N=4,flip=False):
        super(EqOutConv,self).__init__()
        if flip:
            r2_act = gspaces.FlipRot2dOnR2(N=N)
        else:
            r2_act = gspaces.Rot2dOnR2(N=N)
        rep = r2_act.regular_repr

        in_type = nn.FieldType(r2_act,in_channels*[rep])
        out_type = nn.FieldType(r2_act,out_channels*[r2_act.trivial_repr])
        self.conv = nn.R2Conv(in_type,out_type,kernel_size=1)
    def forward(self,x):
        return self.conv(x)
class EqUNet(torch.nn.Module):
    def __init__(self, n_channels, out_channels, N=4,flip=False):
        super(EqUNet, self).__init__()
        self.n_channels = n_channels
        if flip:
            r2_act = gspaces.FlipRot2dOnR2(N=N)
        else:
            r2_act = gspaces.Rot2dOnR2(N=N)
        self.in_type = nn.FieldType(r2_act,n_channels*[r2_act.trivial_repr])
        self.inc = EqDoubleConv(n_channels, 8, mid_channels=None,N=N,flip=flip,trivial=True)
        self.down1 = EqDown(8, 16,N=N,flip=flip)
        self.down2 = EqDown(16, 32,N=N,flip=flip)
        self.down3 = EqDown(32, 64,N=N,flip=flip)
        self.up1 = EqUp(64, 32,N=N,flip=flip)
        self.up2 = EqUp(32, 16,N=N,flip=flip)
        self.up3 = EqUp(16, 8,N=N,flip=flip)
        self.out = EqOutConv(8,out_channels,N=N,flip=flip)

    def forward(self, x):
        x = nn.GeometricTensor(x,self.in_type)
        #turn x into a geometric tensor
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        out = self.out(x).tensor
        return out

class EqUNetFloor(torch.nn.Module):
    def __init__(self, n_channels, out_channels, N=4,flip=False):
        super(EqUNetFloor, self).__init__()
        self.n_channels = n_channels
        if flip:
            r2_act = gspaces.FlipRot2dOnR2(N=N)
        else:
            r2_act = gspaces.Rot2dOnR2(N=N)
        self.in_type = nn.FieldType(r2_act,n_channels*[r2_act.trivial_repr])
        self.inc = EqDoubleConv(n_channels, 8, mid_channels=None,N=N,flip=flip,trivial=True)
        self.down1 = EqDown(8, 16,N=N,flip=flip)
        self.down2 = EqDown(16, 32,N=N,flip=flip)
        self.down3 = EqDown(32, 32,N=N,flip=flip)
        self.up1 = EqUp(32, 32,N=N,flip=flip)
        self.up2 = EqUp(32, 16,N=N,flip=flip)
        self.up3 = EqUp(16, 8,N=N,flip=flip)
        self.out = EqOutConv(8,out_channels,N=N,flip=flip)

    def forward(self, x):
        x = nn.GeometricTensor(x,self.in_type)
        #turn x into a geometric tensor
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        out = self.out(x).tensor
        return out

#Rotational Models - these are only to determine the final rotation of the end effector when it's reaching for the item. We feed in a cropped image of the item right in to calculate the final rotation

class rotCNN(torch.nn.Module):
    def __init__(self, in_channels, n_rotations, n_primitives, n_hidden=32):
        super(rotCNN,self).__init__()
        self.n_rotations = n_rotations
        self.n_primitives = n_primitives


        n1 = int(n_hidden / 4)
        n2 = int(n_hidden / 2)
        n3 = n_hidden
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, n1, kernel_size=7, padding=3),
            torch.nn.ReLU(),

            DoubleConv(n1, n2, kernel_size=3),
            torch.nn.MaxPool2d(2),

            DoubleConv(n2, n3, kernel_size=3),
            torch.nn.MaxPool2d(2),
        )

        self.conv_2 = torch.nn.Sequential(
            torch.nn.Conv2d(n3, n_primitives * n_rotations, kernel_size=4, padding=0),
        )

    def forward(self, img):
        batch_size = img.size(0)
        x = self.conv(img)

        x = self.conv_2(x)
        # print(x.shape)
        x = x.reshape(batch_size, 2, self.n_primitives, -1)
        # print(x.shape)
        # x = self.softmax(x)[:, 1, :] #softmax is not necessary for ce
        x = x[:,1,:]
        # print(x.shape)
        x = x.reshape(batch_size, self.n_primitives, -1)
        # print(x.shape)
        return x

class rotEqCNN(torch.nn.Module):
    def __init__(self, in_channels, n_rotations, n_primitives, n_hidden=32):
        super(rotEqCNN,self).__init__()
        self.n_rotations = n_rotations
        self.n_primitives = n_primitives
        self.N = n_rotations
        self.in_channels = in_channels

        self.r2_act = gspaces.Rot2dOnR2(N=self.N)
        self.inrep = self.r2_act.trivial_repr
        self.outrep = self.r2_act.regular_repr



        n1 = int(n_hidden / 4)
        n2 = int(n_hidden / 2)
        n3 = n_hidden
        # self.conv = nn.SequentialModule(
        #     nn.R2Conv(
        #         nn.FieldType(self.r2_act, in_channels*[self.inrep]),
        #         nn.FieldType(self.r2_act, n1*[self.outrep]),
        #         kernel_size = 7, padding = 3,initialize=True
        #     ),
        #     nn.ReLU(nn.FieldType(self.r2_act,n1*[self.outrep])),
        #
        #     EqDoubleConv(n1,n2,N=self.N,flip=False),
        #     nn.PointwiseMaxPool(nn.FieldType(self.r2_act,n2*[self.outrep]),2),
        #
        #     EqDoubleConv(n2, n3, N=self.N, flip=False),
        #     nn.PointwiseMaxPool(nn.FieldType(self.r2_act, n3 * [self.outrep]),2)
        # )
        self.conv_0 = nn.SequentialModule(
            nn.R2Conv(
                nn.FieldType(self.r2_act, in_channels*[self.r2_act.trivial_repr]),
                nn.FieldType(self.r2_act, n1*[self.r2_act.regular_repr]),
                kernel_size = 7, padding = 3,initialize=True),
            nn.ReLU(nn.FieldType(self.r2_act,n1*[self.outrep])),)
        self.conv_1 = EqDoubleConv(n1,n2,N=self.N,flip=False)
        self.pool_1 = nn.PointwiseMaxPool(nn.FieldType(self.r2_act,n2*[self.outrep]),2)

        self.conv_2 = EqDoubleConv(n2,n3,N=self.N,flip=False)
        self.pool_2 = nn.PointwiseMaxPool(nn.FieldType(self.r2_act,n3*[self.outrep]),2)

        output_rep = n_primitives * [self.outrep]
        self.conv_3 = nn.R2Conv(
            nn.FieldType(self.r2_act, n3 * [self.outrep]),
            nn.FieldType(self.r2_act, output_rep),
            kernel_size=4, padding=0, initialize=True
        )

    def forward(self, img):
        batch_size = img.size(0)
        img = nn.GeometricTensor(img,
                                 nn.FieldType(self.r2_act, self.in_channels * [self.inrep]))
        # x = self.conv(img)
        x = self.conv_0(img)
        x = self.conv_1(x)
        x = self.pool_1(x)
        x = self.conv_2(x)
        x = self.pool_2(x)
        x = self.conv_3(x).tensor
        x = x.reshape(batch_size, 2, self.n_primitives, -1)
        x = x[:,1,:]
        x = x.reshape(batch_size, self.n_primitives, -1)
        return x

