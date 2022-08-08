import torch
from torch.nn import Conv2d, MaxPool2d
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import sampler
from torchvision import datasets, transforms
from os.path import dirname, join, abspath
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import PIL
import PIL.Image as Image
import random
from deep_models import CNNLSTM, SCNN
from equivariant_models import equCNNTest, equCNNLSTM, seperate_stop_eCNN_GRU, dihCNNLSTM
import e2cnn.nn

#IMPORTANT GENERAL STUFF
EPOCHS = 20 #orient 10
BATCH_SIZE = 32
LR = 0.0001 #.0001 for all models except for normal simple equ
WD = 1e-7
USE_GPU = True
EEVEL = True
STOP = True


#setup image transforms
mean = torch.Tensor([0.485, 0.456, 0.406])
std = torch.Tensor([0.229, 0.224, 0.225])
#need to transform and need to normalize after
transform = transforms.Compose(
        [
            # transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.RandomRotation((0,180)),
            transforms.ToTensor(),
            transforms.Normalize(mean.tolist(), std.tolist())
        ]
    )

#Create the dataloader. You may want to create a validation and training set for later too.
class SimDataset(Dataset):
    def __init__(self,csv_path,transform = None,eeVel=False,stop=False):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.eeVel = eeVel
        self.stop = stop

    def __len__(self):
        return len(self.df)


    def __getitem__(self, index):
        filename = self.df["imLoc"][index]
        if not self.eeVel:
            jointVel = [float(item) for item in self.df['jVel'][index].split(",")]
            main = jointVel
        else:
            eeVel = [float(item) for item in self.df['eeJacVel'][index].split(",")]
            main = eeVel
        eePos = [float(item) for item in self.df['eePos'][index].split(",")]
        cPos = [float(item) for item in self.df['cPos'][index].split(",")]
        # print(main)
        if self.stop:
            stop = self.df['stop'][index]

        # #push them into a single array so that we can output them without much issue here. MIGHT WANT TO CHANGE THIS LATER!!!

        main.extend(eePos)
        main.extend(cPos)
        if self.stop:
            main.append(stop)


        image = Image.open(filename)
        if self.transform is not None:
            image = self.transform(image)
        return image,np.array(main)

        # return image, jointVel,eePos,cPos


trainSet = SimDataset("side.csv",transform,EEVEL,STOP)

# print(len(trainSet[0][1])+len(trainSet[0][2])+len(trainSet[0][3]))
#May want to create a trainining and validation set for later

#dataset loader - for each sample, 0 gives image, 1 gives joint vels, 2 gives eepos, 3 gives cPos
trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, num_workers=0)

#---------- MODEL SETUP -------------


#----- RUN THE MODEL -----
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

print_every = 10
dtype = torch.float32
def lossWStop(out,true):
    mL = F.mse_loss(out[:,:-1],true[:,:-1])
    bL = F.binary_cross_entropy(out[:,-1],true[:,-1])
    return mL+.4*bL
def sepLossWStop(out,stop,true):
    mL = F.mse_loss(out,true[:,:-1])
    # print(stop)
    # print(true[:,-1])
    bL = F.binary_cross_entropy(stop,true[:,-1])
    return mL+.4*bL

def train_model(model,optimizer,epochs=1):
    model = model.to(device=device)
    mseLoss = nn.MSELoss()
    hist = []

    for e in range(epochs):
        eHist = []
        for t, (x, jv) in enumerate(trainLoader):
            # for t, (x,jv, ep, cp) in enumerate(trainLoader):
            model.train()
            x = x.to(device=device,dtype=dtype)
            # jv.extend(ep)
            # jv.extend(cp)
            jv = jv.to(device=device,dtype=dtype)

            # ep = ep.to(device=device,dtype=torch.long)
            # cp = cp.to(device=device,dtype=torch.long)

            out = model(x)
            # out, stop = model(x)
            # stop = torch.squeeze(stop)
            # print(out.shape)
            # print(jv.shape)

            if not STOP:
                loss = mseLoss(out,jv)
            else:
                loss = lossWStop(out,jv)
                # loss = sepLossWStop(out,stop,jv)
            # print(loss)
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            eHist.append(loss.item())
            if t % print_every == 0:
                print('Epoch: %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))
        mHist = np.mean(eHist)
        hist.append(mHist)
    return hist

torch.cuda.empty_cache()
if EEVEL:
    numparam = 12
else:
    numparam = 13
# print(numparam)
model = dihCNNLSTM(stop=STOP,num_outputs=numparam) #same sizing for both ResNet34 and 50 depending on the type of residual layer used
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of parameters is: {}".format(params))

hist = train_model(model, optimizer, epochs = EPOCHS)

# save the model
torch.save(model.state_dict(), 'models/dihLSTMSide.pt')

plt.plot(hist)
plt.show()





