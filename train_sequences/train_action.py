import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from model_outlines.action_models import UNet, EqUNet, EqUNetFloor

#IMPORTANT GENERAL STUFF
EPOCHS = 30 #orient 10
BATCH_SIZE = 16
LR = 0.001 #.0001 for all trained_models except for normal simple equ
WD = 1e-7
USE_GPU = True
EQ = True
C8 = False
Floor = False
Padding = False


#setup image transforms
mean = torch.Tensor([0.485, 0.456, 0.406])
std = torch.Tensor([0.229, 0.224, 0.225])
#need to transform and need to normalize after
if Padding:
    transform = transforms.Compose(
            [
                transforms.Pad(4),
                transforms.ToTensor(),
                transforms.Normalize(mean.tolist(), std.tolist())
            ]
        )
    transformOut =  transforms.Compose([
                transforms.Pad(4),
                transforms.ToTensor(),
            ])
else:
    transform = transforms.Compose(
            [
                # transforms.Pad(4),
                transforms.ToTensor(),
                transforms.Normalize(mean.tolist(), std.tolist())
            ]
        )
    transformOut =  transforms.Compose([
                # transforms.Pad(4),
                transforms.ToTensor(),
            ])


#Create the dataloader. You may want to create a validation and training set for later too.
class SimDataset(Dataset):
    def __init__(self,csv_path,transform = None,outTransform = None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.outTransform = outTransform

    def __len__(self):
        return len(self.df)


    def __getitem__(self, index):
        filename0 = "../" +self.df["imLoc"][index]
        filename1 = "../" +self.df['outLoc'][index]

        cPos = [float(item) for item in self.df['cubePos'][index].split(",")]
        cRot = [float(item) for item in self.df['cubeRot'][index].split(",")]

        inImg = Image.open(filename0)
        outImg = Image.open(filename1)
        if self.transform is not None:
            image = self.transform(inImg)
        outImg = self.outTransform(outImg)
        return image,outImg,np.array(cPos),np.array(cRot)



trainSet = SimDataset("../sequences/action_image.csv", transform,transformOut)

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

print_every = 1
dtype = torch.float32
def lossL1L2(out,truth):
    l1 = nn.L1Loss()
    l2 = nn.MSELoss()
    loss = l1(out,truth) *1 + l2(out,truth) * .5
    return loss
toImage = transforms.ToPILImage()
def train_model(model,optimizer,epochs=1):
    model = model.to(device=device)
    hist = []
    l1Loss = nn.L1Loss()
    for e in range(epochs):
        eHist = []
        for t, (x, out,_,_) in enumerate(trainLoader):
            # for t, (x,jv, ep, cp) in enumerate(trainLoader):
            model.train()

            x = x.to(device=device,dtype=dtype)

            truth = out.to(device=device,dtype=dtype)

            out = model(x)
            loss = lossL1L2(out,truth)
            # loss = l1Loss(out,truth)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            eHist.append(loss.item())
            if t % print_every == 0:
                print('Epoch: %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))
        if EPOCHS >= 5:
            mHist = np.mean(eHist)
            hist.append(mHist)
        else:
            hist.extend(eHist)
    return hist

torch.cuda.empty_cache()
# print(numparam)
#just use ResNet50 for now
if not EQ:
    model = UNet(3,1,bilinear=True) #same sizing for both ResNet34 and 50 depending on the type of residual layer used
elif not Floor and not C8:
    model = EqUNet(n_channels=3,out_channels=1,N=4,flip=True)
elif not Floor:
    model = EqUNet(n_channels=3,out_channels=1,N=8,flip=False)
else:
    model = EqUNetFloor(n_channels=3,out_channels=1,N=4,flip=True)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of parameters is: {}".format(params))

hist = train_model(model, optimizer, epochs = EPOCHS)

# save the model
torch.save(model.state_dict(), '../trained_models/actionEq90Try.pt')

plt.plot(hist)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()





