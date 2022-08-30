import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import pandas as pd
import PIL.Image as Image
from model_outlines.deep_models import ClawLSTMEncoder

#IMPORTANT GENERAL STUFF
EPOCHS = 20
BATCH_SIZE = 128 #64
LR = 0.001 #.0004 for GripDecoder
WD = 1e-7
TIMESTEP = 4
USE_GPU = True
EEVEL = True
STOP = True
# MODEL = "CNNLSTM"


#setup image transforms
mean = torch.Tensor([0.485, 0.456, 0.406])
std = torch.Tensor([0.229, 0.224, 0.225])
#need to transform and need to normalize after
transform = transforms.Compose(
        [
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
        lPos = [float(item) for item in self.df["lFin"][index].split(",")]
        rPos = [float(item) for item in self.df["rFin"][index].split(",")]
        cPos = [float(item) for item in self.df['cPos'][index].split(",")]
        # print(main)
        if self.stop:
            stop = self.df['stop'][index]

        # #push them into a single array so that we can output them without much issue here. MIGHT WANT TO CHANGE THIS LATER!!!
        pos = np.array([[lPos,eePos,rPos]])
        main.extend(cPos)
        if self.stop:
            main.append(stop)


        image = Image.open(filename)
        if self.transform is not None:
            image = self.transform(image)
        return image,pos,np.array(main)

        # return image, jointVel,eePos,cPos


trainSet = SimDataset("../sequences/lol.csv", transform, EEVEL, STOP)

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

print_every = 100
dtype = torch.float32
def lossWStop(out,true):
    mL = F.mse_loss(out[:,:-1],true[:,:-1])
    bL = F.binary_cross_entropy(out[:,-1],true[:,-1])
    return mL+.5*bL


def train_model(model,optimizer,epochs=1):
    lossHist = []
    model = model.to(device=device)
    mseLoss = nn.MSELoss()
    for e in range(epochs):
        eLoss = 0
        for t, (x,pos, jv) in enumerate(trainLoader):
        # for t, (x,jv, ep, cp) in enumerate(trainLoader):
            model.train()
            x = x.to(device=device,dtype=dtype)
            pos = pos.to(device=device,dtype=dtype)
            jv = jv.to(device=device,dtype=dtype)

            # ep = ep.to(device=device,dtype=torch.long)
            # cp = cp.to(device=device,dtype=torch.long)
            # print(x.shape)
            # print("-----")

            out = model(x,pos)
            if not STOP:
                loss = mseLoss(out,jv)
            else:
                loss = lossWStop(out,jv)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if t % print_every == 0:
                print('Epoch: %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))
            eLoss += loss.item()
        print("Epoch: %d, Loss Avg: %.4f" % (e,eLoss/t))
        lossHist.append(eLoss/t)
    return lossHist

torch.cuda.empty_cache()
if EEVEL:
    numparam = 9
else:
    numparam = 10
model = ClawLSTMEncoder(stop=STOP,num_outputs=numparam) #same sizing for both ResNet34 and 50 depending on the type of residual layer used
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of parameters is: {}".format(params))

lossHist = train_model(model, optimizer, epochs = EPOCHS)

print(min(lossHist), np.argmin(np.array(lossHist)))

# save the model
torch.save(model.state_dict(), '../trained_models/dConvGripLSTM1.pt')






