"""
A Franka Panda reaches for 10 randomly places targets.
This script contains examples of:
    - Linear (IK) paths.
    - Scene manipulation (creating an object and moving it).
"""
from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
from pyrep.errors import ConfigurationPathError
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.proximity_sensor import ProximitySensor

import numpy as np
import math
import pandas as pd
import os
import PIL
from PIL import Image
from pyrep.objects.joint import JointMode

import torch
from torch.nn import Conv2d, MaxPool2d
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import sampler
from torchvision import datasets, transforms

#back to work
SCENE_FILE = join(dirname(abspath(__file__)), 'simulations/scene_panda_reach_target.ttt')
pr = PyRep()
pr.launch(SCENE_FILE, headless=False)
pr.start()
agent = Panda()
agent_ee_tip = agent.get_tip()
agent.reset_dynamic_object()
agent.set_control_loop_enabled(False)
# agent.set_joint_target_positions([Torque,True,True,True,True,True,True])
vs = VisionSensor("Vision_sensor")
vs.set_resolution([64,64])
ps = ProximitySensor("Proximity_sensor")

cube= Shape.create(type=PrimitiveShape.CUBOID,
                      size=[0.05, 0.05, 0.05],
                      color=[1.0, 0.1, 0.1],
                      static=True, respondable=False)
target = Dummy.create()
cube_size = .1
table= Shape('diningTable_visible')

device = torch.device('cpu')

mean = torch.Tensor([0.485, 0.456, 0.406])
std = torch.Tensor([0.229, 0.224, 0.225])
#need to transform and need to normalize after
transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean.tolist(), std.tolist())
        ]
    )

class ResNet(nn.Module):
    def __init__(self,num_outputs = 13,fconv=[3,1,1]):
        super(ResNet,self).__init__()

        self.conv1 = nn.Sequential(Conv2d(3,64,kernel_size= fconv[0],stride=fconv[1],padding=fconv[2],bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(Conv2d(64,128,kernel_size=fconv[0],stride=fconv[1],padding=fconv[2],bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(Conv2d(128, 256, kernel_size=fconv[0], stride=fconv[1], padding=fconv[2], bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(inplace=True))
        self.fc = nn.Linear(256*8*8,num_outputs)

    def forward(self,x):

        x = self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)

        return x

model = ResNet()
model.load_state_dict(torch.load("models/model.pt"))
model.eval()

cube_min_max = table.get_bounding_box()
cube_min_max = [cube_min_max[0] + cube_size,
                        .6 - cube_size,
                        cube_min_max[2] + cube_size,
                        cube_min_max[3] - cube_size,
                        cube_min_max[5] - .05]

position_min, position_max = [cube_min_max[0], cube_min_max[2], cube_min_max[3]], [cube_min_max[1],
                                                                                           cube_min_max[3],
                                                                                           cube_min_max[3]]


def replaceCube():
    pos = list(np.random.uniform(position_min, position_max))
    cube.set_position(pos, table)
    try:
        pp = agent.get_linear_path(
            position=cube.get_position(),
            euler=[0, math.radians(180), 0],
            steps=100
        )
    except ConfigurationPathError as e:
        print("Cube bad placement. Replacing.")
        replaceCube()


replaceCube()

count = 0
done = False
pr.step()
while not done:
    #take the image from the robot
    img = vs.capture_rgb()
    img = Image.fromarray((img * 255).astype(np.uint8)).resize((64, 64)).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)
    #shove it into the model
    res = model(img)
    res = res.tolist()
    # print(res)
    #only take the joint velocities
    jointVel = res[0][:7]
    print(jointVel)

    agent.set_joint_target_velocities(jointVel)
    pr.step()
    count+=1
    dist = ps.read()
    print(dist)
    if dist<=.11 and dist > 0:
        done = True




pr.stop()
pr.shutdown()








