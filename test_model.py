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
from pyrep.objects.camera import Camera
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.backend import sim

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

from SimpleCNNModel import SCNN
from CNNLSTMModel import CNNLSTM

STOP = False
EEVEL = True
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

if EEVEL:
    numparam = 12
else:
    numparam = 13
model = CNNLSTM(stop=STOP,num_outputs=numparam)
model.load_state_dict(torch.load("models/ee400LSTMModel.pt"))
model.eval()
model.start_newSeq()

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

stops = []

read = pd.read_csv("lol.csv")


def get_trueJacobian(robot):
    robot._ik_target.set_matrix(robot._ik_tip.get_matrix())
    sim.simCheckIkGroup(robot._ik_group,
                        [j.get_handle() for j in robot.joints])
    jacobian, (rows, cols) = sim.simGetIkGroupMatrix(robot._ik_group, 0)
    jacobian = np.array(jacobian).reshape((rows, cols), order='F')
    return np.flip(jacobian, axis=0)
def get_jointVelo(robot, v: np.ndarray) -> np.ndarray:
    robot.set_ik_element_properties(constraint_alpha_beta=True, constraint_gamma=True)
    # v = v[:3]
    # R = robot.get_matrix()[:3,:3]
    # v = np.matmul(np.linalg.inv(R), v)
    J = get_trueJacobian(robot)
    # J = robot.get_jacobian()
    # print(J.T.shape)
    # print(v.shape)
    q = np.matmul(np.linalg.pinv(J.T), v)
    return q

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
    if EEVEL:

        # jVel = [float(item) for item in read['jVel'][count].split(",")]
        # eeVel = [float(item) for item in read['eeJacVel'][count].split(",")]
        eeVel = res[0][:6]



        # jacob = get_trueJacobian(agent).T
        # jacob = agent.get_jacobian().T

        # eVel = jacob@jVel
        # print(eeVel)
        # print(eVel)
        # print("-----")

        #let's feed in the training data

        jointVel = get_jointVelo(agent,eeVel)
        # jointVel = np.flip(jointVel,axis=0)
        # jointVel = jVel

        # jointVel = np.flip(jointVel,axis=0)
        # use the jacobian to calculate jointvel
    else:
        jointVel = res[0][:7]
    # print(jointVel)
    if res[0][-1] >= .5 and STOP:
        done = True

    agent.set_joint_target_velocities(jointVel)
    pr.step()
    count+=1
    dist = ps.read()
    # print(dist)
    stops.append(res[0][-1])
    if dist<=.11 and dist > 0:
        done = True
    if count >= 100:
        done = True

print(max(stops))


pr.stop()
pr.shutdown()








