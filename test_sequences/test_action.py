from os.path import dirname, join, abspath
import os
import glob
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
from pyrep.errors import ConfigurationPathError
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.backend import sim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import math
from PIL import Image, ImageOps
import torch
from torchvision import transforms


from model_outlines.action_models import UNet, EqUNet, EqUNetFloor

'''Configurations for the test instance.'''
RUNS = 50 #the total number of test attempts done. Changes the cube location.
TURN_EE = False
if TURN_EE:
    SCENE_FILE = join(dirname(abspath(__file__)), '../simulations/scene_panda_reach_action_image_turned.ttt')
else:
    SCENE_FILE = join(dirname(abspath(__file__)), '../simulations/scene_panda_reach_action_image.ttt')
TEST_ORIENT = False
TEST_90 = False
EQ = True
C8 = True
Floor = False
Padding = False
RECT = True

# FlipObject = True


'''Model Hyperparameters'''
device = torch.device('cpu')

mean = torch.Tensor([0.485, 0.456, 0.406])
std = torch.Tensor([0.229, 0.224, 0.225])
#need to transform and need to normalize after
transform = transforms.Compose(
        [
            # transforms.RandomVerticalFlip(1),
            # transforms.RandomHorizontalFlip(1),
            # transforms.functional.rotate(),
            transforms.ToTensor(),
            transforms.Normalize(mean.tolist(), std.tolist())
        ]
    )

'''Model Choice'''
if EQ and Floor:
    model = EqUNetFloor(3,1,N=4,flip=True)
elif EQ and C8:
    model = EqUNet(3,1,N=8,flip=False)
elif EQ:
    model = EqUNet(3,1,N=4,flip=True)
else:
    model = UNet(3,1,bilinear=True)

# model = UNet(3,1,bilinear=True)
model.train()
model.load_state_dict(torch.load("../trained_models/actionEq90Try.pt")) #change this based on model
model.eval()

orient_min, orient_max = [0,0,math.radians(-90)], [0,0,math.radians(90)]

'''PyRep Setup'''
pr = PyRep()

pr.launch(SCENE_FILE, headless=False)
pr.start()
agent = Panda()
agent_ee_tip = agent.get_tip()
agent.reset_dynamic_object()
agent.set_control_loop_enabled(True)
agent_state = agent.get_configuration_tree()
initial_joint_position = agent.get_joint_positions()
# agent.set_joint_target_positions([Torque,True,True,True,True,True,True])
vs = VisionSensor("Vision_sensor")
vs.set_resolution([64,64])
ps = ProximitySensor("Proximity_sensor")

cube_size = .025
if RECT:
    cube_sizes = [cube_size, cube_size * 3, cube_size]
else:
    cube_sizes = [cube_size, cube_size, cube_size]
cube = Shape.create(type=PrimitiveShape.CUBOID,
                         size=cube_sizes,
                         color=[1.0, 0.1, 0.1],
                         static=True, respondable=False)

top_right_dum = Dummy("top_right")
bot_left_dum = Dummy("bot_left")
top_right = top_right_dum.get_position()
bot_left = bot_left_dum.get_position()


'''Cube Movement'''

cube_min_max = [bot_left[0] - cube_size / 2,
                top_right[0] + cube_size / 2,
                bot_left[1] - cube_size / 2,
                top_right[1] + cube_size / 2,
                cube_size / 2]

position_min, position_max = [cube_min_max[0], cube_min_max[2], cube_min_max[4]], [cube_min_max[1],
                                                                                           cube_min_max[3],
                                                                                           cube_min_max[4]]
def resetEnv():
    agent.set_joint_target_velocities(np.zeros_like(agent.get_joint_target_velocities()))
    agent.set_motor_locked_at_zero_velocity(True)

    agent.reset_dynamic_object()
    pr.set_configuration_tree(agent_state)

    agent.set_joint_positions(initial_joint_position,disable_dynamics=True)

def replaceCube():
    pos = list(np.random.uniform(position_min, position_max))
    # print(pos)
    rot = list(np.random.uniform(orient_min, orient_max))
    cube.set_position(pos)
    if TEST_ORIENT:
        cube.set_orientation(rot) #is table really the correct thing to base the orientation off of?
    elif TEST_90:
        cube.set_orientation([0,0,math.radians(90)])
    try:
        pp = agent.get_linear_path(
            position=cube.get_position(),
            euler=[0, math.radians(180), 0],
            steps=100
        )
    except ConfigurationPathError as e:
        print("Cube bad placement. Replacing.")
        replaceCube()
def get_path(object):
    ori = cube.get_orientation()
    try:
        path = agent.get_linear_path(
            position=object.get_position(), euler=[0, math.radians(180), math.radians(90)-ori[2]])
        return path, False
    except ConfigurationPathError as e:
        path = agent.get_path(
            position=cube.get_position(),
            euler=[0, math.radians(180), 0]
        )
    # print(path._path_points)
    return path, True

def pixelToCoord(img,sensor_size=(32,32)):
    #need to mirror image
    rot = math.degrees(agent_ee_tip.get_orientation()[2])
    img = img.rotate(rot)
    # img = ImageOps.flip(img)
    # print(img.size)
    pixel = np.array(img)
    tR = top_right_dum.get_position(relative_to=agent_ee_tip)
    bL = bot_left_dum.get_position(relative_to=agent_ee_tip)
    pLoc = np.unravel_index(pixel.argmax(), pixel.shape)

    x = bL[0] - tR[0]
    y = bL[1] - tR[1]


    x = (pLoc[0] /sensor_size[0] * x) + tR[0]
    y = (pLoc[1]/sensor_size[1] * y) + tR[1]

    return x,y
def generateTarget(x,y,z=cube.get_position(relative_to=agent_ee_tip)[2]-.025):
    target = Dummy.create()
    target.set_position([x,y,z],relative_to=agent_ee_tip)
    # print(x,y,z)
    return target

def checkEEBoundary(ee, target):
    #get the position of both the end effector and the cube
    eePos = ee.get_position()
    cubePos = target.get_position()
    #check the boundaries of the cube.
    if eePos[0] > cubePos[0]-cube_size and eePos[0] < cubePos[0]+cube_size:
        if eePos[1] > cubePos[1]-cube_size and eePos[1] < cubePos[1]+cube_size:
            if eePos[2] > cubePos[2]-cube_size and eePos[2] < cubePos[2]+cube_size:
                return True
    return False

def removePad(img,padSize):
    # img = toImage(img)
    border = (padSize,padSize,padSize,padSize)
    ImageOps.crop(img,border)
    return img


if not os.path.isdir(f"../Resulting Images/Verticle Horizontal"):
    os.mkdir(f"../Resulting Images/Verticle Horizontal")
files = glob.glob('../Resulting Images/Verticle Horizontal/*')
for f in files:
    os.remove(f)
correct = 0
for tryy in range(RUNS):
    resetEnv()
    replaceCube()

    count = 0
    done = False
    pr.step()

    stops = []
    img = vs.capture_rgb()
    img = Image.fromarray((img * 255).astype(np.uint8)).resize((64, 64)).convert('RGB')
    imq = img.copy()
    # plt.imshow(img)
    # plt.show()
    # img.show()
    # img = transforms.functional.rotate(img,180)
    img = transform(img)
    img = img.unsqueeze(0)

    # shove it into the model
    # res, stop = model(img)
    res = model(img).squeeze().squeeze()
    # print(res)
    toImage = transforms.ToPILImage()
    res = toImage(res)
    if EQ and Padding:
        res = removePad(res,4)
    # plt.imshow(res)
    # plt.show()
    x,y = pixelToCoord(res,(64,64))
    print(x,y)
    # print(x-cube.get_position(relative_to=agent_ee_tip)[0],y-cube.get_position(relative_to=agent_ee_tip)[1])
    go = generateTarget(x,y)
    path, direct_to_cube = get_path(go)
    while not done:
        if path:
            done = path.step()
        else:
            done = True
        pr.step()
        count+=1
        # print(agent_ee_tip.get_position())
    print(count)
    #need to add a check here to confirm or fail whether the arm reached the target or not.
    #We can check if the tip is within the the cube.
    if checkEEBoundary(agent_ee_tip,cube) and done and not direct_to_cube:
        correct+=1
        if count % 10 == 0:
            imq.save(f'../Resulting Images/Verticle Horizontal/true_img{tryy}.png')
            # plt.imsave(f'../Resulting Images/Verticle Horizontal, .0005, 30 ep, Eq AI, ActionEQ90, .0001 lr, L1 L2, 1 Epoch, Only Action Image, actionTry90/fail_img{tryy}.png',imq)
            # plt.imshow(res)
            plt.imsave(f'../Resulting Images/Verticle Horizontal/true_out{tryy}.png', res)
    else:
        # plt.imshow(img)
        imq.save(f'../Resulting Images/Verticle Horizontal/fail_img{tryy}.png')
        # plt.imsave(f'../Resulting Images/Verticle Horizontal, .0005, 30 ep, Eq AI, ActionEQ90, .0001 lr, L1 L2, 1 Epoch, Only Action Image, actionTry90/fail_img{tryy}.png',imq)
        # plt.imshow(res)
        plt.imsave(f'../Resulting Images/Verticle Horizontal/fail_out{tryy}.png',res)
    # go.remove()
    # pr.stop()


print("Total Correct: ", correct)
print("Percentage: ", correct/RUNS*100)


pr.stop()
pr.shutdown()








