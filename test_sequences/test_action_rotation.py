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


from model_outlines.action_models import UNet, EqUNet, EqUNetFloor, rotCNN, rotEqCNN

'''Configurations for the test instance.'''
RUNS = 50 #the total number of test attempts done. Changes the cube location.
TURN = False
USEROT = False
if TURN:
    SCENE_FILE = join(dirname(abspath(__file__)), '../simulations/scene_panda_reach_action_image_turned.ttt')
else:
    SCENE_FILE = join(dirname(abspath(__file__)), '../simulations/scene_panda_reach_action_image.ttt')
TEST_ORIENT = False
TEST_90 = False
EQ = True
C8 = False
Floor = False
Padding = False
RECT = True
DISTRACT = False
# FlipObject = True
BINSIZE = 16


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
    rotModel = rotEqCNN(3,int(BINSIZE/2),int(BINSIZE/2))
elif EQ and C8:
    model = EqUNet(3,1,N=8,flip=False)
    rotModel = rotEqCNN(3,int(BINSIZE/2),int(BINSIZE/2))
elif EQ:
    model = EqUNet(3,1,N=4,flip=True)
    rotModel = rotEqCNN(3,int(BINSIZE/2),int(BINSIZE/2))
else:
    model = UNet(3,1,bilinear=True)
    rotModel = rotEqCNN(3,int(BINSIZE/2),int(BINSIZE/2))
# model = UNet(3,1,bilinear=True)
model.train()
rotModel.train()
model.load_state_dict(torch.load("../trained_models/actionEq90Try.pt")) #change this based on model
rotModel.load_state_dict(torch.load("../trained_models/rotation_models/actionEq90Try.pt"))
model.eval()
rotModel.eval()
#[3.1394 0.0221 1.5708]

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

dist_items = []



'''Cube Movement'''

cube_min_max = [bot_left[0] - cube_size / 2,
                top_right[0] + cube_size / 2,
                bot_left[1] - cube_size / 2,
                top_right[1] + cube_size / 2,
                cube_size / 2]

position_min, position_max = [cube_min_max[0], cube_min_max[2], cube_min_max[4]], [cube_min_max[1],
                                                                                           cube_min_max[3],
                                                                                           cube_min_max[4]]
def resetEnv(dist_items):
    agent.set_joint_target_velocities(np.zeros_like(agent.get_joint_target_velocities()))
    agent.set_motor_locked_at_zero_velocity(True)

    agent.reset_dynamic_object()
    pr.set_configuration_tree(agent_state)

    agent.set_joint_positions(initial_joint_position,disable_dynamics=True)

        #Adding distractors
    if DISTRACT:
        if dist_items:
            for shape in dist_items:
                shape.remove()
        size_limits = [.01, .01, .01, .07, .07, .07]  # first three is the min and the last 3 is the max
        dist_sphere = Shape.create(type=PrimitiveShape.SPHERE,
                                        size=list(np.random.uniform(size_limits[:3], size_limits[3:])),
                                        color=[.3, .5, .7],
                                        static=True,
                                        respondable=False)
        dist_cone = Shape.create(type=PrimitiveShape.CONE,
                                      size=list(np.random.uniform(size_limits[:3], size_limits[3:])),
                                      color=[.7, .3, .5],
                                      static=True,
                                      respondable=False)
        dist_cylinder = Shape.create(type=PrimitiveShape.CYLINDER,
                                          size=list(np.random.uniform(size_limits[:3], size_limits[3:])),
                                          color=[.3, .8, .2],
                                          static=True,
                                          respondable=False)
        dist_items = [dist_sphere, dist_cone, dist_cylinder]


        for shape in dist_items:
            pos = list(np.random.uniform(position_min, position_max))
            shape.set_position(pos)
            shape.set_orientation(list(np.random.uniform(orient_min, orient_max)))
        return dist_items
    return []
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
def get_path(object,rot=None,binSize = 16):
    if rot is None:
        ori = cube.get_orientation()[2]
    else:
        cubeOri = cube.get_orientation(relative_to=agent_ee_tip)[2]
        # print(math.degrees(cubeOri))
        cubeOri = np.argmax(generateRotationBin(cubeOri))

        ori = getRotation(rot,binSize)

    try:
        path = agent.get_linear_path(
            position=object.get_position(), euler=[0, math.radians(180), math.radians(90)+agent_ee_tip.get_orientation()[2]-ori])
        if rot:
            return path, False, (np.argmax(rot) in range(cubeOri-1,cubeOri+2))
        else:
            return path, False, True
    except ConfigurationPathError as e:
        path = agent.get_path(
            position=cube.get_position(),
            euler=[0, math.radians(180), 0]
        )
    # print(path._path_points)
    return path, True, False


def generateRotationBin(rot, binSize=16):
    bins = np.zeros(binSize)
    # determining the rotation cutoff:
    binVal = math.radians(
        180) / binSize  # It doesn't matter if we rotate more than 180 degrees for the objects we are using

    bin = int((rot + math.radians(90)) / binVal) -1
    if bin >= binSize:
        bin = bin % (binSize-1)
    bins[bin] = 1
    return bins

# def getRotationFromBin(rot,binSize = 8):
#     bins = np.zeros(binSize*2)
#     #The idea here is utilizing the cos and sin components seperately
#     cosVal = math.cos(2*rot)#doubling the value so we hopefully avoid 0 values
#     sinVal = math.cos(2*rot)
#     #then, bin each between 0 and 1
#     binVal = 2/binSize
#     cosBin = int((cosVal + 1)/binVal)
#     sinBin = int((sinVal + 1)/binVal)
#     bins[cosBin] = 1
#     bins[sinBin+8] = 1
#     return bins

def pixelToCoord(img,sensor_size=(32,32)):
    #need to mirror image
    img = ImageOps.mirror(img)
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

    return x,y, pLoc
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


def generateCroppedImg(img, coord):
    # img = ImageOps.flip(img)
    img = ImageOps.mirror(img)
    # add padding
    pix = img.load()
    img = ImageOps.expand(img, border=10, fill=pix[63, 63])

    # return cropped version
    cropLoc = (coord[1], coord[0], coord[1] + 20, coord[0] + 20)
    print(cropLoc)
    ret = img.crop(cropLoc)
    ret = ImageOps.mirror(ret)
    # ret = ImageOps.flip(ret)
    return ret

def getRotation(bin,binSize = 16):
    #determining the rotation cutoff:
    binVal = math.radians(180)/binSize #It doesn't matter if we rotate more than 180 degrees for the objects we are using
    maxBin = np.argmax(bin)
    maxBin -= int(binSize/2)
    out = binVal * (maxBin) + math.radians(-75)
    return out


if not os.path.isdir(f"../Resulting Images/Verticle Horizontal"):
    os.mkdir(f"../Resulting Images/Verticle Horizontal")
files = glob.glob('../Resulting Images/Verticle Horizontal/*')
for f in files:
    os.remove(f)
correct = 0
correctBin = 0
for tryy in range(RUNS):
    dist_items = resetEnv(dist_items)
    replaceCube()

    count = 0
    done = False
    pr.step()

    stops = []
    img = vs.capture_rgb()
    img = Image.fromarray((img * 255).astype(np.uint8)).resize((64, 64)).convert('RGB')
    #rotate the image according to the rotation of the end-effector
    # print("Rotation: ",agent_ee_tip.get_orientation())
    rot = math.ceil(math.degrees(agent_ee_tip.get_orientation()[2])) - 90
    if TURN:
        img = ImageOps.flip(img)
    # print(rot)
    img = img.rotate(0)
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
    x,y,pLoc = pixelToCoord(res,(64,64))
    print(x,y)
    #now getting the rotation from the rotationalModel
    go = generateTarget(x,y)
    cropped = generateCroppedImg(imq,pLoc)
    cropped = transform(cropped)
    cropped = cropped.unsqueeze(0)
    rot = rotModel(cropped).detach().numpy()


    #get the path
    if USEROT:
        path, direct_to_cube, binMatch = get_path(go,rot=rot,binSize=BINSIZE)
    else:
        path, direct_to_cube, binMatch = get_path(go, rot=None, binSize=BINSIZE)
    if direct_to_cube:
        continue
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
        if binMatch:
            correctBin+=1
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
print("Total Correct Rotation: ", correctBin)
print("Percentage: ", correctBin/RUNS*100)


pr.stop()
pr.shutdown()








