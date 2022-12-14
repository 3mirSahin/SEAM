#based off of reinforcement learning env code

'''this one is adjusted versus the initial one to always grab the cube depending on its orientation
as the gripper is likely to perform better if it grasps the object with flat sides rather than non-flat sides

Though, on a second thought, the approach to the cube will be the same since the approach is based on the gripper's orientation
versus the cube. Due to this, only turning the cube in -45 to 45 degrees makes the most sense as the rest is equivariant. We wouldn't want the gripper to turn several times for a similar grip
as the cube does not have a defined front or back.'''
from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
from pyrep.errors import ConfigurationPathError
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
import numpy as np
import math
import pandas as pd
import os
import PIL
from PIL import Image, ImageOps
from pyrep.objects.joint import JointMode
from pyrep.objects.proximity_sensor import ProximitySensor



#Setup
SCENE_FILE = join(dirname(abspath(__file__)), "../simulations/scene_panda_reach_action_image.ttt")
EPISODE = 150 #number of total episodes to run

EPISODE_LENGTH = 100 #number of total steps to reach the target
GRAD = True #If you want the gradient output using L1 - the further away the end point is, the lower the pixel value.
VerticalStop = False #Only trains on one vertical side
HorizontalStop = False #only trains on one horizontal side
RECTANGLE = True
ROTATIONS = True
BINSIZE = 16



class Environment(object):

    def __init__(self,rect = False):
        #launch pyrep
        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless = True)
        self.pr.start()
        #--Robot
        self.agent = Panda()
        # self.agent.set_control_loop_enabled(False)
        # self.agent.set_motor_locked_at_zero_velocity(True)
        self.agent_ee_tip = self.agent.get_tip()
        self.initial_joint_positions = self.agent.get_joint_positions()
        self.agent_state = self.agent.get_configuration_tree()

        #proximity sensor
        self.ps = ProximitySensor("Proximity_sensor")
        #--Vision Sensor
        self.vs = VisionSensor("Vision_sensor")
        self.vs.set_resolution([64, 64])
        self.top_right_dum = Dummy("top_right")
        self.bot_left_dum = Dummy("bot_left")
        self.top_right = self.top_right_dum.get_position()
        self.bot_left = self.bot_left_dum.get_position()
        cube_size = .025
        #--Cube
        if rect:
            cube_sizes = [cube_size,cube_size*3,cube_size]
        else:
            cube_sizes = [cube_size,cube_size,cube_size]
        self.cube = Shape.create(type=PrimitiveShape.CUBOID,
                                 size=cube_sizes,
                                 color=[1.0, 0.1, 0.1],
                                 static=True, respondable=False)

        #--Cube Spawn


        cube_min_max = [self.bot_left[0]-cube_size/2,
                        self.top_right[0]+cube_size/2,
                        self.bot_left[1]-cube_size/2,
                        self.top_right[1]+cube_size/2,
                        cube_size/2]
        self.position_min, self.position_max = [cube_min_max[0], cube_min_max[2], cube_min_max[4]], [cube_min_max[1],
                                                                                                     cube_min_max[3],
                                                                                                     cube_min_max[4]]
        if HorizontalStop:
            self.position_max[0] = self.position_max[0] / 2
        if VerticalStop:
            self.position_max[1] = self.position_max[1] / 2
        print(cube_min_max)

        self.orient_min, self.orient_max = [0,0,math.radians(-90)], [0,0,math.radians(90)] #make s
        col_name = ["imLoc", "outLoc","cropLoc","rotBin","cubePos","cubeRot"]
        self.df = pd.DataFrame(columns=col_name)
        self.path = None
        self.path_step = None
    def setup(self):
        # self.pr.stop()
        # self.pr.start()

        #----general scene stuff

        self.agent.set_joint_target_velocities(np.zeros_like(self.agent.get_joint_target_velocities()))

        #----Dynamic and config tree reset
        self.agent.reset_dynamic_object()
        self.pr.set_configuration_tree(self.agent_state)

        #----Robot Pose Reset
        self.agent.set_joint_positions(self.initial_joint_positions,disable_dynamics=True)
        self.path=None
        self.path_step = None
    def generateActionImage(self,item,sensor_size=(32,32),Grad = False):
        tR = self.top_right_dum.get_position(relative_to=self.agent_ee_tip) #lowest values, so need to subtract
        bL = self.bot_left_dum.get_position(relative_to=self.agent_ee_tip) #higher values, so need to add
        pos = item.get_position(relative_to=self.agent_ee_tip)

        # print(pos[0],bL[0],tR[0])
        x = bL[0] - tR[0]
        posX = pos[0] - tR[0]

        y = bL[1] - tR[1]
        posY = pos[1] - tR[1]
        # print(x,y)
        # print(posX,posY)
        x = int((posX/x) * sensor_size[0])
        y = int((posY/y) * sensor_size[1])

        def l1(a, b):
            return sum(abs(val1 - val2) for val1, val2 in zip(a, b))

        touched = set((x,y))
        img = Image.new('L',(sensor_size))
        pixels = img.load()
        pixels[y,x] = 1 * 255
        nav = [[0,1],[1,0],[-1,0],[0,-1]]
        nav2 = [[1,1],[-1,-1],[1,-1],[-1,1]]
        minVal = -l1([sensor_size[0],sensor_size[1]],[0,0])
        if Grad:
            for nX in range(64):
                for nY in range(64):
                    # print(nY,nX)
                    curr = [nX,nY]
                    real = [x,y]
                    pVal = -l1(curr,real)
                    pixels[nY,nX] = int(((pVal - minVal) / (0 - minVal)) * 100)
            #then normalize it
        pixels[y, x] = 1 * 255


        for nY, nX in nav:
            nY += y
            nX += x
            if 0<= nX < sensor_size[0] and 0<= nY < sensor_size[1] and (nX,nY) not in touched:
                pixels[nY,nX] = int(.75*255)
                touched.add((nX,nY))

        for nY, nX in nav2:
            nY += y
            nX += x
            if 0<= nX < sensor_size[0] and 0<= nY < sensor_size[1]:
                pixels[nY,nX] = int(.50*255)
        if RECTANGLE:
            for i in range(-2,3):
                if 0 <= (x+i) < sensor_size[1]:
                    pixels[y,x+i] = int(.80 * 255)
        #rotate based on ee rotation:
        img = ImageOps.mirror(img)
        rot = -int(math.degrees(self.vs.get_orientation()[2]))
        print(rot)
        # img = img.rotate(rot)




        return img,(x,y)
    def generateCroppedImg(self,img,coord):
        # img = ImageOps.flip(img)
        # img = ImageOps.mirror(img)
        #add padding
        pix = img.load()
        img = ImageOps.expand(img, border=10, fill=pix[63,63])

        #return cropped version
        cropLoc = (coord[1],coord[0],coord[1]+20,coord[0]+20)
        print(cropLoc)
        ret = img.crop(cropLoc)
        # ret = ImageOps.mirror(ret)
        # ret = ImageOps.flip(ret)
        return ret

    def generateRotationBin(self,rot, binSize=16):
        bins = np.zeros(binSize)
        # determining the rotation cutoff:
        binVal = math.radians(
            180) / binSize  # It doesn't matter if we rotate more than 180 degrees for the objects we are using

        bin = int((rot + math.radians(90)) / binVal) - 1
        if bin >= binSize:
            bin = bin % (binSize - 1)
        bins[bin] = 1
        return bins
    # def generateRotationBin(self,rot,binSize = 8):
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

    def pixelToCoord(self,img,sensor_size=(32,32)):
        #need to mirror image
        # img = ImageOps.mirror(img)
        img = img.rotate(math.degrees(self.agent_ee_tip.get_orientation()[2]))
        # img = ImageOps.flip(img)
        pixel = np.array(img)
        tR = self.top_right_dum.get_position(relative_to=self.agent_ee_tip)
        bL = self.bot_left_dum.get_position(relative_to=self.agent_ee_tip)
        pLoc = np.unravel_index(pixel.argmax(), pixel.shape)


        x = bL[0] - tR[0]
        y = bL[1] - tR[1]

        x = pLoc[0] /sensor_size * x
        y = pLoc[1]/sensor_size * y

        return x,y
    def pixelToCrop(self,img):

        # img = img.rotate(-math.degrees(self.agent_ee_tip.get_orientation()[2]))
        pixel = np.array(img)
        pLoc = np.unravel_index(pixel.argmax(), pixel.shape)

        return pLoc
    def replaceCube(self,onlyOr=False):
        pos = list(np.random.uniform(self.position_min, self.position_max))
        rot = list(np.random.uniform(self.orient_min,self.orient_max))
        self.cube.set_position(pos)
        if ROTATIONS:
            self.cube.set_orientation(rot) #is table really the correct thing to base the orientation off of?
        try:
            pp = self.agent.get_linear_path(
                position=self.cube.get_position(),
                euler=[0, math.radians(180), 0],
                steps=100
            )
        except ConfigurationPathError as e:
            print("Cube bad placement. Replacing.")
            self.replaceCube()

    def get_path(self):
        ori = self.cube.get_orientation()
        try:
            self.path = self.agent.get_linear_path(
                position=self.cube.get_position(), euler=[0, math.radians(180), math.radians(90)-ori[2]])
        except ConfigurationPathError as e:
            self.path = self.agent.get_linear_path(
                position=self.cube.get_position(),
                euler=[0, math.radians(180), 0],
                steps=100
            )
        self.path_step = self.path._path_points
    def gatherInfo(self,ep):
        im = self.vs.capture_rgb()
        if not os.path.isdir(f"../action_image_rotation/action_images_in"):
            os.mkdir(f"../action_image_rotation/action_images_in")
        location_in = f"../action_image_rotation/action_images_in/episode{ep}.png"
        im = Image.fromarray((im * 255).astype(np.uint8)).resize((64, 64)).convert('RGB')
        im.save(location_in)

        #now, we need to generate the x/y coordinates - this is based on the cube's location within the given image.
        cube_pos = ",".join(self.cube.get_position(relative_to=self.agent).astype(str))
        cube_rot = ",".join(self.cube.get_orientation(relative_to=self.agent).astype(str))
        bin_rot = self.generateRotationBin(self.cube.get_orientation(relative_to=self.agent_ee_tip)[2],binSize=BINSIZE)
        bin_rot = ",".join(bin_rot.astype(str))
        act_img, cropLoc = self.generateActionImage(self.cube,(64,64),GRAD)
        cropLoc = self.pixelToCrop(act_img)
        print(cropLoc)
        crop_img = self.generateCroppedImg(im,cropLoc)

        if not os.path.isdir(f"../action_image_rotation/action_images_out"):
            os.mkdir(f"../action_image_rotation/action_images_out")

        location_out = f"../action_image_rotation/action_images_out/episode{ep}.png"
        act_img.save(location_out)

        if not os.path.isdir(f"../action_image_rotation/action_images_cropped"):
            os.mkdir(f"../action_image_rotation/action_images_cropped")

        location_crop = f"../action_image_rotation/action_images_cropped/episode{ep}.png"
        crop_img.save(location_crop)


        line = [location_in, location_out,location_crop,bin_rot, cube_pos,cube_rot]
        df_length = len(self.df)
        self.df.loc[df_length] = line

    def step(self,pathstep = True):
        if pathstep:
            done = self.path.step()
        self.pr.step()
        if pathstep:
            return done
    def checkStop(self): #currently stops a bit too early. Adjust accordingly.
        dist = self.ps.read()
        # print(dist)
        done = False
        if dist <= .11 and dist > 0:
            print("reached", dist)
            done = True
        return done
    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()







env = Environment(rect=RECTANGLE)
for e in range(EPISODE):
    print(f"---EP {e}---")
    env.setup()
    env.replaceCube()
    env.get_path()
    done=False
    sq=0
    env.step(pathstep=False)
    env.gatherInfo(e)
    while not done:
        done = env.step()
        stp = env.checkStop()
        sq+=1

env.shutdown()



env.df.to_csv("../sequences/action_image_rotation.csv")



