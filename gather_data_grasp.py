#based off of reinforcement learning env code

from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
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
from PIL import Image
from pyrep.objects.joint import JointMode
from pyrep.objects.proximity_sensor import ProximitySensor



#Setup
SCENE_FILE = join(dirname(abspath(__file__)), "simulations/scene_panda_reach_target.ttt")
EPISODE = 100 #number of total episodes to run
RUNS = 4 #number of total different approaches to take
EPISODE_LENGTH = 100 #number of total steps to reach the target




class Environment(object):

    def __init__(self):
        #launch pyrep
        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless = False)
        self.pr.start()
        #--Robot
        self.agent = Panda()
        self.gripper = PandaGripper()
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

        #--Cube
        self.cube = Shape.create(type=PrimitiveShape.CUBOID,
                      size=[0.05, 0.05, 0.05],
                      color=[1.0, 0.1, 0.1],
                      static=False, respondable=True)
        self.target = Dummy.create()

        #--Cube Spawn
        cube_size = .1
        self.table = Shape('diningTable_visible')
        cube_min_max = self.table.get_bounding_box()
        cube_min_max = [cube_min_max[0] + cube_size,
                        .6 - cube_size,
                        cube_min_max[2] + cube_size,
                        cube_min_max[3] - cube_size,
                        cube_min_max[5] - .05]
        self.position_min, self.position_max = [cube_min_max[0], cube_min_max[2], cube_min_max[3]-.05], [cube_min_max[1],
                                                                                           cube_min_max[3],
                                                                                           cube_min_max[3]-.05]
        self.target_min, self.target_max = [-.02, -.02, -.01], [.02, .02, .01]

        col_name = ["imLoc", "jVel", "jPos", "eeVel","eeJacVel", "eePos", "cPos","stop"]
        self.df = pd.DataFrame(columns=col_name)
        self.path=None
        self.path_step = None
    def setup(self):
        # self.pr.stop()
        # self.pr.start()

        #----general scene stuff

        # self.agent.set_model_dynamic(False)
        # self.agent.set_control_loop_enabled(False)
        # self.agent.set_motor_locked_at_zero_velocity(True)
        # self.agent_ee_tip = self.agent.get_tip()
        self.agent.set_joint_target_velocities(np.zeros_like(self.agent.get_joint_target_velocities()))
        # self.agent.set_joint_positions(self.initial_joint_positions)
        # self.agent.set_model_dynamic(True)
        # self.vs = VisionSensor("Vision_sensor")
        # self.vs.set_resolution([64, 64])
        # self.cube = Shape.create(type=PrimitiveShape.CUBOID,
        #               size=[0.05, 0.05, 0.05],
        #               color=[1.0, 0.1, 0.1],
        #               static=True, respondable=False)
        # self.target = Dummy.create()

        #----Dynamic and config tree reset
        self.agent.reset_dynamic_object()
        self.pr.set_configuration_tree(self.agent_state)

        #----Robot Pose Reset
        self.agent.set_joint_positions(self.initial_joint_positions,disable_dynamics=True)
        self.path=None
        self.path_step = None


    def replaceCube(self):
        pos = list(np.random.uniform(self.position_min, self.position_max))
        self.cube.set_position(pos, self.table)
        self.cube.set_orientation([0,0,0])
        ori = self.cube.get_orientation()
        try:
            pp = self.agent.get_linear_path(
                position=self.cube.get_position(),
                euler=[0, math.radians(180), math.radians(90)+ori[2]],
                steps=100
            )
        except ConfigurationPathError as e:
            print("Cube bad placement. Replacing.")
            self.replaceCube()
        self.replaceTarget()
    def replaceTarget(self):
        targpos = list(np.random.uniform(self.target_min, self.target_max))
        self.target.set_position(targpos, self.cube)
        ori = self.cube.get_orientation()
        try:
            self.path = self.agent.get_linear_path(
                position=self.target.get_position(),
                euler=[0, math.radians(180), math.radians(90)+ori[2]],
                steps=100
            )
        except ConfigurationPathError as e:
            print("Cube bad placement. Replacing.")
            self.replaceTarget()
    def gatherInfo(self,ep,r,s,stop=False):
        im = self.vs.capture_rgb()
        if not os.path.isdir(f"images/episode{ep}"):
            os.mkdir(f"images/episode{ep}")
        if not os.path.isdir(f"images/episode{ep}/run{r}"):
            os.mkdir(f"images/episode{ep}/run{r}")
        location = f"images/episode{ep}/run{r}/s{s}.jpg"
        im = Image.fromarray((im * 255).astype(np.uint8)).resize((64, 64)).convert('RGB')
        im.save(location)
        jacob = self.agent.get_jacobian()
        jacob = np.flip(jacob,axis=0)
        jacob = jacob.T
        # print(jacob)
        # jacob = np.flip(jacob,axis=0)
        joint_vel = ",".join(np.array(self.agent.get_joint_velocities()).astype(str))
        joint_pos = ",".join(np.array(self.agent.get_joint_positions()).astype(str))
        jVel = [float(item) for item in joint_vel.split(",")]
        ee_pos = ",".join(self.agent.get_tip().get_position(relative_to=self.agent).astype(str))
        ee_j_vel = ",".join(np.array(jacob@jVel).astype(str))
        ee_vel = ",".join(np.concatenate(list(self.agent.get_tip().get_velocity()), axis=0).astype(str))
        cube_pos = ",".join(self.cube.get_position(relative_to=self.agent).astype(str))
        #this is to try and teach the last frame to the neural network.
        stp = 0
        if stop:
            stp = 1
        # if stop:
        #     joint_vel = ",".join(np.zeros_like(np.array(self.agent.get_joint_velocities())).astype(str))
        #     ee_vel = ",".join(np.zeros_like(np.concatenate(list(self.agent.get_tip().get_velocity()), axis=0)).astype(str))
        line = [location, joint_vel, joint_pos, ee_vel,ee_j_vel, ee_pos, cube_pos,stp]
        df_length = len(self.df)
        self.df.loc[df_length] = line
    def get_path(self,cube=False):
        ori = self.cube.get_orientation()
        if not cube:
            self.path = self.agent.get_linear_path(
                position=self.target.get_position(), euler=[0, math.radians(180), math.radians(90)-ori[2]], steps=100)
            self.path_step = self.path._path_points
        else:
            self.path = self.agent.get_linear_path(
                position=self.cube.get_position(), euler=[0, math.radians(180), math.radians(90)-ori[2]], steps=100)
            self.path_step = self.path._path_points
        # print(self.path_step)

    def checkStop(self):
        dist = self.ps.read()
        # print(dist)
        done = False
        if dist <= .15 and dist > 0:
            print("reached")
            done = True
        return done
    def step(self,pathstep = True, gripstep=False):
        if pathstep:
            done = self.path.step()
        if gripstep:
            done = self.gripper.actuate(0,.1)
        self.pr.step()
        if pathstep or gripstep:
            return done

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()







env = Environment()
# for e in range(EPISODE):
#     env.setup()
#     env.replaceCube()
for r in range(EPISODE):
    env.setup()
    env.replaceCube()
    # env.replaceTarget()
    env.get_path()
    doneReach=False
    doneGrip = False
    sq=0
    while not doneReach:
        doneReach = env.step()
        stp = env.checkStop()
        # env.gatherInfo(e,r,sq,stop=stp)
        sq+=1
    if doneReach:
        print("Reached")
        while not doneGrip:
            doneGrip = env.step(pathstep=False,gripstep=True)
            sq+=1
    if doneReach and doneGrip:
        print("Gripped")
        for i in range(50):
            new = env.agent.get_joint_positions()
            new[0] += .1
            env.agent.set_joint_target_positions(new)
            env.step(pathstep=False)

        # if done:
        #
        #     for i in range(10):
        #         print(f"It's now supposed to stop. {r},{e}")
        #         env.step(False)
        #         env.gatherInfo(e,r,sq,stop=True)
        #         sq+=1
env.shutdown()



# env.df.to_csv("lol.csv")



