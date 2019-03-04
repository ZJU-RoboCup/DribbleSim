# -*- coding: utf-8 -*-

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

try:
    import vrep
except:
    print('ERROR: Cannot import vrep.py')

class DribbleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 1000
    }
    
    def __init__(self):
        self.dt = 0.001
        self.t = 0
        self.targetAngle = -1.517
        self.motor = None
        self.ball = None
        self.shelf = None
        self.threshold = 0.068
        
        self.low_action = np.array([-1.0])
        self.high_action = np.array([1.0])
        self.action_space = spaces.Box(low=self.low_action, high=self.high_action, dtype=np.float32)
        # robot observation space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        
        vrep.simxFinish(-1) # just in case, close all opened connections
        self.clientID=vrep.simxStart('127.0.0.1',19997,True,True,5000,5) # Connect to V-REP
        if self.clientID == -1:
            print ('ERROR: Cannot connected to remote API server')
            
        # enable the synchronous mode on the client:
        vrep.simxSynchronous(self.clientID,True)
        
        # start the simulation:
        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_blocking)
        [res,self.motor] = vrep.simxGetObjectHandle(self.clientID,'tubeJoint',vrep.simx_opmode_blocking)       
        [res,self.ball] = vrep.simxGetObjectHandle(self.clientID,'ball',vrep.simx_opmode_blocking)
        [res,self.shelf] = vrep.simxGetObjectHandle(self.clientID,'shelf',vrep.simx_opmode_blocking)
        #[res,robot] = vrep.simxGetObjectHandle(clientID,'robot',vrep.simx_opmode_blocking)
        vrep.simxGetObjectOrientation(self.clientID,self.shelf,-1,vrep.simx_opmode_streaming)
        vrep.simxSynchronousTrigger(self.clientID)
        vrep.simxSetObjectFloatParameter(self.clientID, self.ball,3001,100,vrep.simx_opmode_oneshot)
        
        vrep.simxSynchronousTrigger(self.clientID)
        
        self.done = False
        #reset
        vrep.simxGetObjectPosition(self.clientID, self.ball,-1,vrep.simx_opmode_streaming)
        
    def step(self, torque):
        torque = torque/20.0 + 0.05
        # apply torque
        vrep.simxSetJointForce(self.clientID, self.motor,torque,vrep.simx_opmode_oneshot)
        vrep.simxSynchronousTrigger(self.clientID)
        # get information
        [res, angle] = vrep.simxGetObjectOrientation(self.clientID, self.shelf,-1,vrep.simx_opmode_buffer)
        currentAngle = angle[2]
        error = currentAngle - self.targetAngle

        [res, dist] = vrep.simxGetObjectPosition(self.clientID, self.ball,-1,vrep.simx_opmode_buffer)
        if dist[1] > self.threshold:
            self.done = True
        else:
            self.done = False
        # reward = float(2*np.exp(-(torque[0]-0.03)*(torque[0]-0.03)/0.0005))
        # if torque > 0.02:
        #     reward -= float(20.0*(torque - 0.02))

        if self.done == True:
            reward -= 100.0
            
        if self.t > 0.5:
            self.done = True
        #update
        self.t = self.t + self.dt
        #print(self.t, torque)
        return [error, torque], reward, self.done, [torque, currentAngle, dist[1]]
        
    def reset(self):
        self.t = 0.0
        # stop last simulation
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_blocking)
        # start new simulation
        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_blocking)
        [res,self.motor] = vrep.simxGetObjectHandle(self.clientID,'tubeJoint',vrep.simx_opmode_blocking)       
        [res,self.ball] = vrep.simxGetObjectHandle(self.clientID,'ball',vrep.simx_opmode_blocking)
        [res,self.shelf] = vrep.simxGetObjectHandle(self.clientID,'shelf',vrep.simx_opmode_blocking)
        #[res,robot] = vrep.simxGetObjectHandle(clientID,'robot',vrep.simx_opmode_blocking)
        vrep.simxGetObjectOrientation(self.clientID,self.shelf,-1,vrep.simx_opmode_streaming)
        vrep.simxSynchronousTrigger(self.clientID)
        vrep.simxSetObjectFloatParameter(self.clientID, self.ball,3001,100,vrep.simx_opmode_oneshot)
        
        vrep.simxSynchronousTrigger(self.clientID)
        
        self.done = False
        #reset
        vrep.simxGetObjectPosition(self.clientID, self.ball,-1,vrep.simx_opmode_streaming)
        return [0, 0]
        
    def close(self):
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_blocking)
        vrep.simxFinish(self.clientID)
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]