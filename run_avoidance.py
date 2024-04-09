#angle graphing code from: https://github.com/WaveShapePlay/Arduino_RealTimePlot/blob/master/Part2_RealTimePlot_UsingClass/ArduinoRealTimePlot.py
from stable_baselines3 import SAC
from coproc_only_static import OnlyStaticEnv
import ntables
import numpy as np
import math
import util

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

#50,50 in pixels
env = OnlyStaticEnv([100,100],[100,200])
observation, info  = env.reset()

print(env.robot)
print(env.target)

terminated = False
truncated = False

episodes = 10000

# num_target_touches = 0
# num_obstacle_touches = 0
# #model loading code: https://pythonprogramming.net/saving-and-loading-reinforcement-learning-stable-baselines-3-tutorial/
model_path = "(best)65_-25_-10.zip"
model = SAC.load(model_path, env=env)

euc_dynamic_obstacle1 = 10000 #math.dist(self.robot,self.dynamic_obstacle1) 
euc_dynamic_obstacle2 = 10000 #math.dist(self.robot,self.dynamic_obstacle1) 
euc_dynamic_obstacle3 = 10000 #math.dist(self.robot,self.dynamic_obstacle1) 
euc_dynamic_obstacle4 = 10000#math.dist(self.robot,self.dynamic_obstacle1) 
euc_dynamic_obstacle5 = 10000#math.dist(self.robot,self.dynamic_obstacle1) 
euc_static_obstacle1 =  10000#math.dist(self.robot,self.static_obstacle1) 
euc_static_obstacle2 = 10000#math.dist(self.robot,self.static_obstacle2) 
euc_static_obstacle3 = 10000#math.dist(self.robot,self.static_obstacle3) 
euc_static_obstacle4 = 10000#math.dist(self.robot,self.static_obstacle4) 
euc_static_obstacle5 = 10000#math.dist(self.robot,self.static_obstacle5) 

while True:
    if math.dist(env.robot,env.target) < 20: 
        print("helo")

    # pass observation to model to get predicted action
    action, _states = model.predict(observation)

    #obs, rewards, terminated, truncated, info = env.step(action)

    #resize the action to -180 to 180 to drive robot to an angle
    resized_action = np.float32(np.interp(action,[-1,1],[-180,180]))

    # print(-resized_action[0])

    # #flip it because angles are inverted in simulation and real life
    ntables.publish_drive_angle(-resized_action[0]) #TODO: !!!!!! CHECK WHETHER OR NOT THIS ANGLE SHOULD BE POSITIVE OR NEGATIVE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    print(-resized_action[0])
    # #robot moves at 1m/s for 0.1 seconds...
    # #time.sleep(0.1)

    # #update coords of robot:
    env.robot = ntables.get_robot_coord()
    
    current_euc_target_dist = math.dist(env.robot,env.target) 
    # print(current_euc_target_dist)

    # #calculate distances for observation
    # current_euc_target_dist = math.dist(env.robot,env.target) 
    # euc_dynamic_obstacle1 = 10000 #math.dist(env.robot,env.dynamic_obstacle1) 
    # euc_dynamic_obstacle2 = 10000 #math.dist(env.robot,env.dynamic_obstacle1) 
    # euc_dynamic_obstacle3 = 10000 #math.dist(env.robot,env.dynamic_obstacle1) 
    # euc_dynamic_obstacle4 = 10000#math.dist(env.robot,env.dynamic_obstacle1) 
    # euc_dynamic_obstacle5 = 10000#math.dist(env.robot,env.dynamic_obstacle1) 
    # euc_static_obstacle1 =  10000#math.dist(env.robot,env.static_obstacle1) 
    # euc_static_obstacle2 = 10000#math.dist(env.robot,env.static_obstacle2) 
    # euc_static_obstacle3 = 10000#math.dist(env.robot,env.static_obstacle3) 
    # euc_static_obstacle4 = 10000#math.dist(env.robot,env.static_obstacle4) 
    # euc_static_obstacle5 = 10000#math.dist(env.robot,env.static_obstacle5) 

    observation = [env.robot[0]/(util.FIELD_WIDTH-util.ROBOT_SIZE),
                    env.robot[1]/(util.FIELD_HEIGHT-util.ROBOT_SIZE),
                    env.target[0]/(util.FIELD_WIDTH-util.ROBOT_SIZE),
                    env.target[1]/(util.FIELD_HEIGHT-util.ROBOT_SIZE),
                       
                    env.dynamic_obstacle1[0]/(util.FIELD_WIDTH-util.ROBOT_SIZE),
                    env.dynamic_obstacle1[1]/(util.FIELD_HEIGHT-util.ROBOT_SIZE),
                    env.dynamic_obstacle2[0]/(util.FIELD_WIDTH-util.ROBOT_SIZE),
                    env.dynamic_obstacle2[1]/(util.FIELD_HEIGHT-util.ROBOT_SIZE),                     
                    env.dynamic_obstacle3[0]/(util.FIELD_WIDTH-util.ROBOT_SIZE),            
                    env.dynamic_obstacle3[1]/(util.FIELD_HEIGHT-util.ROBOT_SIZE),                      
                    env.dynamic_obstacle4[0]/(util.FIELD_WIDTH-util.ROBOT_SIZE),                      
                    env.dynamic_obstacle4[1]/(util.FIELD_HEIGHT-util.ROBOT_SIZE),
                    env.dynamic_obstacle5[0]/(util.FIELD_WIDTH-util.ROBOT_SIZE),                      
                    env.dynamic_obstacle5[1]/(util.FIELD_HEIGHT-util.ROBOT_SIZE),
                    
                    current_euc_target_dist/(2*(util.FIELD_WIDTH-2*util.ROBOT_SIZE)), 
                    euc_dynamic_obstacle1/(2*(util.FIELD_WIDTH-2*util.ROBOT_SIZE)),
                    euc_dynamic_obstacle2/(2*(util.FIELD_WIDTH-2*util.ROBOT_SIZE)),
                    euc_dynamic_obstacle3/(2*(util.FIELD_WIDTH-2*util.ROBOT_SIZE)),
                    euc_dynamic_obstacle4/(2*(util.FIELD_WIDTH-2*util.ROBOT_SIZE)),
                    euc_dynamic_obstacle5/(2*(util.FIELD_WIDTH-2*util.ROBOT_SIZE)),
                    euc_static_obstacle1/(2*(util.FIELD_WIDTH-2*util.ROBOT_SIZE)),
                    euc_static_obstacle2/(2*(util.FIELD_WIDTH-2*util.ROBOT_SIZE)),
                    euc_static_obstacle3/(2*(util.FIELD_WIDTH-2*util.ROBOT_SIZE)),
                    euc_static_obstacle4/(2*(util.FIELD_WIDTH-2*util.ROBOT_SIZE)),
                    euc_static_obstacle5/(2*(util.FIELD_WIDTH-2*util.ROBOT_SIZE)),
                ]
                
    observation = np.array(observation,dtype=np.float32)