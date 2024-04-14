import random 
from stable_baselines3 import SAC
from coproc_only_static import OnlyStaticEnv
import numpy as np
import math
import ntables
import time
import util

env = OnlyStaticEnv([100,100],[250,100])
obs,info = env.reset()

terminated = False
truncated = False

episodes = 10000

# num_target_touches = 0
# num_obstacle_touches = 0
#model loading code: https://pythonprogramming.net/saving-and-loading-reinforcement-learning-stable-baselines-3-tutorial/
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

dynamic_obstacle2 = [10000,10000]#util.spawn()
dynamic_obstacle1 = [10000,10000]#util.spawn()
dynamic_obstacle3 = [10000,10000]#util.spawn()
dynamic_obstacle4 = [10000,10000] #util.spawn()
dynamic_obstacle5 = [10000,10000]#util.spawn()

static_obstacle1 = [10000,10000]#util.spawn()
static_obstacle2 = [10000,10000]#util.spawn()
static_obstacle3 = [10000,10000]#util.spawn()
static_obstacle4 = [10000,10000]#util.spawn()
static_obstacle5 = [10000,10000] #util.spawn()

# This function is called periodically from FuncAnimation
while True: 
    if math.dist(env.robot,env.target) < 20: 
        print("helo")
        break

    action, _states = model.predict(obs)
        
    resized_action = np.float32(np.interp(action,[-1,1],[-180,180]))[0]

    ntables.publish_drive_angle(resized_action) #TODO: !!!!!! CHECK WHETHER OR NOT THIS ANGLE SHOULD BE POSITIVE OR NEGATIVE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    print("resized_action: " + str(resized_action))
    #robot moves...
    time.sleep(1)
    
    # #update coords of robot:
    env.robot = ntables.get_robot_coord()
    print("robot_coord: " + str(env.robot))
    current_euc_target_dist = math.dist(env.robot,env.target) 

    obs = [env.robot[0]/(util.FIELD_WIDTH-util.ROBOT_SIZE),
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
                
    obs = np.array(obs,dtype=np.float32)

    # obs,reward,terminated,truncated,info = env.step(action)

    # env.render()

    # print(env.num_steps)
