#code adapted from: https://github.com/plemaster01/LeMasterTechYT/blob/main/snake.py
#environment made using template: https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
#environment made with inspirtaion from: https://github.com/LetTheCodePlay/OpenAIGym/blob/main/Pygame/Snake/snakeenv.py

import gymnasium as gym
import numpy as np
#import pygame
from gymnasium import spaces
import random
import cv2
import math
import util

class OnlyStaticEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self,robot_center,target_center): 
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=-1,high=1,shape=(1,),dtype=np.float32)
        # Example for using image as input (channel-first; channel-last also works):
        
        #observation space adapted from: https://www.youtube.com/watch?v=bD6V3rcr_54&t=659s
        #https://www.youtube.com/watch?v=POyBab-M6tM&t=1545s
        self.observation_space = spaces.Box(low=-60, high=1000,
                                            shape=(25,), dtype=np.float32)
        
        self.robot = robot_center
        self.target = target_center

        self.dynamic_obstacle1 = [10000,10000]#util.spawn()
        self.dynamic_obstacle2 = [10000,10000]#util.spawn()
        self.dynamic_obstacle3 = [10000,10000]#util.spawn()
        self.dynamic_obstacle4 = [10000,10000] #util.spawn()
        self.dynamic_obstacle5 = [10000,10000]#util.spawn()

        self.static_obstacle1 = [10000,1000]#util.spawn()
        self.static_obstacle2 = [10000,10000]#util.spawn()
        self.static_obstacle3 = [10000,10000]#util.spawn()
        self.static_obstacle4 = [10000,10000]#util.spawn()
        self.static_obstacle5 = [10000,10000] #util.spawn()
        self.obstacles = [self.dynamic_obstacle1, 
                          self.dynamic_obstacle2,
                          self.dynamic_obstacle3,
                          self.dynamic_obstacle4,
                          self.dynamic_obstacle5,
                          self.static_obstacle1,
                          self.static_obstacle2,
                          self.static_obstacle3,
                          self.static_obstacle4,
                          self.static_obstacle5
                        ]
        
        util.fix_stacked(self.robot, self.target, self.obstacles)

        self.reward_d = 0
        self.reward_b = 0
        self.reward_f = 0
        self.reward_c = 0

        self.prev_euc_target_dist = 0 
        
        self.num_steps = 0
        self.max_steps = 65 #100

        self.counter = 0

        self.obstacle_angle1 = 0
        self.obstacle_angle2 = 0
        self.obstacle_angle3 = 0
        self.obstacle_angle4 = 0
        self.obstacle_angle5 = 0

        self.prev_angle = 0; 

    def step(self, angle):
        print(self.robot)
        print(self.target)

        new_angle = np.float32(np.interp(angle,[-1,1],[-180,180]))
        terminated = False
        truncated = False

        self.screen = np.zeros((util.FIELD_HEIGHT,util.FIELD_WIDTH,3), np.uint8)

        # self.obstacles[0] = self.dynamic_obstacle1
        # self.obstacles[1] = self.dynamic_obstacle2
        # self.obstacles[2] = self.dynamic_obstacle3
        #self.obstacles[3] = self.dynamic_obstacle4
        #self.obstacles[4] = self.dynamic_obstacle5
        
        self.counter +=1
        self.num_steps +=1

        util.move(self.robot,new_angle)

        if self.counter == 10: 
            self.obstacle_angle1 = random.randint(-180,180)
            self.obstacle_angle2 = random.randint(-180,180)
            self.obstacle_angle3 = random.randint(-180,180)
            #self.obstacle_angle4 = random.randint(-180,180)
            #self.obstacle_angle5 = random.randint(-180,180)

            self.counter = 0

        # util.move(self.dynamic_obstacle1, self.obstacle_angle1)
        # util.move(self.dynamic_obstacle2, self.obstacle_angle2)
        # util.move(self.dynamic_obstacle3, self.obstacle_angle3)
        #util.move(self.dynamic_obstacle4, self.obstacle_angle4)
        #util.move(self.dynamic_obstacle5, self.obstacle_angle5)

        # util.rectify_obstacle_coords(self.dynamic_obstacle1)
        # util.rectify_obstacle_coords(self.dynamic_obstacle2)
        # util.rectify_obstacle_coords(self.dynamic_obstacle3)
        #util.rectify_obstacle_coords(self.dynamic_obstacle4)
        #util.rectify_obstacle_coords(self.dynamic_obstacle5)

        # for obstacle_center in self.obstacles: 
        #     util.draw_obstacle(self.screen,obstacle_center)
    
        robot_pts = util.get_unrotated_pts(self.robot)
        rotated_robot_pts = util.rotate_robot_pts_for_drawing(robot_pts,angle)
        util.draw_robot(self.screen, rotated_robot_pts)
        util.draw_target(self.screen, self.target)

        current_euc_target_dist = math.dist(self.robot,self.target) 
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
        # x_offset_to_dynamic_obstacle1 = 
        # x_offset_to_dynamic_obstacle2 = 
        # x_offset_to_dynamic_obstacle3 = 

        #check if there is a collision with wall or target, check other stuff, assign rewards 
        if util.check_robot_wall_collision(self.robot):
            terminated = True
            truncated = False
            reward_a = -100
        else: 
            reward_a = 0

        if util.check_robot_obstacle_collision(self.robot,self.obstacles): 
            #terminated = True
            #truncated = False
            self.reward_b = -100 
        else: 
            self.reward_b = 0

        if (self.num_steps >= self.max_steps):
            truncated = True 
            terminated = False
            reward_c = -100 
            #print(f"I took too many steps: {reward_c}")
        else:
            reward_c = 0

        if(current_euc_target_dist<self.prev_euc_target_dist): 
            self.reward_d = 45 #45
            #print(f"I got closer: {self.reward_d}")

        if(current_euc_target_dist>self.prev_euc_target_dist):
            self.reward_d = -25 #-25
            #print(f"I got farther: {self.reward_d}")
        
        if(current_euc_target_dist <=20): 
            terminated = True
            truncated = False
            reward_e = 100
            #print(f"YAY: {reward_e}")
        else:
            reward_e = 0
        
        if(angle == self.prev_angle): 
            reward_f = 25
        else:
            reward_f = -10

        self.prev_angle = angle
        self.prev_euc_target_dist = current_euc_target_dist

        reward = reward_a + reward_c + self.reward_d + reward_e + reward_f - 10 #-1 #+ self.reward_b

        observation = [self.robot[0]/(util.FIELD_WIDTH-util.ROBOT_SIZE),
                       self.robot[1]/(util.FIELD_HEIGHT-util.ROBOT_SIZE),
                       self.target[0]/(util.FIELD_WIDTH-util.ROBOT_SIZE),
                       self.target[1]/(util.FIELD_HEIGHT-util.ROBOT_SIZE),
                       
                       self.dynamic_obstacle1[0]/(util.FIELD_WIDTH-util.ROBOT_SIZE),
                       self.dynamic_obstacle1[1]/(util.FIELD_HEIGHT-util.ROBOT_SIZE),
                       self.dynamic_obstacle2[0]/(util.FIELD_WIDTH-util.ROBOT_SIZE),
                       self.dynamic_obstacle2[1]/(util.FIELD_HEIGHT-util.ROBOT_SIZE),                     
                       self.dynamic_obstacle3[0]/(util.FIELD_WIDTH-util.ROBOT_SIZE),            
                       self.dynamic_obstacle3[1]/(util.FIELD_HEIGHT-util.ROBOT_SIZE),                      
                       self.dynamic_obstacle4[0]/(util.FIELD_WIDTH-util.ROBOT_SIZE),                      
                       self.dynamic_obstacle4[1]/(util.FIELD_HEIGHT-util.ROBOT_SIZE),
                       self.dynamic_obstacle5[0]/(util.FIELD_WIDTH-util.ROBOT_SIZE),                      
                       self.dynamic_obstacle5[1]/(util.FIELD_HEIGHT-util.ROBOT_SIZE),
                       
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
        
        info = {}

        if(terminated or truncated): 
            self.reset()

        #print(observation.shape)
        return observation, reward, terminated, truncated, info

    def reset(self,seed=None, options=None):
        # self.robot = self.robot
        # self.target = util.spawn()

        self.dynamic_obstacle1 = [10000,10000]#util.spawn()
        self.dynamic_obstacle2 = [10000,10000]#util.spawn()
        self.dynamic_obstacle3 = [10000,10000]#util.spawn()
        self.dynamic_obstacle4 = [10000,10000] #util.spawn()
        self.dynamic_obstacle5 = [10000,10000]#util.spawn()

        self.static_obstacle1 = [10000,10000]#util.spawn()
        self.static_obstacle2 = [10000,10000]#util.spawn()
        self.static_obstacle3 = [10000,10000]#util.spawn()
        self.static_obstacle4 = [10000,10000]#util.spawn()
        self.static_obstacle5 = [10000,10000] #util.spawn()
        self.obstacles = [self.dynamic_obstacle1, 
                          self.dynamic_obstacle2,
                          self.dynamic_obstacle3,
                          self.dynamic_obstacle4,
                          self.dynamic_obstacle5,
                          self.static_obstacle1,
                          self.static_obstacle2,
                          self.static_obstacle3,
                          self.static_obstacle4,
                          self.static_obstacle5
                        ]
        
        util.fix_stacked(self.robot, self.target, self.obstacles)

        current_euc_target_dist = math.dist(self.robot,self.target) 
        euc_dynamic_obstacle1 = 10000 #math.dist(self.robot,self.dynamic_obstacle1) 
        euc_dynamic_obstacle2 = 10000 #math.dist(self.robot,self.dynamic_obstacle1) 
        euc_dynamic_obstacle3 = 10000 #math.dist(self.robot,self.dynamic_obstacle1) 
        euc_dynamic_obstacle4 = 10000#math.dist(self.robot,self.dynamic_obstacle1) 
        euc_dynamic_obstacle5 = 10000#math.dist(self.robot,self.dynamic_obstacle1) 
        euc_static_obstacle1 = 10000#math.dist(self.robot,self.static_obstacle1) 
        euc_static_obstacle2 = 10000#math.dist(self.robot,self.static_obstacle2) 
        euc_static_obstacle3 = 10000#math.dist(self.robot,self.static_obstacle3) 
        euc_static_obstacle4 = 10000#math.dist(self.robot,self.static_obstacle4) 
        euc_static_obstacle5 = 10000#math.dist(self.robot,self.static_obstacle5) 

        observation = [self.robot[0]/(util.FIELD_WIDTH-util.ROBOT_SIZE),
                       self.robot[1]/(util.FIELD_HEIGHT-util.ROBOT_SIZE),
                       self.target[0]/(util.FIELD_WIDTH-util.ROBOT_SIZE),
                       self.target[1]/(util.FIELD_HEIGHT-util.ROBOT_SIZE),
                       
                       self.dynamic_obstacle1[0]/(util.FIELD_WIDTH-util.ROBOT_SIZE),
                       self.dynamic_obstacle1[1]/(util.FIELD_HEIGHT-util.ROBOT_SIZE),
                       self.dynamic_obstacle2[0]/(util.FIELD_WIDTH-util.ROBOT_SIZE),
                       self.dynamic_obstacle2[1]/(util.FIELD_HEIGHT-util.ROBOT_SIZE),                     
                       self.dynamic_obstacle3[0]/(util.FIELD_WIDTH-util.ROBOT_SIZE),            
                       self.dynamic_obstacle3[1]/(util.FIELD_HEIGHT-util.ROBOT_SIZE),                      
                       self.dynamic_obstacle4[0]/(util.FIELD_WIDTH-util.ROBOT_SIZE),                      
                       self.dynamic_obstacle4[1]/(util.FIELD_HEIGHT-util.ROBOT_SIZE),
                       self.dynamic_obstacle5[0]/(util.FIELD_WIDTH-util.ROBOT_SIZE),                      
                       self.dynamic_obstacle5[1]/(util.FIELD_HEIGHT-util.ROBOT_SIZE),
                       
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
        
        info = {}

        self.num_steps = 0 

        return observation,info


    def render(self):
        cv2.imshow("env",self.screen)
        cv2.waitKey(200)
        pass

    def close(self):
        cv2.destroyAllWindows()
        pass
