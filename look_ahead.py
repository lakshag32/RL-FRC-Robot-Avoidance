from stable_baselines3 import SAC
from env import OnlyStaticEnv
import numpy as np
import math

env = OnlyStaticEnv([100,100],[500,200])
obs,info = env.reset()

terminated = False
truncated = False

episodes = 10000

# num_target_touches = 0
# num_obstacle_touches = 0
#model loading code: https://pythonprogramming.net/saving-and-loading-reinforcement-learning-stable-baselines-3-tutorial/
model_path = "(best)65_-25_-10.zip"
model = SAC.load(model_path, env=env)
action_sum = 0
num_look_ahead_steps = 15 #30
action_cntr = 1
robot_coord = env.robot

#training loop code from: https://pythonprogramming.net/introduction-reinforcement-learning-stable-baselines-3-tutorial/
while True:

    action, _states = model.predict(obs)
    action_cntr +=1
    action_sum +=action

    obs,_reward,_terminated,_truncated,_info = env.look_ahead_step(action)

    if action_cntr == num_look_ahead_steps:
        env.robot = robot_coord
        avg_action = action_sum/num_look_ahead_steps
        obs,_reward,_term,_trunc,_info = env.step(avg_action)    
        env.render()

        if math.dist(env.robot,env.target) < 30: 
            print("helo")
            break
            
        robot_coord = env.robot
        action_cntr = 0

        resized_avg_action = -np.float32(np.interp(avg_action,[-1,1],[-180,180]))
        print(resized_avg_action)