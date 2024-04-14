import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random 
from stable_baselines3 import SAC
from env import OnlyStaticEnv
import numpy as np
import math

env = OnlyStaticEnv([100,100],[800,500])
robot_coord = env.robot

obs,info = env.reset()

terminated = False
truncated = False

episodes = 10000
action_sum = 0 
action_cntr = 0
num_look_ahead_steps = 1
avg_action = 0 

# num_target_touches = 0
# num_obstacle_touches = 0
#model loading code: https://pythonprogramming.net/saving-and-loading-reinforcement-learning-stable-baselines-3-tutorial/
model_path = "(best)65_-25_-10.zip"
model = SAC.load(model_path, env=env)

# Parameters
x_len = 50       # Number of points to display
y_range = [-180, 180]  # Range of possible Y values to display

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
xs = list(range(0, x_len))
ys = [0] * x_len
ax.set_ylim(y_range)

# Create a blank line. We will update the line in animate
line, = ax.plot(xs, ys)

# Add labels
plt.title('TMP102 Temperature over Time')
plt.xlabel('Samples')
plt.ylabel('Temperature (deg C)')

# This function is called periodically from FuncAnimation
def animate(i, ys):
    global obs
    global action_cntr
    global action_sum
    global avg_action
    global robot_coord
    
    action, _states = model.predict(obs)
    action_cntr +=1
    action_sum +=action

    obs,_reward,_terminated,_truncated,_info = env.look_ahead_step(action)

    if action_cntr == num_look_ahead_steps:
        env.robot = robot_coord
        avg_action = action_sum/num_look_ahead_steps
        obs,_reward,_term,_trunc,_info = env.step(avg_action)    
        env.render()

        if math.dist(env.robot,env.target) < 20: 
            print("helo")
            return
            
        robot_coord = env.robot
        action_cntr = 0

        resized_avg_action = -np.float32(np.interp(avg_action,[-1,1],[-180,180]))
        print(resized_avg_action)


    # resized_avg_action = -np.float32(np.interp(avg_action,[-1,1],[-180,180]))
    # print(resized_avg_action)

    temp_c = 0

    # Add y to list
    ys.append(temp_c)

    # Limit y list to set number of items
    ys = ys[-x_len:]

    # Update line with new Y values
    line.set_ydata(ys)

    return line,

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(fig,
    animate,
    fargs=(ys,),
    interval=50,
    blit=True)

plt.show()