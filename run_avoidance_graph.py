import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from stable_baselines3 import SAC
from coproc_only_static import OnlyStaticEnv
import numpy as np
import math

env = OnlyStaticEnv([100,100],[200,100])
obs,info = env.reset()

terminated = False
truncated = False

episodes = 10000

# num_target_touches = 0
# num_obstacle_touches = 0
#model loading code: https://pythonprogramming.net/saving-and-loading-reinforcement-learning-stable-baselines-3-tutorial/
model_path = "(best)65_-25_-10.zip"
model = SAC.load(model_path, env=env)

# Parameters
x_len = 200         # Number of points to display
y_range = [10, 40]  # Range of possible Y values to display

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
xs = list(range(0, 200))
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



    # action, _states = model.predict(obs)
    
    # resized_action = -np.float32(np.interp(action,[-1,1],[-180,180]))[0]

    # obs,reward,terminated,truncated,info = env.step(action)

    # print(resized_action)

    # env.render()	# pass observation to model to get predicted action

    # # Read temperature (Celsius) from TMP102
    temp_c = int(3)

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