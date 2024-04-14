import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random 
from stable_baselines3 import SAC
from env import OnlyStaticEnv
import numpy as np
import math
import ntables
import time
import util

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
def animate(i, ys):
    if math.dist(env.robot,env.target) < 20: 
        print("helo")
        return

    global obs

    action, _states = model.predict(obs)
        
    resized_action = -np.float32(np.interp(action,[-1,1],[-180,180]))[0]

    ntables.publish_drive_angle(-resized_action) #TODO: !!!!!! CHECK WHETHER OR NOT THIS ANGLE SHOULD BE POSITIVE OR NEGATIVE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    print(-resized_action)
    
    #robot moves...
    time.sleep(0.0001)
    
    # #update coords of robot:
    env.robot = ntables.get_robot_coord()
    
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

    # Read temperature (Celsius) from TMP102
    temp_c = resized_action

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