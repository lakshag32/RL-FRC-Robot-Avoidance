#angle plotting code: https://www.digikey.com/en/maker/projects/graph-sensor-data-with-python-and-matplotlib/93f919fd4e134c48bc5cd2bc0e5a5ba2#:~:text=To%20create%20a%20real%2Dtime,new%20frame%20in%20the%20animation.

import cv2
import numpy as np 
import random
import math

import matplotlib.pyplot as plt
import matplotlib.animation as animation

FIELD_WIDTH = 600
FIELD_HEIGHT = 300
ROBOT_SIZE = 46
MOVE_AMOUNT = 12
DANGER_ZONE = int(math.hypot(ROBOT_SIZE,ROBOT_SIZE))+10

# Parameters
x_len = 200         # Number of points to display
y_range = [-200, 200]  # Range of possible Y values to display

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

angle = 0

# This function is called periodically from FuncAnimation
def animate(i, ys):

    # Read temperature (Celsius) from TMP102
    temp_c = get_resized_angle()

    # Add y to list
    ys.append(temp_c)

    # Limit y list to set number of items
    ys = ys[-x_len:]

    # Update line with new Y values
    line.set_ydata(ys)

    return line,

def get_resized_angle():
    return angle 

def set_resized_angle(): 
    return angle 


def get_unrotated_pts(center):
    pt_2 = [int(center[0] + (ROBOT_SIZE)/2), int(center[1] - (ROBOT_SIZE)/2)]
    pt_4 = [int(center[0] - (ROBOT_SIZE)/2), int(center[1] + (ROBOT_SIZE)/2)]
    pt_1 = [pt_2[0]-ROBOT_SIZE,pt_2[1]]
    pt_3 = [pt_4[0]+ROBOT_SIZE,pt_4[1]]

    pts = np.array([pt_1,pt_2,pt_3,pt_4],dtype=np.int32)
    return pts

def draw_obstacle(screen, obstacle):
    pt_high = (int(obstacle[0] + ROBOT_SIZE/2), int(obstacle[1] - ROBOT_SIZE/2))
    pt_low = (int(obstacle[0] - ROBOT_SIZE/2), int(obstacle[1] + ROBOT_SIZE/2))
    
    #source: opencv docs
    cv2.rectangle(screen, pt_high, pt_low, (255,0,0), -1)  

def rotate_robot_pts_for_drawing(robot_pts,angle):
    #https://stackoverflow.com/questions/69556993/i-try-to-rotate-rectangle-with-function-but-the-boxes-not-in-correct-position
    ANGLE = np.deg2rad(angle)
    SIN = math.sin(ANGLE)
    COS = math.cos(ANGLE)
    
    c_x, c_y = np.mean(robot_pts, axis=0)

    return np.array(
        [
            [
                c_x + COS * (px - c_x) - SIN * (py - c_y),
                c_y + SIN * (px - c_x) + COS * (py - c_y),
            ]
            for px, py in robot_pts
        ]
    ).astype(int)
 
def draw_robot(screen,robot_pts): 
    #https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
    robot_pts = robot_pts.reshape((-1,1,2))
    cv2.polylines(screen,[robot_pts],True,(0,0,255)) 

    #https://www.geeksforgeeks.org/draw-a-filled-polygon-using-the-opencv-function-fillpoly/
    cv2.fillPoly(screen, pts=[robot_pts], color=(0, 0, 255))

def draw_target(screen,target): 
    cv2.circle(screen,target,10,(0,255,0),-1)

def move(center,angle): 
    angle_rad = np.deg2rad(angle)
    move_x = int(MOVE_AMOUNT*math.cos(angle_rad))
    move_y = int(MOVE_AMOUNT*math.sin(angle_rad))

    center[0] += move_x
    center[1] += move_y

def check_obstacle_wall_collisions(center):
    if(center[0] >= 400): 
        return True
    if(center[1] >= 400): 
        return True 
    if(center[0] <= 0): 
        return True
    if(center[1] <=0): 
        return True
    
    return False

def rectify_obstacle_coords(center): 
    if(center[0] >= FIELD_WIDTH): 
        center[0] = FIELD_WIDTH-ROBOT_SIZE/2 -2
    if(center[1] >= FIELD_HEIGHT): 
        center[1] = FIELD_HEIGHT-ROBOT_SIZE/2 -2
    if(center[0] <= 0): 
        center[0] = ROBOT_SIZE/2 +2
    if(center[1] <=0): 
        center[1] = ROBOT_SIZE/2 +2

def spawn():
    return [random.randint(20,FIELD_WIDTH-ROBOT_SIZE),random.randint(20,FIELD_HEIGHT-ROBOT_SIZE)]

def fix_stacked(robot_center,target_center,obstacles):
    for obstacle_center in obstacles: 
        if math.dist(target_center,obstacle_center) <= DANGER_ZONE: 
            target_center = spawn()
            obstacle_center = spawn()
        
        if(math.dist(robot_center,target_center) <= DANGER_ZONE): 
            robot_center = spawn()
            target_center = spawn()

        if(math.dist(robot_center,obstacle_center) <= DANGER_ZONE):
            robot_center = spawn()
            obstacle_center = spawn()

def check_robot_wall_collision(robot_center):
    if(robot_center[0] >= FIELD_WIDTH): 
        return True
    if(robot_center[1] >= FIELD_HEIGHT): 
        return True 
    if(robot_center[0] <= 0): 
        return True
    if(robot_center[1] <=0): 
        return True 

def check_robot_obstacle_collision(robot_center, obstacles): 
    for obstacle_center in obstacles:
        if math.dist(robot_center,obstacle_center) <= DANGER_ZONE:
            return True

    return False   

def calc_x_offset_to_dynamic_obtsacle(robot_center,dynamic_obstacle_center): 
    x_dist = dynamic_obstacle_center[0]-robot_center[0]
    y_dist = dynamic_obstacle_center[1] - robot_center[1]
    
    return math.atan2()