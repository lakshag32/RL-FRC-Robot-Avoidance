import ntcore
import time

#if __name__ == "__main__": #TODO: THIS MAY BE NEEDED
inst = ntcore.NetworkTableInstance.getDefault()
table = inst.getTable("RobotPosition")
xSub = table.getDoubleTopic("x").subscribe(0)
ySub = table.getDoubleTopic("y").subscribe(0); 
anglePub = table.getDoubleTopic("robotAngle").publish(); 
inst.startClient4("Coprocessor")
inst.setServerTeam(972) # where TEAM=190, 294, etc, or use inst.setServer("hostname") or similar

    # while True:
    #     time.sleep(1)

    #     x = xSub.get()
    #     y = ySub.get()
    #     print(f"X: {x} Y: {y}")

def get_robot_coord():
    #time.sleep(1) #TODO: THIS MAY BE NEEDED not sure if this is neccesary
    return [100+50*xSub.get(), 100+50*ySub.get()]

def publish_drive_angle(angle):
    anglePub.set(float(int(angle))) 
