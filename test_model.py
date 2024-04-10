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

#training loop code from: https://pythonprogramming.net/introduction-reinforcement-learning-stable-baselines-3-tutorial/
while True:

	if math.dist(env.robot,env.target) <= 30: 
		break 

	action, _states = model.predict(obs)
    
	resized_action = -np.float32(np.interp(action,[-1,1],[-180,180]))[0]

	print(resized_action)

	# obs,reward,terminated,truncated,info = env.step(action)

	# env.render()