from stable_baselines3.common.env_checker import check_env
from env import OnlyStaticEnv
env = OnlyStaticEnv([50,50],[50,100])
check_env(env)