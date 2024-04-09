from stable_baselines3.common.env_checker import check_env
from coproc_only_static import OnlyStaticEnv
env = OnlyStaticEnv([50,50],[50,100])
print(check_env(env))