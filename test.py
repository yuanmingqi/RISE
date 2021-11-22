import gym

env = gym.make('SpaceInvadersNoFrameskip-v4')
print(env.action_space.n, env.action_space.__class__.__name__)


#####hahahha