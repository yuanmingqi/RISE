import sys
import numpy as np
import math
import random

import gym
import gym_maze


def simulate():
	# Instantiating the learning related parameters
	learning_rate = get_learning_rate(0)
	explore_rate = get_explore_rate(0)
	discount_factor = 0.99

	num_streaks = 0

	# Render tha maze
	# env.render()

	rise = RISE(total_states_num=np.prod(MAZE_SIZE), alpha=0.1)
	for episode in range(NUM_EPISODES):

		# Reset the environment
		obv = env.reset()

		# the initial state
		state_0 = state_to_bucket(obv)
		total_reward = 0

		for t in range(MAX_T):
			# Select an action
			action = select_action(state_0, explore_rate)

			# execute the action
			obv, reward, done, _ = env.step(action)

			# Observe the result
			state = state_to_bucket(obv)
			total_reward += reward
			reward += rise.compute_intrinsic_reward(state[0] * MAZE_SIZE[0] + state[1])

			# Update the Q based on the result
			best_q = np.amax(q_table[state])
			q_table[state_0 + (action,)] += learning_rate * (
						reward + discount_factor * (best_q) - q_table[state_0 + (action,)])

			# Setting up for the next iteration
			state_0 = state

			# Print data
			if DEBUG_MODE == 2:
				print("\nEpisode = %d" % episode)
				print("t = %d" % t)
				print("Action: %d" % action)
				print("State: %s" % str(state))
				print("Reward: %f" % reward)
				print("Best Q: %f" % best_q)
				print("Explore rate: %f" % explore_rate)
				print("Learning rate: %f" % learning_rate)
				print("Streaks: %d" % num_streaks)
				print("")

			elif DEBUG_MODE == 1:
				if done or t >= MAX_T - 1:
					print("\nEpisode = %d" % episode)
					print("t = %d" % t)
					print("Explore rate: %f" % explore_rate)
					print("Learning rate: %f" % learning_rate)
					print("Streaks: %d" % num_streaks)
					print("Total reward: %f" % total_reward)
					print("")

			# Render tha maze
			if RENDER_MAZE:
				env.render()

			if env.is_game_over():
				sys.exit()

			if done:
				num_visited_states = np.unique(rise.visited_states).shape[0]
				rise.total_exploration_steps += t
				print("Episode %d finished after %f time steps with total reward = %f (streak %d), visited states = %d."
				      % (episode, t, total_reward, num_streaks, num_visited_states
				         ))

				if num_visited_states == np.prod(MAZE_SIZE):
					print('Total exploration steps ', rise.total_exploration_steps)
					exit(0)

				if t <= SOLVED_T:
					num_streaks += 1
				else:
					num_streaks = 0
				break

			elif t >= MAX_T - 1:
				print("Episode %d timed out at %d with total reward = %f."
				      % (episode, t, total_reward))

		# It's considered done when it's solved over 120 times consecutively
		if num_streaks > STREAK_TO_END:
			print('Total exploration steps ', rise.total_exploration_steps)
			break

		# Update parameters
		explore_rate = get_explore_rate(episode)
		learning_rate = get_learning_rate(episode)


def select_action(state, explore_rate):
	# Select a random action
	if random.random() < explore_rate:
		action = env.action_space.sample()
	# Select the action with the highest q
	else:
		action = int(np.argmax(q_table[state]))
	return action


def get_explore_rate(t):
	return max(MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t + 1) / DECAY_FACTOR)))


def get_learning_rate(t):
	return max(MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t + 1) / DECAY_FACTOR)))


def state_to_bucket(state):
	bucket_indice = []
	for i in range(len(state)):
		if state[i] <= STATE_BOUNDS[i][0]:
			bucket_index = 0
		elif state[i] >= STATE_BOUNDS[i][1]:
			bucket_index = NUM_BUCKETS[i] - 1
		else:
			# Mapping the state bounds to the bucket array
			bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
			offset = (NUM_BUCKETS[i] - 1) * STATE_BOUNDS[i][0] / bound_width
			scaling = (NUM_BUCKETS[i] - 1) / bound_width
			bucket_index = int(round(scaling * state[i] - offset))
		bucket_indice.append(bucket_index)
	return tuple(bucket_indice)



class RISE:
	def __init__(self, total_states_num, alpha):
		self.visited_states = []
		self.counter = np.zeros(shape=(total_states_num))
		self.alpha = alpha
		self.total_exploration_steps = 0

	def compute_intrinsic_reward(self, new_state):
		self.visited_states.append(new_state)
		self.counter[new_state] += 1
		prob = self.counter[new_state] / np.sum(self.counter)
		reward = np.power(prob, self.alpha)

		return reward

# class JFI:
# 	def __init__(self, total_states_num):
# 		self.visited_states = []
# 		self.total_states_num = total_states_num
# 		self.former_jfi = 1. / total_states_num
# 		self.total_exploration_steps = 0
#
# 	def compute_jfi(self, new_state):
# 		self.visited_states.append(new_state)
# 		counter = np.zeros(shape=(self.total_states_num,))
# 		for i, visited_state in enumerate(self.visited_states):
# 			counter[visited_state] += 1
#
# 		current_jfi = np.power(np.mean(counter), 2) / np.mean(np.power(counter, 2))
#
# 		reward = 0.99 * current_jfi - self.former_jfi
#
# 		self.former_jfi = current_jfi
#
# 		return reward


if __name__ == "__main__":

	# Initialize the "maze" environment
	env = gym.make("maze-random-30x30-plus-v0")

	'''
	Defining the environment related constants
	'''
	# Number of discrete states (bucket) per state dimension
	MAZE_SIZE = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
	NUM_BUCKETS = MAZE_SIZE  # one bucket per grid

	# Number of discrete actions
	NUM_ACTIONS = env.action_space.n  # ["N", "S", "E", "W"]
	# Bounds for each discrete state
	STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))

	'''
	Learning related constants
	'''
	MIN_EXPLORE_RATE = 0.001
	MIN_LEARNING_RATE = 0.2
	DECAY_FACTOR = np.prod(MAZE_SIZE, dtype=float) / 10.0

	'''
	Defining the simulation related constants
	'''
	NUM_EPISODES = 50000
	MAX_T = np.prod(MAZE_SIZE, dtype=int) * 100
	STREAK_TO_END = 100
	SOLVED_T = np.prod(MAZE_SIZE, dtype=int)
	DEBUG_MODE = 0
	RENDER_MAZE = True
	ENABLE_RECORDING = False

	'''
	Creating a Q-Table for each state-action pair
	'''
	q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,), dtype=float)

	'''
	Begin simulation
	'''
	recording_folder = "/tmp/maze_rise"

	if ENABLE_RECORDING:
		env.monitor.start(recording_folder, force=True)

	simulate()

	if ENABLE_RECORDING:
		env.monitor.close()
