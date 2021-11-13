from builtins import range
from builtins import object
import math
import numpy as np
import random
import copy
import time
import random


class ProbabiilisticShield():
	def __init__(self, max_speed, dmin = 0.12, k = 1, max_acc = 0.04):
		self.dmin = dmin
		self.k = k
		self.max_speed = max_speed
		self.max_acc = max_acc
		self.forecast_step = 3

	def probshield_control(self, robot_state, obs_states, f, g, u0, field, unsafe_obstacle_ids, unsafe_obstacles, cur_step):
		# find path for unsafe obstacles
		#forecast_steps = 
		u0 = np.array(u0).reshape((2,1))
		is_safe = True
		for i, obs_state in enumerate(obs_states):
			d = np.array(robot_state - obs_state[:4])
			d_pos = d[:2] # pos distance
			d_vel = d[2:] # vel 
			d_abs = np.linalg.norm(d_pos)
			d_dot = self.k * (d_pos @ d_vel.T) / np.linalg.norm(d_pos)
			phi = np.power(self.dmin, 2) - np.power(np.linalg.norm(d_pos), 2) - d_dot
			if (phi > 0):
				is_safe = False

		if (not is_safe):
			obstacle_paths = []
			for id in unsafe_obstacle_ids:
				obstacle_path = []
				obstacle = field.asteroids[id]
				for i in range(1, 6):
					obstacle_path.append([obstacle.x(cur_step+i), obstacle.y(cur_step+i)])
				obstacle_paths.append(obstacle_path)
			# find possible paths for vehicle
			possible_ax = [-0.04, -0.02, -0.01, -0.005, -0.0025, -0.001, 0, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.04] 
			possible_ay = [-0.04, -0.02, -0.01, -0.005, -0.0025, -0.001, 0, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.04] 
			scores = {}
			for ax in possible_ax:
				for ay in possible_ay:
					vehicle_path = []
					vx = max(min(robot_state[2] + ax, self.max_speed), -self.max_speed)
					vy = max(min(robot_state[3] + ay, self.max_speed), -self.max_speed)
					for i in range(1, 6):
						vehicle_path.append([robot_state[0]+vx*i, robot_state[1]+vy*i])
					# score vehicle paths and find the path with highest score
					scores[(ax, ay)] = self.score_path(vehicle_path, obstacle_paths, vy)					
			best_ax, best_ay = max(scores, key = scores.get)
			#import pdb
			#pdb.set_trace()
			return np.array([best_ax, best_ay]), True
		else:
			u0 = u0.reshape(1,2)
			return u0[0], False

	def score_path(self, vehicle_path, obstacle_paths, vy):
		score = 0
		for obstacle_path in obstacle_paths:
			for i in range(len(obstacle_path)):
				obstacle_pos = obstacle_path[i]
				vehicle_pos = vehicle_path[i]
				distance = math.sqrt((vehicle_pos[0]-obstacle_pos[0])**2 + (vehicle_pos[1]-obstacle_pos[1])**2)
				score += distance
				if (distance < self.dmin * 3):
					score -= 3*distance
				if (distance < self.dmin):
					score -= float("inf")
		return score 