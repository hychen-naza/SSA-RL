from builtins import range
from builtins import object
import math
import numpy as np
import tensorflow as tf
import random
import copy
import time
import random

SUCCESS = 'success'
FAILURE_TOO_MANY_STEPS = 'too_many_steps'

# Custom failure states for navigation.
NAV_FAILURE_COLLISION = 'collision'
NAV_FAILURE_OUT_OF_BOUNDS = 'out_of_bounds'

def l2( xy0, xy1 ):
    ox = xy1[0]
    oy = xy1[1]
    dx = xy0[0] - xy1[0]
    dy = xy0[1] - xy1[1]
    dist = math.sqrt( (dx * dx) + (dy * dy) )
    if (xy1[0] < -0.9):
    	warp_dx = xy0[0] - (1 + (xy1[0] + 1))
    	dist1 = math.sqrt( (warp_dx * warp_dx) + (dy * dy) )
    	if (dist1 < dist):
    		ox = (1 + (xy1[0] + 1))
    		dist = dist1
    elif (xy1[0] > 0.9):
    	warp_dx = xy0[0] - (-1 + (xy1[0] - 1))
    	dist1 = math.sqrt( (warp_dx * warp_dx) + (dy * dy) )
    	if (dist1 < dist):
    		ox = (-1 + (xy1[0] - 1))
    		dist = dist1
    return dist, ox, oy



class Env(object):
	def __init__(self, display, field, robot_state,
                    min_dist,
                    noise_sigma,
                    in_bounds,
                    goal_bounds,
                    nsteps):

		self.init_robot_state = copy.deepcopy(robot_state)
		self.robot_state = copy.deepcopy(self.init_robot_state)

		self.field = field
		self.display = display

		self.min_dist = min_dist
		self.in_bounds = in_bounds
		self.goal_bounds = goal_bounds
		self.nsteps = nsteps
		self.cur_step = 0

		self.max_acc = 0.005
		self.max_steering = np.pi / 8

		self.forecast_steps = 5

	def reset(self):
		self.cur_step = 0
		self.robot_state = copy.deepcopy(self.init_robot_state)
		self.display.setup( self.field.x_bounds, self.field.y_bounds,
							self.in_bounds, self.goal_bounds,
							margin = self.min_dist)
		self.field.random_init() # randomize the init position of obstacles
		cx,cy,_ = self.robot_state.position
		obstacle_id, obstacle_pos, _ = self.find_nearest_obstacle(cx,cy)
		state = [cx, cy, self.robot_state.v_x, self.robot_state.v_y]
		relative_pos = [cx - obstacle_pos[0], cy - obstacle_pos[1]]
		return np.array(state + relative_pos)

	def find_nearest_obstacle(self, cx, cy, unsafe_obstacle_ids = []):
		astlocs = self.field.obstacle_locations(self.cur_step, cx, cy, self.min_dist * 5)
		nearest_obstacle = None
		nearest_obstacle_id = -1
		nearest_obstacle_dist = np.float("inf")    
		collisions = ()
		for i,x,y in astlocs:
			self.display.obstacle_at_loc(i,x,y)
			if (i in unsafe_obstacle_ids):
				self.display.obstacle_set_color(i, 'blue')
			dist, ox, oy = l2( (cx,cy), (x,y) )
			if dist < self.min_dist:
				collisions += (i,)
			if dist < nearest_obstacle_dist:
				nearest_obstacle_dist = dist
				nearest_obstacle = [ox, oy]
				nearest_obstacle_id = i
		if (nearest_obstacle_id == -1):
			nearest_obstacle = [-1, -1]
		return nearest_obstacle_id, nearest_obstacle, collisions

	def display_start(self):
		self.display.begin_time_step(self.cur_step)

	def display_end(self):
		self.display.end_time_step(self.cur_step)

	def save_env(self):
		self.cur_step_copy = self.cur_step
		self.robot_state_copy = copy.deepcopy(self.robot_state)
		self.field_copy = copy.deepcopy(self.field)
		return

	def read_env(self):
		self.cur_step = self.cur_step_copy
		self.robot_state = self.robot_state_copy
		self.field = self.field_copy
		return 

	def step(self, action, is_safe = False, unsafe_obstacle_ids = []):
		'''
		action: [dv_x, dv_y]
		'''
		self.cur_step += 1
		self.robot_state = self.robot_state.steer( action[0], action[1] )
		cx,cy,ch = self.robot_state.position
		self.display.robot_at_loc( cx, cy, ch, is_safe)
		nearest_obstacle_id, nearest_obstacle, collisions = self.find_nearest_obstacle(cx, cy, unsafe_obstacle_ids)

		next_robot_state = [cx, cy, self.robot_state.v_x, self.robot_state.v_y]
		relative_pos = [cx - nearest_obstacle[0], cy - nearest_obstacle[1]]
		next_state = next_robot_state + relative_pos

		# done
		done = False
		arrive = False
		reward_wo_cost = 0
		if collisions:
			ret = (NAV_FAILURE_COLLISION, self.cur_step)
			self.display.navigation_done(*ret)
			done = True
			reward = -500
		elif self.goal_bounds.contains( (cx,cy) ):
			ret = (SUCCESS, self.cur_step)
			self.display.navigation_done(*ret)
			done = True
			reward = 2000
			arrive = True
		#elif not self.in_bounds.contains( (cx,cy) ):
		#	done = True
		#	reward = -1000
		elif self.cur_step > self.nsteps:
			done = True
			reward = 0
		else:
			# rewards depend on current state
			relative_dist = np.sqrt(relative_pos[0]**2 + relative_pos[1]**2)
			reward = 0 #(cy+1) - min((0.1 / relative_dist), 1)
			reward_wo_cost = 0 #(cy+1)
		info = {'arrive':arrive, 'reward_wo_cost':reward_wo_cost}
		return np.array(next_state), reward, done, info

	def random_action(self):
		dv_x = (2 * random.random() - 1) * self.max_acc
		dv_y = (2 * random.random() - 1) * self.max_acc
		return [dv_x, dv_y]

	def suboptimal_control(self):
		# find path for unsafe obstacles
		unsafe_obstacle_ids, unsafe_obstacle_info = self.find_unsafe_obstacles(self.min_dist * 10)
		obstacle_paths = []
		for id in unsafe_obstacle_ids:
			obstacle_path = []
			obstacle = self.field.obstacles[id]
			for i in range(1, self.forecast_steps+1):
				obstacle_path.append([obstacle.x(self.cur_step+i), obstacle.y(self.cur_step+i)])
			obstacle_paths.append(obstacle_path)
		# find possible paths for vehicle
		possible_ax = [-0.005, -0.0025, -0.001, 0.001, 0.0025, 0.005] 
		possible_ay = [-0.005, -0.0025, -0.001, 0.001, 0.0025, 0.005] 
		scores = {}
		for ax in possible_ax:
			for ay in possible_ay:
				vehicle_path = []
				vx = min(self.robot_state.v_x + ax, self.robot_state.max_speed)
				vy = min(self.robot_state.v_y + ay, self.robot_state.max_speed)
				for i in range(1, self.forecast_steps+1):
					vehicle_path.append([self.robot_state.x+vx*i, self.robot_state.y+vy*i])
				# score vehicle paths and find the path with highest score
				scores[(ax, ay)] = self.score_path(vehicle_path, obstacle_paths, vy)
				scores[(ax, ay)] += 10*vy
				'''
				if (self.fake_env.model.reward_mse < 1e-3):
					robot_state = np.array([self.robot_state.x, self.robot_state.y, self.robot_state.v_x, self.robot_state.v_y])
					action = np.array([ax, ay])
					# _, reward, _ = self.fake_env.step(robot_state, action)
					#print(f"action {action}, scores {scores[(ax, ay)]}, reward {reward[0]}, obstacle_paths {len(obstacle_paths)}")
					scores[(ax, ay)] += 10*vy#10*reward[0]
				'''
				
		best_ax, best_ay = max(scores, key = scores.get)
		#if (self.fake_env.model.reward_mse < 1e-3):
		#	print(f"pick action [{best_ax}, {best_ay}]")
		return [best_ax, best_ay]

	def score_path(self, vehicle_path, obstacle_paths, vy):
		score = 0
		for obstacle_path in obstacle_paths:
			for i in range(len(obstacle_path)):
				obstacle_pos = obstacle_path[i]
				vehicle_pos = vehicle_path[i]
				distance = math.sqrt((vehicle_pos[0]-obstacle_pos[0])**2 + (vehicle_pos[1]-obstacle_pos[1])**2)
				score += distance
				if (distance < self.min_dist * 3):
					score += -3*distance
		# print(f"score {score}, vy {vy}")
		#score += vy * 8
		return score 

	def find_unsafe_obstacles(self, min_dist):
		cx, cy, _ = self.robot_state.position
		unsafe_obstacles = self.field.unsafe_obstacle_locations(self.cur_step, cx, cy, min_dist)
		unsafe_obstacle_ids = [ele[0] for ele in unsafe_obstacles]
		unsafe_obstacle_info = [np.array(ele[1]) for ele in unsafe_obstacles]
		return unsafe_obstacle_ids, unsafe_obstacle_info
