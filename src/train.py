from __future__ import print_function
from __future__ import absolute_import

######################################################################
# This file copyright the Georgia Institute of Technology
#
# Permission is given to students to use or modify this file (only)
# to work on their assignments.
#
# You may NOT publish this file or make it available to others not in
# the course.
#
######################################################################

# python modules
import argparse
import importlib
import math
import random
import numpy as np
import os.path
import sys
import collections
import tensorflow as tf
from tensorflow import keras

# project files
import dynamic_obstacle
import bounds
import robot # two integrator robot
import simu_env
import runner
import param
from turtle_display import TurtleRunnerDisplay
from utils import ReplayBuffer
from td3 import TD3
from pe_model import PE
from fake_env import FakeEnv
from ssa import SafeSetAlgorithm
from cautious_rl import ProbabiilisticShield
from cbf import ControlBarrierFunction
'''
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)
'''
class RND(keras.Model):
    '''
        RND
    '''
    def __init__(self):
        super().__init__()
        self.l1 = keras.layers.Dense(128, activation="relu")
        self.l2 = keras.layers.Dense(128)

    def call(self, state):
        '''
            Returns the output for both critics. Using during critic training.
        '''
        if not tf.is_tensor(state):
            state = tf.convert_to_tensor(state)
        q1 = self.l1(state)
        q1 = self.l2(q1)
        return q1

def display_for_name( dname ):
    # choose none display or visual display
    if dname == 'turtle':
        return TurtleRunnerDisplay(800,800)
    else:
        return runner.BaseRunnerDisplay()


def run_kwargs( params ):
    in_bounds = bounds.BoundsRectangle( **params['in_bounds'] )
    goal_bounds = bounds.BoundsRectangle( **params['goal_bounds'] )
    min_dist = params['min_dist']
    ret = { 'field': dynamic_obstacle.ObstacleField(),
            'robot_state': robot.DoubleIntegratorRobot( **( params['initial_robot_state'] ) ),
            'in_bounds': in_bounds,
            'goal_bounds': goal_bounds,
            'noise_sigma': params['noise_sigma'],
            'min_dist': min_dist,
            'nsteps': 1000 }
    return ret

def parser():
    prsr = argparse.ArgumentParser()
    prsr.add_argument( '--display',
                       choices=('turtle','text','none'),
                       default='none' )
    prsr.add_argument( '--explore',
                   choices=('psn','rnd','none'),
                   default='none' )
    prsr.add_argument( '--qp',dest='is_qp', action='store_true')
    prsr.add_argument( '--no-qp',dest='is_qp', action='store_false')
    prsr.add_argument( '--ssa-buffer',dest='enable_ssa_buffer', action='store_true')
    prsr.add_argument( '--no-ssa-buffer',dest='enable_ssa_buffer', action='store_false')
    return prsr

def main(display_name, exploration, qp, enable_ssa_buffer):
    # testing env
    try:
        params = param.params
    except Exception as e:
        print(e)
        return
    display = display_for_name(display_name)
    env_params = run_kwargs(params)
    
    # rl policy
    robot_state_size = 4 #(x,y,v_x,v_y)
    robot_action_size = 2
    nearest_obstacle_state_size = 2 #(delta_x, delta_y)
    state_dim = robot_state_size + nearest_obstacle_state_size

    model_update_freq = 1000
    env = simu_env.Env(display, **(env_params))

    policy_replay_buffer = ReplayBuffer(state_dim = state_dim, action_dim = robot_action_size, max_size=int(1e6))
    policy = TD3(state_dim, robot_action_size, env.max_acc, env.max_acc, exploration = exploration)
    #policy.load("./model/ssa1")
    ssa_replay_buffer = ReplayBuffer(state_dim = state_dim, action_dim = robot_action_size, max_size=int(1e6))
    # ssa
    safe_controller = SafeSetAlgorithm(max_speed = env.robot_state.max_speed, is_qp = qp)
    cbf_controller = ControlBarrierFunction(max_speed = env.robot_state.max_speed)
    shield_controller = ProbabiilisticShield(max_speed = env.robot_state.max_speed)
    # parameters
    max_steps = int(1e6)
    start_timesteps = 2e3
    episode_reward = 0
    episode_num = 0
    last_episode_reward = 0
    teacher_forcing_rate = 0
    total_rewards = []
    total_steps = 0
    # dynamic model parameters
    fx = np.array([[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]])
    gx = np.array([[1,0],[0,1],[1,0],[0,1]])
    state, done = env.reset(), False
    collision_num = 0
    failure_num = 0
    success_num = 0

    # Random Network Distillation
    rnd_fixed = RND()
    rnd_train = RND()
    rnd_optimizer = keras.optimizers.Adam(learning_rate=3e-4)
    rnd_loss = keras.losses.MeanSquaredError()

    is_meet_requirement = False
    reward_records = []

    for t in range(max_steps):
      # disturb the policy parameters at beginning of each episodes when using PSN
      if (exploration == 'psn' and env.cur_step == 0):
        policy.parameter_explore()
        print(f"parameter_explore in {t}")
      
      # train the random network prediction when using rnd
      if (exploration == 'rnd' and t > 1024):
        with tf.GradientTape() as tape:
          state_batch, action_batch, next_state_batch, reward_batch, not_done_batch =  policy_replay_buffer.sample(256)
          q_fixed = rnd_fixed.call(state_batch)
          q_train = rnd_train.call(state_batch)
          loss = rnd_loss(q_fixed, q_train)
          gradients = tape.gradient(loss, rnd_train.trainable_weights)
          rnd_optimizer.apply_gradients(zip(gradients, rnd_train.trainable_weights))
      
      action = policy.select_action(state)
      original_action = action
      env.display_start()
      # ssa parameters
      unsafe_obstacle_ids, unsafe_obstacles = env.find_unsafe_obstacles(env.min_dist * 6)
      #safe_action = cautious_control(env.field, env.robot_state, unsafe_obstacle_ids, unsafe_obstacles, env.cur_step, env.min_dist)
      #action, is_safe = cbf_controller.get_safe_control(state[:4], unsafe_obstacles, fx, gx, action)
      #action, is_safe = shield_controller.probshield_control(state[:4], unsafe_obstacles, fx, gx, action, env.field, unsafe_obstacle_ids, unsafe_obstacles, env.cur_step)
      action, is_safe, is_unavoidable, danger_obs = safe_controller.get_safe_control(state[:4], unsafe_obstacles, fx, gx, action)
      #is_safe = False
      # take safe action
      s_new, reward, done, info = env.step(action, is_safe, unsafe_obstacle_ids) 
      original_reward = reward
      episode_reward += original_reward
      # add the novelty to reward when using rnd
      if (exploration == 'rnd'):
        rnd_state = tf.convert_to_tensor(state.reshape(1, -1))
        q_fixed = rnd_fixed.call(rnd_state)
        q_train = rnd_train.call(rnd_state)                    
        loss = np.sum(np.square(q_fixed - q_train))      
        reward += loss      
      
      env.display_end()
      # Store data in replay buffer
      if (enable_ssa_buffer):
        if (is_safe):
          ssa_replay_buffer.add(state, action, s_new, reward, done)          
        else:
          policy_replay_buffer.add(state, action, s_new, reward, done)
      else:
        policy_replay_buffer.add(state, original_action, s_new, reward, done)
      old_state = state
      state = s_new
      
      # train policy
      if (policy_replay_buffer.size > 1024):
        state_batch, action_batch, next_state_batch, reward_batch, not_done_batch =  [np.array(x) for x in policy_replay_buffer.sample(256)]
        if enable_ssa_buffer and ssa_replay_buffer.size > 128:
            model_batch_size = int(0.4*256) # batch size is 256, ratio is 0.4
            idx = np.random.choice(256, model_batch_size, replace=False)
            state_batch[idx], action_batch[idx], next_state_batch[idx], reward_batch[idx], not_done_batch[idx] =  ssa_replay_buffer.sample(model_batch_size)
        policy.train_on_batch(state_batch, action_batch, next_state_batch, reward_batch, not_done_batch)

      if (done and original_reward == -500):          
        #print(safe_controller.records) 
        collision_num += 1      
        # plot control half-space
        #    state[:4], unsafe_obstacles, fx, gx, action
        #safe_controller.plot_control_subspace(old_state[:4], unsafe_obstacles, fx, gx, original_action)
        #break
      elif (done and original_reward == 2000):
        success_num += 1
      elif (done):
        failure_num += 1
      
      if (done):      
        total_steps += env.cur_step
        print(f"Train: episode_num {episode_num}, total_steps {total_steps}, reward {episode_reward}, is_qp {qp}, exploration {exploration}, last state {state[:2]}")
        total_rewards.append(episode_reward)
        episode_reward = 0
        episode_num += 1
        state, done = env.reset(), False
        if (episode_num >= 100):
          policy.save("./model/ssa1")
          break

      # check reward threshold
      '''
      if (len(total_rewards) >= 20 and np.mean(total_rewards[-20:]) >= 1900 and not is_meet_requirement):
        print(f"\n\n\nWe meet the reward threshold episode_num {episode_num}, total_steps {total_steps}\n\n\n")
        is_meet_requirement = True
        break
      '''

      # evalution part at every 1000 steps
      '''
      if (t % 1000 == 0):
        env.save_env()
        eval_reward = eval(policy, env, safe_controller, fx, gx)
        print(f"t {t}, eval_reward {eval_reward}")
        reward_records.append(eval_reward)
        env.read_env()
        if (len(reward_records) == 100):
          break
      '''

    return reward_records

def eval(policy, env, safe_controller, fx, gx):
  episode_num = 0
  episode_reward = 0
  state, done = env.reset(), False
  episode_rewards = []
  arrives = []
  while (True):
    action = policy.select_action(state)  
    unsafe_obstacle_ids, unsafe_obstacles = env.find_unsafe_obstacles(env.min_dist * 6)
    action, _, _,_ = safe_controller.get_safe_control(state[:4], unsafe_obstacles, fx, gx, action)
    s_new, reward, done, info = env.step(action)
    episode_reward += reward
    state = s_new
    if (done):
      state, done = env.reset(), False
      return episode_reward

if __name__ == '__main__':
    args = parser().parse_args()
    all_reward_records = []
    for i in range(100):
      all_reward_records.append([])
    for i in range(1):
      reward_records = main(display_name = args.display, 
          exploration = args.explore,
          qp = args.is_qp,
          enable_ssa_buffer = args.enable_ssa_buffer)
      for j, n in enumerate(reward_records):
        all_reward_records[j].append(n)
      print(all_reward_records)
    #np.save('plot_result/ssa_rl.npy', np.array(all_reward_records))

