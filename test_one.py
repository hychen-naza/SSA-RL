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

# project files
import asteroid
import bounds
import craft
import cases
import simu_env
import runner
import tensorflow as tf
from tensorflow import keras
from turtle_display import TurtleRunnerDisplay
from utils import ReplayBuffer
from td3 import TD3
from pe_model import PE
from fake_env import FakeEnv
from ssa import SafeSetAlgorithm
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

    if dname == 'turtle':
        return TurtleRunnerDisplay(800,800)
    else:
        return runner.BaseRunnerDisplay()


def run_kwargs( params ):

    in_bounds = bounds.BoundsRectangle( **params['in_bounds'] )

    goal_bounds = bounds.BoundsRectangle( **params['goal_bounds'] )

    min_dist = params['min_dist']

    ret = { 'field': asteroid.AsteroidField(),
            'craft_state': craft.CraftState( **( params['initial_craft_state'] ) ),
            'in_bounds': in_bounds,
            'goal_bounds': goal_bounds,
            'noise_sigma': params['noise_sigma'],
            'min_dist': min_dist,
            'nsteps': 1000 }

    return ret

def parser():
    prsr = argparse.ArgumentParser()
    prsr.add_argument( 'method',
                       help="Which method to test",
                       type=str,
                       choices=('estimate', 'navigate'),
                       default='estimate')
    prsr.add_argument( '--case',
                       help="test case number (one of %s) or test case file" % list(cases.index.keys()),
                       type=str,
                       default=1)
    prsr.add_argument( '--display',
                       choices=('turtle','text','none'),
                       default='none' )
    prsr.add_argument( '--qp',dest='is_qp', action='store_true')
    prsr.add_argument( '--no-qp',dest='is_qp', action='store_false')
    prsr.add_argument( '--ssa-buffer',dest='enable_ssa_buffer', action='store_true')
    prsr.add_argument( '--no-ssa-buffer',dest='enable_ssa_buffer', action='store_false')
    #parser.set_defaults(is_qp=True)
    return prsr

def main(method_name, case_id, display_name, qp, enable_ssa_buffer):
    # testing env
    try:
        params = cases.index[ int(case_id) ]
    except Exception as e:
        print(e)
        return
    display = display_for_name(display_name)
    env_params = run_kwargs(params)
    
    # rl policy
    craft_state_size = 4 #(x,y,v_x,v_y)
    craft_action_size = 2
    nearest_obstacle_state_size = 2 #(delta_x, delta_y)
    state_dim = craft_state_size + nearest_obstacle_state_size

    model_update_freq = 1000
    env_model = PE(state_dim = craft_state_size, action_dim = craft_action_size) #Dynamics model
    fake_env = FakeEnv(env_model) # FakeEnv to help model unrolling
    env = simu_env.Env(display, fake_env, **(env_params))

    policy_replay_buffer = ReplayBuffer(state_dim = state_dim, action_dim = craft_action_size, max_size=int(1e6))
    policy = TD3(state_dim, craft_action_size, env.max_acc, env.max_acc)
    #policy.load("rl_model/rl")
    ssa_replay_buffer = ReplayBuffer(state_dim = state_dim, action_dim = craft_action_size, max_size=int(1e6))
    # ssa
    safe_controller = SafeSetAlgorithm(max_speed = env.craft_state.max_speed, fake_env = fake_env, is_qp = qp)
    # cbf
    # safe_controller = ControlBarrierFunction()
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

    robot_xs = []
    robot_ys = []
    obs_xs = []
    obs_ys = []
    safe_obs_xs = []
    safe_obs_ys = []
    for t in range(max_steps):
      #if (env.cur_step == 0):
      #  policy.parameter_explore()
      #  print(f"parameter_explore in {t}")
      '''
      if (t > 1024):
        with tf.GradientTape() as tape:
          state_batch, action_batch, next_state_batch, reward_batch, not_done_batch =  policy_replay_buffer.sample(256)
          q_fixed = rnd_fixed.call(state_batch)
          q_train = rnd_train.call(state_batch)
          loss = rnd_loss(q_fixed, q_train)
          gradients = tape.gradient(loss, rnd_train.trainable_weights)
          rnd_optimizer.apply_gradients(zip(gradients, rnd_train.trainable_weights))
      '''
      action = policy.select_action(state)
      #if (t % 50 == 0):
      #  print(action)
      env.display_start()
      # ssa parameters
      unsafe_obstacle_ids, unsafe_obstacles = env.find_unsafe_obstacles(env.min_dist * 6)
      original_action = action
      action, is_safe, is_unavoidable, danger_obs = safe_controller.get_safe_control(state[:4], unsafe_obstacles, fx, gx, action)
      '''
      if (len(danger_obs) > 0):
        for obs in danger_obs:
          obs_xs.append(obs[0])
          obs_ys.append(obs[1])

      for obs in env.field.asteroids:
        safe_obs_xs.append(obs.c_x)
        safe_obs_ys.append(obs.c_y)
      '''
      #is_safe = False
      # take safe action
      #robot_xs.append(state[0])
      #robot_ys.append(state[1])
      s_new, reward, done, info = env.step(action, is_safe, unsafe_obstacle_ids) 
      original_reward = reward
      # RND
      '''
      rnd_state = tf.convert_to_tensor(state.reshape(1, -1))
      q_fixed = rnd_fixed.call(rnd_state)
      q_train = rnd_train.call(rnd_state)                    
      loss = np.sum(np.square(q_fixed - q_train))      
      reward += loss
      '''
      episode_reward += original_reward
      env.display_end()
      # Store data in replay buffer
      # Every SSA + RL should use safe action to train RL policy
      if (enable_ssa_buffer):
        if (is_safe):
          ssa_replay_buffer.add(state, action, s_new, reward, done)          
        else:
          policy_replay_buffer.add(state, action, s_new, reward, done)
      else:
        #print("here")
        policy_replay_buffer.add(state, original_action, s_new, reward, done) # 直接把safe u拿来学习可以么
      state = s_new
      if (policy_replay_buffer.size > 1024):
        state_batch, action_batch, next_state_batch, reward_batch, not_done_batch =  [np.array(x) for x in policy_replay_buffer.sample(256)]
        if enable_ssa_buffer and ssa_replay_buffer.size > 128:
            model_batch_size = int(0.4*256)
            idx = np.random.choice(256, model_batch_size, replace=False)
            state_batch[idx], action_batch[idx], next_state_batch[idx], reward_batch[idx], not_done_batch[idx] =  ssa_replay_buffer.sample(model_batch_size)
        policy.train_on_batch(state_batch, action_batch, next_state_batch, reward_batch, not_done_batch)

      if (done and original_reward == -500):   
        #print("collision")         
        #print(safe_controller.records) 
        collision_num += 1        
      elif (done and original_reward == 2000):
        success_num += 1
      elif (done):
        failure_num += 1
      
      if (done):      
        total_steps += env.cur_step
        print(f"Train: episode_num {episode_num}, total_steps {total_steps}, reward {episode_reward}, is_qp {qp}, last state {state[:4]}")
        total_rewards.append(episode_reward)
        episode_reward = 0
        episode_num += 1
        state, done = env.reset(), False
        '''
        if (qp):
          np.save('trajectory_result/xs_qp.npy', np.array(robot_xs))
          np.save('trajectory_result/ys_qp.npy', np.array(robot_ys))
          np.save('trajectory_result/obs_xs_qp.npy', np.array(obs_xs))
          np.save('trajectory_result/obs_ys_qp.npy', np.array(obs_ys))
        else:
          np.save('trajectory_result/xs.npy', np.array(robot_xs))
          np.save('trajectory_result/ys.npy', np.array(robot_ys))
          np.save('trajectory_result/obs_xs.npy', np.array(obs_xs))
          np.save('trajectory_result/obs_ys.npy', np.array(obs_ys))
          np.save('trajectory_result/safe_obs_xs.npy', np.array(safe_obs_xs))
          np.save('trajectory_result/safe_obs_ys.npy', np.array(safe_obs_ys))

        robot_xs = []
        robot_ys = []
        obs_xs = []
        obs_ys = []
        safe_obs_xs = []
        safe_obs_ys = []
        '''
      #if (len(total_rewards) >= 20 and np.mean(total_rewards[-20:]) >= 1900 and not is_meet_requirement):
      #  print(f"\n\n\nWe meet the reward threshold episode_num {episode_num}, total_steps {total_steps}\n\n\n")
      #  is_meet_requirement = True
      #  break
      #if (episode_num >= 50):
      #  print(f"success_num {success_num}, failure_num {failure_num}, collision_num {collision_num}, avg_reward {np.mean(total_rewards)}")
      #  break
      
      if (t % 1000 == 0):
        #print(f"avg reward {np.mean(total_rewards)}, success {success_num}, failure {failure_num}, collision {collision_num}")
        env.save_env()
        eval_reward = eval(policy, env, safe_controller, fx, gx)
        print(f"t {t}, eval_reward {eval_reward}")
        reward_records.append(eval_reward)
        env.read_env()
        #print(reward_records)
        if (len(reward_records) == 100):
          break
      
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
    for i in range(10):
      reward_records = main( method_name  = args.method,
          case_id      = args.case,
          display_name = args.display,
          qp = args.is_qp,
          enable_ssa_buffer = args.enable_ssa_buffer)
      for j, n in enumerate(reward_records):
        all_reward_records[j].append(n)
      print(all_reward_records)
    np.save('plot_result/ssa_rl.npy', np.array(all_reward_records))

