'''
    This module contains the FakeEnv class, which is a wrapper class around the
    PE dynamics model to help unrolling.
'''
import tensorflow as tf

class FakeEnv:
    '''
        A wrapper class around your dynamics model to facilitate model unrolling.
    '''
    def __init__(self, model):
        self.model = model

    def step(self, state, action):
        '''
            state: (B, X) tensor/array, action: (B, U) tensor/array
            X is dimension of state space
            U is dimensino of action space
            B is batch size
            Given state and action , the step function queries the dynamics model,
            and returns next_x (the next states), rewards and the boolean done signal,
            as numpy arrays.
            Do not modify.
        '''
        if not tf.is_tensor(state):
            state = tf.convert_to_tensor(state.astype("float32"))
        if not tf.is_tensor(action):
            action = tf.convert_to_tensor(action.astype("float32"))

        samples = self.model.predict(state, action) # (B, -1)
        # (B,), (B,), (B, X)
        rewards, not_done, next_x = samples[:, 0], samples[:, 1], samples[:, 2:]
        done = (1. - not_done)

        # Convert to all numpy, done should be boolean
        return next_x, rewards, (done > 0.5)
