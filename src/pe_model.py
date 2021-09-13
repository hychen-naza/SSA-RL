'''
    The probabilistic ensemble dynamics model
'''
# pylint: disable=C0103, R0902, R0913, W0201, E0401, E1120
import time
import itertools
import numpy as np
import tensorflow as tf
from tensorflow import keras

class PEModel(keras.Model):
    '''
        An individual Probabilistic Neural Network.
        Multiple Networks with identical structure form the Probabilistic Ensemble.
        Notice that each PEModel network predicts the mean and variance of
        reward, done, delta_state in order.
        Therefore, the output layer has (state_dim + 1 + 1) * 2
    '''
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # print(f"state_dim {state_dim}, action_dim {action_dim}")
        self.l1 = keras.layers.Dense(256, activation="relu")
        self.l2 = keras.layers.Dense(256, activation="relu")
        self.l3 = keras.layers.Dense(256, activation="relu")
        # mean and variance for reward, done, delta_state (in this order)
        self.l4 = keras.layers.Dense((state_dim + 2) * 2)
        # this step to populate trainable_weights. Without this step,
        # PE.trainable_weights will be empty.
        self.forward(np.zeros((1, state_dim + action_dim)))

    def forward(self, net_input):
        '''
            Calls the network on a batch of inputs.
            net_input should have size (batch_size, state_dim+action_dim)
        '''
        out = self.l1(net_input)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        return out

class PE():
    '''
        The probabilistic ensemble dynamics model class.
        Contains code to initialize, train and then predict with the ensemble.
        You will implement part of this class.
    '''
    def __init__(
        self,
        state_dim,
        action_dim,
        num_networks = 7,
        num_elites = 5,
        learning_rate = 1e-3,
    ):
        self.num_networks = num_networks
        self.num_elites = num_elites
        self.networks = [PEModel(state_dim, action_dim) for i in range(num_networks)]
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.output_dim = state_dim + 2
        # For smoothing the log-variance output
        self.max_logvar = tf.convert_to_tensor(-3 * np.ones([1, self.state_dim + 2]), \
                                               dtype=tf.float32)
        self.min_logvar = tf.convert_to_tensor(-7 * np.ones([1, self.state_dim + 2]), \
                                               dtype=tf.float32)

        self.total_it = 0
        self._model_inds = list(range(self.num_networks)) # for choosing elite models in inference!
        self.reward_mse = float("inf")

    def get_output(self, output, ret_logvar=False):
        """
            output: tf tensor, shape (batch_size, (state_dim+2) * 2)
            Given network outputs, returns mean and log variance tf tensors if ret_logvar = True.
            mean: shape (batch_size, state_dim + 2)
            logvar: shape (batch_size, state_dim + 2)
            Do not modify
        """
        mean = output[:, 0:self.output_dim]
        raw_v = output[:, self.output_dim:]
        # Log variance smoothing
        logvar = self.max_logvar - tf.math.softplus(self.max_logvar - raw_v)
        logvar = self.min_logvar + tf.math.softplus(logvar - self.min_logvar)
        if ret_logvar: # for training
            return mean, logvar
        return mean, tf.math.exp(logvar) # for testing

    def _train_loss_one(self, network, train_in, train_targ):
        '''
            Compute the MLE Training Loss for a given Probabilistic Neural Network.
            train_in: tf tensor, shape (batch_size, state_dim + action_dim)
            tarin_targ: tf tensor, shape (batch_size, state_dim + 2), target output
            This function should compute the Gaussian MLE loss, summed across the entire batch.
        '''
        # raise NotImplementedError
        train_out = network.forward(train_in)
        mean, logvar = self.get_output(train_out, True)
        # train_loss = 0.5 * (self.output_dim * tf.math.log(tf.constant(2 * np.pi)) + tf.math.reduce_sum(logvar, 1))
        # train_loss += tf.math.reduce_sum(0.5 * tf.math.divide(tf.math.pow(train_targ - mean, 2), tf.math.exp(logvar)), 1)
        # train_loss = tf.math.reduce_sum(train_loss)
        train_loss = tf.math.reduce_sum(logvar)
        train_loss += tf.math.reduce_sum(tf.math.divide(tf.math.pow(train_targ - mean, 2), tf.math.exp(logvar)))
 
        # regularization step. populate train_loss with correct Gaussian MLE loss
        train_loss += 0.01 * (tf.math.reduce_sum(self.max_logvar) - \
                      tf.math.reduce_sum(self.min_logvar))
        return train_loss


    def _MSE_loss(self, valid_in, valid_targ, final=False):
        """
            Computes the MSE loss for each Probabilistic Neural Network, for validation only.
            valid_in: tf tensor, shape (batch_size, state_dim + action_dim), validation input
            valid_targ: tf tensor, shape (batch_size, state_dim + 2), validation target
            Do not modify.
        """
        mse_losses = np.zeros(self.num_networks)
        rew_losses = np.zeros(self.num_networks)
        not_done_losses = np.zeros(self.num_networks)
        dynamics_losses = np.zeros(self.num_networks)

        for i, network in enumerate(self.networks):
            mean, _ = self.get_output(network.forward(valid_in), ret_logvar=True)
            if final:
                mse_loss = tf.reduce_mean(((mean - valid_targ) ** 2), 0)
                rew_loss = mse_loss[0]
                not_done_loss = mse_loss[1]
                dynamics_loss = tf.reduce_mean(mse_loss[2:], 0)
                mse_losses[i] = tf.reduce_mean(mse_loss, 0)
                rew_losses[i] = rew_loss
                not_done_losses[i] = not_done_loss
                dynamics_losses[i] = dynamics_loss
            else:
                mse_loss = tf.reduce_mean((mean - valid_targ) ** 2, 0)
                mse_losses[i] = tf.reduce_mean(mse_loss, 0)
        if final:
            return mse_losses, rew_losses, not_done_losses, dynamics_losses
        return mse_losses

    def _prepare_dataset(self, buffer):
        '''
            Given a replay buffer containing real environment transitions,
            prepare a dataset for training the PE of neural networks.
            The dataset contains ALL transitions in the replay buffer.
            Do not modify.
            inputs: tf tensor, shape (buffer_size, state_dim + action_dim)
            targets: tf tensor, shape (buffer_size, state_dim + 2)
        '''
        state, action, next_state, reward, not_done = buffer.sample_all() # already shuffled

        delta_state = next_state - state
        inputs = tf.concat((state, action), -1)
        targets = tf.concat((reward, not_done, delta_state), -1)
        # Both TF tensors
        return inputs, targets

    def _start_train(self, max_epochs_since_update):
        '''
            Setup some internal bookkeeping variables to determine convergence.
            Do not modify.
        '''
        self._snapshots = np.array([1e10 for i in range(self.num_networks)])
        self._epochs_since_update = 0
        self._max_epochs_since_update = max_epochs_since_update

    def _end_train(self):
        '''
            Book keeping and console output. Do not modify.
        '''
        sorted_inds = np.argsort(self._snapshots)
        self._model_inds = sorted_inds[:self.num_elites].tolist() # first elite models
        '''
        print('Final holdout_losses: ', self._snapshots)
        print('Model MSE', np.mean(self._snapshots[self._model_inds]))
        print('Rew MSE', np.mean(self._reward_mse[self._model_inds]))
        print('Not Done MSE', np.mean(self._not_done_mse[self._model_inds]))
        print('Dyn MSE', np.mean(self._dynamics_mse[self._model_inds]))
        '''
        self.reward_mse = np.mean(self._reward_mse[self._model_inds])

    def _save_best(self, epoch, holdout_losses):
        '''
            Determines the stopping condition for PE model training.
            The training is determined to have converged if for max_epochs_since_update epochs,
            no network in the ensemble has improved for more than 1%.
            Do not modify.
        '''
        updated = False
        for i in range(len(holdout_losses)):
            current = holdout_losses[i]
            best = self._snapshots[i]
            improvement = (best - current) / best
            if improvement > 0.01: # if decrease over 1%, save
                self._snapshots[i] = current
                #self._save_model(i)
                updated = True
                #print('epoch {} | updated {} | improvement: {:.4f} | best: {:.4f} | current: {:.4f}'.format(\
                #    epoch, i, improvement, best, current))

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1

        if self._epochs_since_update > self._max_epochs_since_update:
            #print('[ PE ] Breaking at epoch {}: {} epochs since update ({} max)'.format(epoch,
            #    self._epochs_since_update, self._max_epochs_since_update))
            return True
        else:
            return False


    def train(self, buffer, batch_size=256, holdout_ratio=0.2, max_logging=5000,
                max_grad_updates=None, max_t=None, max_epochs_since_update=5):
        '''
            For model training, uses all transitions in real buffer, and train to convergence
            in valid set. You will implement part of this training function.
        '''
        self._start_train(max_epochs_since_update)
        inputs, targets = self._prepare_dataset(buffer)

        # Split into training and holdout sets
        num_holdout = min(int(inputs.shape[0] * holdout_ratio), max_logging)
        inputs, holdout_inputs = inputs[num_holdout:], inputs[:num_holdout]
        targets, holdout_targets = targets[num_holdout:], targets[:num_holdout]

        #print('[ Euler PE ] Training {} | Target {} | Holdout: {}'.format(inputs.shape, targets.shape,
        #        holdout_inputs.shape))

        idxs = tf.convert_to_tensor(np.random.randint(inputs.shape[0], size=(inputs.shape[0],)))
        num_batch = int(np.ceil(idxs.shape[-1] / batch_size))

        # global counter
        t0 = time.time()
        grad_updates = 0

        for epoch in itertools.count(): # infinite loop
            for batch_num in range(num_batch):
                batch_idxs = idxs[batch_num * batch_size:(batch_num + 1) * batch_size]
                # (N, <=B): will include the remainder batch even if out of bounds!
                train_in = tf.gather(inputs, batch_idxs)
                train_targ = tf.gather(targets, batch_idxs)
                
                # For each network, get loss, compute gradient of loss
                # And apply optimizer step.
                # raise NotImplementedError
                # print(f"train_in {train_in}")
                for network in self.networks:
                    with tf.GradientTape() as tape:
                        loss = self._train_loss_one(network, train_in, train_targ)
                    grad = tape.gradient(loss, network.trainable_weights)
                    self.optimizer.apply_gradients(zip(grad, network.trainable_weights))
                
                grad_updates += 1

            idxs = tf.random.shuffle(idxs) # shuffle its dataset for each model

            # validate each model using same valid set
            holdout_losses = self._MSE_loss(holdout_inputs, holdout_targets) # (N,)
            break_train = self._save_best(epoch, holdout_losses)
            #print("[ PE ] holdout_losses: ", f"Epoch {epoch}", holdout_losses) # write to log.txt

            t = time.time() - t0
            if break_train or (max_grad_updates and grad_updates > max_grad_updates):
                break

            if max_t and t > max_t:
                print('Breaking because of timeout: {}! (max: {})'.format(t, max_t))
                break

        self._snapshots, self._reward_mse, self._not_done_mse, self._dynamics_mse \
            = self._MSE_loss(holdout_inputs, holdout_targets, final=True)

        self._end_train()
        #print(f"End of Model training {epoch} epochs and time {t:.0f}s")
        #print('Model training epoch', epoch)
        #print('Model training time', int(t))
        return grad_updates

    ### Rollout / Inference Code

    def _prepare_input(self, state, action):
        '''
            Prepares inputs for inference.
            state: tf tensor, size (batch_size, state_dim) or (state_dim, )
            action: tf tensor, size (batch_size, action_dim) or (action_dim, )
            inputs: tf tensor, size (batch_size, state_dim + action_dim)
            Do not modify.
        '''
        if state.ndim == 1:
            state = tf.expand_dims(state, 0)
        if action.ndim == 1:
            action = tf.expand_dims(action, 0) \
                     if action.shape[0] == self.action_dim else tf.expand_dims(action, 1)
        inputs = tf.concat((state, action), -1)
        assert inputs.ndim == 2
        return inputs

    def _random_inds(self, batch_size):
        '''
            Uniformly randomly pick one *elite* model for each (state, action) in batch.
            This may help you implement predict.
        '''
        inds = np.random.choice(self._model_inds, size=batch_size)
        return inds

    def predict(self, state, action, deterministic=False):
        '''
            Predicts next states, rewards and not_done using the probabilistic ensemble
            For each (state, action) pair, pick a elite model uniformly at random, then
            use that elite model to predict next state, reward and not_done. The model
            can de different for each sample in the batch.
            If deterministic=True, then the prediction should simply be the predicted mean.
            If deterministic=False, then the prediction should be sampled from N(mean, var),
            where mean is the predicted mean and var is the predicted variance.
            state: np tensor, shape (batch_size, state_dim) or (state_dim, )
            action: np tensor, shape (batch_size, action_dim) or (action_dim, )
            samples (return value): np tensor, shape (batch_size, state_dim+2)
            samples[:, 0] should be the rewards, samples[:, 1] should be the not-done signals,
            and samples[:, 2:] should be the next states.
        '''
        inputs = self._prepare_input(state, action)

        # raise NotImplementedError
        batch_size = inputs.shape[0]
        inds = self._random_inds(batch_size)
        outputs = [np.squeeze(self.networks[inds[i]].forward(inputs[i][None,:])) for i in range(batch_size)]
        mean, var = self.get_output(np.array(outputs))
        mean[:, 2:] += state
        if deterministic:
            return mean
        samples = mean + tf.math.sqrt(var) * tf.random.normal(var.shape)
        return samples


# Sanity Check to test your PE model implementation.
if __name__ == '__main__':
    import pybullet_envs
    import gym
    import utils

    env = gym.make("InvertedPendulumBulletEnv-v0")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    replay_buffer = utils.ReplayBuffer(state_size, action_size, max_size=int(1e6))

    o = env.reset()
    total_steps = 25000 # one episode has 1000 steps
    step = 0
    while step < total_steps:
        a = env.action_space.sample()
        o2, r, d, info = env.step(a)
        step += 1
        replay_buffer.add(o, a, o2, r, float(d))
        o = o2
        if d:
            o = env.reset()

    model = PE(state_size, action_size)
    model.train(replay_buffer)
