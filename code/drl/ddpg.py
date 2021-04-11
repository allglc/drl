import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

class DDPG():
    """ DDPG Agent from Lillicrap and al., 2016, Continuous Control with Deep Reinforcement Learning"""
    
    def __init__(self, env, neurons_per_layer, replay_memory_max_size, discount_factor, L2_coeff=0, LR_critic=0.001, LR_actor=0.0001, noise_std=0.1):
        
        nb_actions = np.prod(env.action_space.shape)
        dim_states = np.prod(env.observation_space.shape)
        self.discount_factor = discount_factor
        self._action_range_agent = np.tile([-1, 1], (nb_actions, 1))
        self._action_range_env = np.array((env.action_space.low, env.action_space.high)).T

        self.warmup_finished = False
        regu = tf.keras.regularizers.l2(L2_coeff)
        self.noise_std = noise_std
        
        # Optimizer
        self.optimizer_critic = tf.keras.optimizers.Adam(LR_critic)
        self.optimizer_actor = tf.keras.optimizers.Adam(LR_actor)
        
        # Critic network
        self.critic_network = tf.keras.Sequential()
        self.critic_network.add(Dense(neurons_per_layer[0], activation='relu', kernel_regularizer=regu, bias_regularizer=regu,
                                      input_shape=(dim_states+nb_actions, )))           
        for i in range(len(neurons_per_layer)-1): 
            self.critic_network.add(Dense(neurons_per_layer[i+1], activation='relu', kernel_regularizer=regu, bias_regularizer=regu))
        self.critic_network.add(Dense(1, activation='linear', kernel_regularizer=regu, bias_regularizer=regu))
        self.critic_network.compile(self.optimizer_critic, loss='mse')
        
        # Actor network
        self.actor_network = tf.keras.Sequential()
        self.actor_network.add(Dense(neurons_per_layer[0], activation='relu', kernel_regularizer=regu, bias_regularizer=regu,
                                     input_shape=(dim_states, )))           
        for i in range(len(neurons_per_layer)-1): 
            self.actor_network.add(Dense(neurons_per_layer[i+1], activation='relu', kernel_regularizer=regu, bias_regularizer=regu))
        self.actor_network.add(Dense(nb_actions, activation='tanh', kernel_regularizer=regu, bias_regularizer=regu,
                                     kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.003, maxval=0.003))) # to ensure gradient so close to 0 as we're using tanh
        
        # Target networks
        self.target_critic_network = tf.keras.models.clone_model(self.critic_network)
        self.target_critic_network.set_weights(self.critic_network.get_weights())
        self.target_actor_network = tf.keras.models.clone_model(self.actor_network)
        self.target_actor_network.set_weights(self.actor_network.get_weights())
        
        # Experience (replay memory)
        self.replay_memory_max_size = replay_memory_max_size
        self.exp = {'states':np.zeros((replay_memory_max_size, dim_states)),
                    'actions':np.zeros((replay_memory_max_size, nb_actions)),
                    'rewards':np.zeros(replay_memory_max_size), 
                    'states_next':np.zeros((replay_memory_max_size, dim_states)),
                    'done':np.zeros(replay_memory_max_size, dtype=bool)}
        
        self.exp_len = 0 # used to find random indexes only on indexes where samples were saved; plafonates at the value of the max size of replay memory
        self.exp_idx = 0 # used to know when the memory is full, and is reset so that the memory is refiled, the saved samples are replaced with new samples


    def _scale_action(self, action):
        action = action.numpy()
        for i in range(action.shape[1]):
            action[0, i] = ((action[0, i] - self._action_range_agent[i, 0]) / (self._action_range_agent[i, 1] - self._action_range_agent[i, 0])
                         * (self._action_range_env[i, 1] - self._action_range_env[i, 0]) + self._action_range_env[i,  0])

        return action.flatten()


    def _warmup(self, env, warmup_steps):
        """Pre-fill the experience with some transitions obtained with random actions applied on the environment"""
        
        print(f'Warmup: random actions for {warmup_steps} steps to pre-fill experience')
        done = False
        for step in range(warmup_steps):
            if (step == 0) | done: state = env.reset()

            # Select random action
            action = np.random.normal(0, 0.1, size=env.action_space.shape)
         
            # Execute action, see result (new state, reward and if episode is done)
            state_next, reward, done, _ = env.step(action)

            # Save transition
            self._store_transition(state, action, reward, state_next, done)
            state = state_next
        self.warmup_finished = True
        
        return None
        
        
    def _store_transition(self, state, action, reward, state_next, done):
        """Store transition (state, action, reward, state_next, done) in replay memory"""
        
        # Check if replay memory is full and drop first elements if so
        if self.exp_idx == self.replay_memory_max_size - 1:
            self.exp_idx = 0
        
        # Store transition
        self.exp['states'][self.exp_idx] = state
        self.exp['actions'][self.exp_idx] = action
        self.exp['rewards'][self.exp_idx] = reward
        self.exp['states_next'][self.exp_idx] = state_next
        self.exp['done'][self.exp_idx] = done

        self.exp_idx += 1
        self.exp_len = np.minimum(self.exp_len + 1, self.replay_memory_max_size)
        
        
    def _train_networks(self, minibatch_size, tau_update_target):

        # Sample random minibatch of transitions (standard DQN)
        idx = np.random.randint(self.exp_len, size=minibatch_size)
        s = self.exp['states'][idx]
        a = self.exp['actions'][idx]
        r = self.exp['rewards'][idx]
        s_next = self.exp['states_next'][idx]
        d = self.exp['done'][idx]

        # Produce target action value y
        y = np.zeros(shape=minibatch_size)
        # cases terminal state:
        y[d] = r[d]
        # cases not terminal state:
        action_next = self.target_actor_network(s_next[~d])
        state_action_next = tf.concat([s_next[~d], action_next], 1)
        Q_next = self.target_critic_network(state_action_next)
        Q_next = tf.reshape(Q_next, [-1])
        y[~d] = r[~d] + self.discount_factor*Q_next

        # Train critic network on minibatch
        state_action = tf.concat([s, a], 1)
        self.critic_network.fit(state_action, y, verbose=0)
        
        # Train actor network on minibatch
        with tf.GradientTape() as tape:
            a = self.actor_network(s)
            state_action = tf.concat([s, a], 1)
            Q = self.critic_network(state_action)
            Q = - Q # we want to maximize Q
        gradients = tape.gradient(Q, self.actor_network.trainable_variables)
        self.optimizer_actor.apply_gradients(zip(gradients, self.actor_network.trainable_variables))

        # Update target critic network
        critic_network_theta = self.critic_network.get_weights()
        target_network_theta = self.target_critic_network.get_weights()
        counter = 0
        for weight, target_weight in zip(critic_network_theta, target_network_theta):
            target_weight = target_weight*(1-tau_update_target) + weight*tau_update_target
            target_network_theta[counter] = target_weight
            counter += 1
        self.target_critic_network.set_weights(target_network_theta)
        
        # Update target actor network
        actor_network_theta = self.actor_network.get_weights()
        target_network_theta = self.target_actor_network.get_weights()
        counter = 0
        for weight, target_weight in zip(actor_network_theta, target_network_theta):
            target_weight = target_weight*(1-tau_update_target) + weight*tau_update_target
            target_network_theta[counter] = target_weight
            counter += 1
        self.target_actor_network.set_weights(target_network_theta)
        
        return None
    
    
    def train(self, env, nb_training_steps, minibatch_size=32, tau_update_target=0.1, warmup_steps=3000, visualize=False):
        
        # Warmup phase to fill up experience before training
        if warmup_steps>0 and not self.warmup_finished: self._warmup(env, warmup_steps)

        # Initialize
        state = env.reset()
        done = False
        total_reward = 0
        start_time = time.time()
        list_total_reward = []
        list_steps = []

        # Train
        for step in range(1, nb_training_steps+1):

            # Noisy action
            noise = np.random.normal(0, self.noise_std)
            action = self.actor_network(state.reshape((1, -1))) + noise
            action_scaled = self._scale_action(action)
            # Execute action, see result (new state, reward and if episode is done)
            state_next, reward, done, _ = env.step(action_scaled)
            state_next = state_next.flatten()

            # Save transition
            self._store_transition(state, action, reward, state_next, done)
            state = state_next

            # Train network
            self._train_networks(minibatch_size, tau_update_target)

            # Save information
            total_reward += float(reward)
            
            # Visualize
            if visualize:
                if done: time.sleep(0.5)
                env.render()

            # Episode finished
            if done:
                # Save and print info
                list_total_reward.append(total_reward)
                list_steps.append(step)
                # print('Training episode #{} done ({}/{} steps), elapsed time: {:.0f}s, total reward: {:.2f}'.format(
                #     len(list_total_reward), step, nb_training_steps, (time.time()-start_time), list_total_reward[-1]))
                
                # Reset environment
                state = env.reset()
                done = False
                total_reward = 0
                start_time = time.time()
                
        return list_total_reward, list_steps
    
    
    def test(self, env, nb_testing_steps, visualize=False):

        # Initialize
        state = env.reset()
        done = False
        total_reward = 0
        start_time = time.time()
        list_total_reward = []

        # Test
        for step_ in range(1, nb_testing_steps+1):

            # non-noisy action
            action = self.actor_network(state.reshape((1, -1)))
            action_scaled = self._scale_action(action)

            # Execute action, see result (new state, reward and if episode is done)
            state_next, reward, done, _ = env.step(action_scaled)            
            state = state_next

            # Save information
            total_reward += float(reward)
            
            # Visualize
            if visualize:
                if done: time.sleep(0.5)
                env.render()

            # Episode finished
            if done:
                # Save and print info
                list_total_reward.append(total_reward)
                
                # Reset environment
                state = env.reset()
                done = False
                total_reward = 0
                
        # print('Testing finished ({} steps), elapsed time: {:.0f}s, average reward: {:.2f}'.format(
        #             nb_testing_steps, (time.time()-start_time), np.mean(list_total_reward)))

        return list_total_reward