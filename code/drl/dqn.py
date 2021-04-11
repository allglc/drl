import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense


class DQN():
    """DQN Agent from Mnih et al., 2015, Human level control through deep reinforcement learning"""
    
    def __init__(self, env, neurons_per_layer, LR=0.001, replay_memory_max_size=100000, discount_factor=0.99, sarsa=False, double_dqn=False):

        self.discount_factor = discount_factor
        self.sarsa = sarsa
        self.double_dqn = double_dqn

        regu = tf.keras.regularizers.l2(1e-4)

        # Q network
        self.q_network = tf.keras.Sequential()
        self.q_network.add(Dense(neurons_per_layer[0], input_shape=env.observation_space.shape,
                                 activation='relu', kernel_regularizer=regu, bias_regularizer=regu))           
        for i in range(len(neurons_per_layer)-1): 
            self.q_network.add(Dense(neurons_per_layer[i+1], activation='relu', kernel_regularizer=regu, bias_regularizer=regu))
        self.q_network.add(Dense(env.action_space.n, activation='linear', kernel_regularizer=regu, bias_regularizer=regu))
        self.q_network.compile(optimizer=tf.keras.optimizers.Adam(LR), loss='mse')

        # Target Q network
        self.target_q_network = tf.keras.models.clone_model(self.q_network)
        self.target_q_network.set_weights(self.q_network.get_weights())

        # Experience (replay memory)
        self.replay_memory_max_size = replay_memory_max_size
        self.exp = {'states':np.zeros((replay_memory_max_size, env.observation_space.shape[0])), 
                    'actions':np.zeros(replay_memory_max_size, dtype=int),
                    'rewards':np.zeros(replay_memory_max_size), 
                    'states_next':np.zeros((replay_memory_max_size, env.observation_space.shape[0])),
                    'done':np.zeros(replay_memory_max_size, dtype=bool)}
            
        self.exp_len = 0 # used to find random indexes only on indexes where samples were saved; plafonates at the value f the max size of replay memory
        self.exp_idx = 0 # used to know when the memory is full, and is reset so that the memory is refiled, the saved samples are replaced with new samples
        
        self.states_meanQ = np.array([env.reset() for i in range(100)]) # list random states to evaluate the evolution of Q predicted

        self.warmup_finished = False


    def _warmup(self, env, warmup_steps):
        """Pre-fill the experience with some transitions obtained with random actions applied on the environment"""
        
        print(f'Warmup: random actions for {warmup_steps} steps to pre-fill experience')
        done = False
        for step in range(warmup_steps):
            if (step == 0) | done: state = env.reset()

            # Select random action
            action = np.random.randint(env.action_space.n)
         
            # Execute action, see result (new state, reward and if episode is done)
            state_next, reward, done, _ = env.step(action)

            # Save transition
            self._store_transition(state, action, reward, state_next, done)
            state = state_next

        self.warmup_finished = True
            
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
            

    def _train_network(self, policy, minibatch_size, target_network_update_period, nb_training_steps, step_):

        # Train Q network on a random minibatch of transitions
        idx = np.random.randint(self.exp_len, size=minibatch_size) # Sample random minibatch of transitions (standard DQN)
        s = self.exp['states'][idx]
        a = self.exp['actions'][idx]
        r = self.exp['rewards'][idx]
        s_next = self.exp['states_next'][idx]
        d = self.exp['done'][idx]

        # Produce y_target 
        y_target = np.zeros(shape=minibatch_size)
        # cases episodes done:
        y_target[d] = r[d]
        # cases episodes not done:
        if self.double_dqn:
            if self.sarsa: # double SARSA
                rows = np.arange(0, len(s_next[~d]))
                cols = [policy.select_action(s, self.q_network) for s in s_next[~d]]
                max_Q = self.target_q_network.predict(s_next[~d])[rows, cols]
            else: # double DQN
                rows = np.arange(0, len(s_next[~d]))
                cols = np.argmax(self.q_network.predict(s_next[~d]), axis=1)
                max_Q = self.target_q_network.predict(s_next[~d])[rows, cols]
        else:
            if self.sarsa: # SARSA
                rows = np.arange(0, len(s_next[~d]))
                cols = [policy.select_action(s, self.target_q_network) for s in s_next[~d]]
                max_Q = self.target_q_network.predict(s_next[~d])[rows, cols]
            else: # standard DQN
                max_Q = np.max(self.target_q_network.predict(s_next[~d]), axis=1)
                   
        y_target[~d] = r[~d] + self.discount_factor*max_Q

        # Initialize Y with a model prediction to ensure the loss corresponding to actions not taken is null
        Y = self.q_network.predict(s)
        for i, a_i in enumerate(a):
            Y[i, a_i] = y_target[i]
        # Train model on minibatch
        self.q_network.fit(s, Y, verbose=0)

        # Update network Q target
        # Hard update
        if target_network_update_period >= 1:
            if (step_ % target_network_update_period) == 0:
                self.target_q_network.set_weights(self.q_network.get_weights())
        # Soft update
        else:
            tau = target_network_update_period
            q_network_theta = self.q_network.get_weights()
            target_network_theta = self.target_q_network.get_weights()
            counter = 0
            for q_weight, target_weight in zip(q_network_theta, target_network_theta):
                target_weight = target_weight*(1-tau) + q_weight*tau
                target_network_theta[counter] = target_weight
                counter += 1
            self.target_q_network.set_weights(target_network_theta)
        
        return None    


    def train(self, env, policy, nb_training_steps, minibatch_size=32, target_network_update_period=100, warmup_steps=0, visualize=False):
        """Train Q network on a random minibatch of transitions"""
        
        # Warmup phase to fill up experience before training
        if warmup_steps>0 and not self.warmup_finished: self._warmup(env, warmup_steps)

        # Initialize
        state = env.reset()
        done = False
        total_reward = 0
        start_time = time.time()
        list_total_reward = []
        list_meanQ = []

        # Train
        for step_ in range(1, nb_training_steps+1):

            # Select action according to policy
            action = policy.select_action(state, self.q_network)

            # Execute action, see result (new state, reward and if episode is done)
            state_next, reward, done, _ = env.step(action)

            # Save transition
            self._store_transition(state, action, reward, state_next, done)
            state = state_next

            # Train network
            self._train_network(policy, minibatch_size, target_network_update_period, nb_training_steps, step_)

            # Save information
            total_reward += reward
            
            # Visualize
            if visualize:
                if done: time.sleep(0.5)
                env.render()

            # Episode finished
            if done:
                # Save and print info
                list_total_reward.append(total_reward)
                list_meanQ.append(np.mean(self.q_network.predict(self.states_meanQ)))
                # print('Training episode #{} done ({}/{} steps), elapsed time: {:.0f}s, total reward: {:.2f}, mean Q: {:.2f}'.format(
                #     len(list_total_reward), step_, nb_training_steps, (time.time()-start_time), list_total_reward[-1], list_meanQ[-1]))
                # if hasattr(policy, 'epsilon'): print('   (epsilon = {:.2f})'.format(policy.epsilon))
                # Reset environment
                state = env.reset()
                done = False
                total_reward = 0
                start_time = time.time()

        return list_total_reward, list_meanQ


    def test(self, env, policy, nb_testing_steps, visualize=False):
        """Test agent on potentially new environment and with new policy"""

        # Initialize
        state = env.reset()
        done = False
        total_reward = 0
        start_time = time.time()
        list_total_reward = []

        # Test
        for step_ in range(1, nb_testing_steps+1):

            # Select action according to policy
            action = policy.select_action(state, self.q_network)

            # Execute action, see result (new state, reward and if episode is done)
            state, reward, done, _ = env.step(action)

            # Save information
            total_reward += reward
            
            # Visualize
            if visualize:
                if done: time.sleep(0.5)
                env.render()

            # Episode finished
            if done:
                # Save and print info
                list_total_reward.append(total_reward)
                # print('Testing episode #{} done ({}/{} steps), elapsed time: {:.0f}s, total reward: {:.2f}'.format(
                #     len(list_total_reward)+1, step_, nb_testing_steps, (time.time()-start_time), list_total_reward[-1]))
                # Reset environment
                state = env.reset()
                done = False
                total_reward = 0
                start_time = time.time()

        return list_total_reward
    
    
    def save_model(self, path=None):
        '''Save Q network in chosen path

        Args:
            path ([Path], optional): [save the model at this location]. Defaults to None.
        '''
        
        self.q_network.save(path)