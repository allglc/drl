import numpy as np
import tensorflow as tf


class GreedyPolicy():

    def __init__(self, nb_actions):
        self.nb_actions = nb_actions

    def select_action(self, state, q_network):

        action = np.argmax(q_network.predict(state.reshape((1,-1))))

        return action


class EpsGreedyPolicy():

    def __init__(self, epsilon, nb_actions):
        self.epsilon = epsilon
        self.nb_actions = nb_actions

    def select_action(self, state, q_network):
        if np.random.random() <= self.epsilon:
            action = np.random.randint(self.nb_actions)
        else:
            action = np.argmax(q_network.predict(state.reshape((1,-1))))

        return action


class LinearlyDecreasingEpsGreedyPolicy():

    def __init__(self, epsilon_max, epsilon_min, nb_steps, nb_actions):
        self.epsilon = epsilon_max
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.nb_steps = round(nb_steps)
        self.nb_actions = nb_actions

    def select_action(self, state, q_network):
        self.epsilon = np.maximum(self.epsilon - (self.epsilon_max-self.epsilon_min)/self.nb_steps, self.epsilon_min)
        if np.random.random() <= self.epsilon:
            action = np.random.randint(self.nb_actions)
        else:
            action = np.argmax(q_network.predict(state.reshape((1,-1))))

        return action

