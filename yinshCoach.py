# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from kerasify import export_model

EPISODES = 100
STATE_SIZE = 10
BATCH_SIZE = 32
experienceFile = "YinshExp.exp"

# Deep Q-learning Agent
class DQNAgent:
    def __init__(self, state_size):
        self.state_size = state_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.00025
        self.model = self._build_model()
        self.reward = 0

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.compile(loss='mean_squared_error', optimizer='adamax')
        return model

    def updateReward(self, newReward):
        self.reward = newReward

    def remember(self, state, next_states):
        self.memory.append((state, next_states))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        i = 0
        while i < len(minibatch):
            target = self.reward
            state, next_states = minibatch[i]
            if i != (len(minibatch) - 1):
              target = self.reward + self.gamma * \
                       np.amax(self.model.predict(next_states)[0])
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
		
    def kerasify(self, name):
        export_model(self.model, 'yinsh.model')
		

if __name__ == "__main__":
    agent = DQNAgent(STATE_SIZE)
    with open(experienceFile) as f:
        # Parse reward of experience
        reward = f.readline()
        agent.updateReward(reward)
        # Parse file for state and next state
        state = f.readline()
        next_states = f.readline()
        agent.remember(state, next_states)

    # Iterate the game experience
    for e in range(EPISODES):
        # train the agent with the experience of the episode
        agent.replay(BATCH_SIZE)
        print("episode: {}/{}, score: {}"
        .format(e, EPISODES, agent.reward))
        