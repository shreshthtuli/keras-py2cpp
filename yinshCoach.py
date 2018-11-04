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

EPISODES = 10
STATE_SIZE = 10

# Deep Q-learning Agent
class DQNAgent:
    def __init__(self, state_size):
        self.state_size = state_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
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

    def remember(self, state):
        self.memory.append(state)

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        i = 0
        while i < len(minibatch):
            target = self.reward
            state = minibatch[i]
            if i != (len(minibatch) - 1):
              next_state = minibatch[i+1]
              target = self.reward + self.gamma * \
                       np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
		
    def kerasify(self, name):
        export_model(self.model, 'yinsh.model')
		


if __name__ == "__main__":
    agent = DQNAgent(STATE_SIZE)
    # Iterate the game
    for e in range(EPISODES):
        # time_t represents each frame of the game
        # Our goal is to keep the pole upright as long as possible until score of 500
        # the more time_t the more score
        for time_t in range(500):
            # Decide action
            action = agent.act(state)
            # Advance the game to the next frame based on the action.
            # Reward is 1 for every frame the pole survived
            next_state = np.reshape(next_state, [1, 4])
            # Remember the previous state, action, reward, and done
            agent.remember(state)
            # make next_state the new current state for the next frame.
            state = next_state
            # done becomes True when the game ends
            # ex) The agent drops the pole
            if done:
                # print the score and break out of the loop
                print("episode: {}/{}, score: {}"
                      .format(e, EPISODES, time_t))
                break
        # train the agent with the experience of the episode
        agent.replay(32)