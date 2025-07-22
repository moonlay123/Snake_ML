import pygame
import numpy as np
import random
import sys
from collections import deque
from tensorflow.keras import layers, models, optimizers
from snake import Snake_game

class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        model = models.Sequential([
            layers.Reshape((*self.state_shape, 1)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=optimizers.Adam(0.001))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state[np.newaxis], verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state[np.newaxis], verbose=0)[0])
            target_f = self.model.predict(state[np.newaxis], verbose=0)
            target_f[0][action] = target
            self.model.fit(state[np.newaxis], target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_dqn():
    game = Snake_game(train_mode=True)
    agent = DQNAgent(state_shape=(game.GRID_WIDTH, game.GRID_HEIGHT), action_size=3)
    episodes = 1000
    batch_size = 32

    for e in range(episodes):
        state = game.reset()
        total_reward = 0
        while not game.done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            action = agent.act(state)
            next_state, reward, done = game.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if e % 5 == 0:
                game.render()
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        print(f"Episode: {e}, Score: {game.score}, Epsilon: {agent.epsilon:.2f}")
