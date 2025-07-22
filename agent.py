import pygame
import numpy as np
import random
import sys
from collections import deque
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.models import save_model, load_model
from snake import Snake_game
import os, json, glob

class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model, self.start_episode = self._build_model()

    def _build_model(self):
        try:
            model_files = glob.glob("saved_models/snake_dqn_episode_*.keras")
            if model_files:
                latest_model = max(model_files, key=os.path.getctime)
                print(f"Загружаем модель: {latest_model}")
                model = load_model(latest_model)
                checkpoint_path = "saved_models/last_checkpoint.txt"
                if os.path.exists(checkpoint_path):
                    with open(checkpoint_path, "r") as f:
                        checkpoint = json.load(f)
                    self.epsilon = checkpoint.get('epsilon', 1.0)
                    start_episode = checkpoint.get('last_episode', 0) + 1
                    return model, start_episode
                return model, 0
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}. Создаём новую модель.")

        model = models.Sequential([
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=optimizers.Adam(0.001))
        return model, 0

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
    batch_size = 64
    os.makedirs("saved_models", exist_ok=True)
    for e in range(agent.start_episode, episodes):
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

        if e % 5 == 0:
            model_path = f"saved_models/snake_dqn_episode_{e}.keras"
            save_model(agent.model, model_path)

            checkpoint = {
                "last_episode": e,
                "epsilon": agent.epsilon,
                "score": game.score
            }
            with open("saved_models/last_checkpoint.txt", "w") as f:
                json.dump(checkpoint, f)

            print(f"Сохранено: {model_path} | ε={agent.epsilon:.2f}")

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        print(f"Episode: {e}, Score: {game.score}, Epsilon: {agent.epsilon:.2f}, Total reward: {total_reward}")
