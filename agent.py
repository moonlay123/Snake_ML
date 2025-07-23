import pygame
import numpy as np
import random
import sys
from collections import deque
from tensorflow.keras import layers, models, optimizers
import tensorflow as tf
from tensorflow.keras.models import save_model, load_model
from snake import Snake_game
import os, json, glob
import matplotlib.pyplot as plt

class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.short_memory = deque(maxlen=10)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
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
        except Exception as episode:
            print(f"Ошибка загрузки модели: {episode}. Создаём новую модель.")

        model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(12,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(4, activation='linear')
        ])
        model.compile(loss='mse', optimizer=optimizers.Adam(0.00025))
        return model, 0

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.short_memory.append((state, action, reward, next_state, done))
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state[np.newaxis], verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size, use_short_memory=False):
        if use_short_memory:
            if len(self.short_memory) < batch_size // 2:
                return
            minibatch = random.sample(self.short_memory, batch_size // 2) + random.sample(self.memory, batch_size // 2)
        else:
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

def save_keras(agent, game, episode):
    model_path = f"saved_models/snake_dqn_episode_{episode}.keras"
    save_model(agent.model, model_path)
    checkpoint = {
        "last_episode": episode,
        "epsilon": agent.epsilon,
        "score": game.score
    }
    with open("saved_models/last_checkpoint.txt", "w") as f:
        json.dump(checkpoint, f)
    print(f"Сохранено: {model_path} | ε={agent.epsilon:.2f}")

def train_dqn():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f"{len(gpus)} Физических GPU, {len(logical_gpus)} Логических GPU")
        except RuntimeError as episode:
            print(episode)


    game = Snake_game(train_mode=True)
    agent = DQNAgent(state_shape=12, action_size=4)
    episodes = 1000
    batch_size = 64
    os.makedirs("saved_models", exist_ok=True)
    scores = []

    for episode in range(agent.start_episode, episodes):
        state = game.reset()
        total_reward = 0
        step_count = 0

        while not game.done:

            for event in pygame.event.get():
                if event.type ==pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            if episode % 5 == 0:
                game.render()

            action = agent.act(state)
            next_state, reward, done = game.step_ai(action)
            step_count += 1

            if len(agent.short_memory) > batch_size // 2:
                agent.replay(batch_size // 2, use_short_memory=True)

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward



        if episode % 5 == 0:
            save_keras(agent, game, episode)
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        print(f"Episode: {episode}, Score: {game.score}, Epsilon: {agent.epsilon:.2f}, Total reward: {total_reward}")
