import numpy as np
import random
import pygame
import pickle
import os
from events import handle_pygame_events

class Agent:
    def __init__(self, env):
        self.episodes = 10000
        self.maxSteps = (env.size * env.size)
        self.learningRate = 0.1
        self.discount = 0.99
        self.epsilon = 1.0
        self.minEpsilon = 0.01
        self.epsilonDecay = 0.001
        
        self.size = env.size
        self.presentStates = env.presentStates
        self.env = env

        self.qTable = np.zeros((self.size, self.size, 2 ** len(self.presentStates), 4))


    def greedy_policy(self, state, collectedPresents):
        presentIndex = int(''.join(['1' if (i, j) in collectedPresents else '0' for (i, j) in self.presentStates]), 2)

        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 3)
        else:
            return np.argmax(self.qTable[state[0]][state[1]][presentIndex])

    def train_agent(self, screen, cellSize):
        for episode in range(self.episodes):
            state, collectedPresents = self.env.reset()
            done = False
            t = 0

            while not done and t < self.maxSteps:
                handle_pygame_events()
                action = self.greedy_policy(state, collectedPresents)
                
                nextState, nextPresents, reward, done = self.env.step(action)
                
                presentIndex = int(''.join(['1' if (i, j) in collectedPresents else '0' for (i, j) in self.env.presentStates]), 2)
                nextPresentIndex = int(''.join(['1' if (i, j) in nextPresents else '0' for (i, j) in self.env.presentStates]), 2)
                
                self.qTable[state[0]][state[1]][presentIndex][action] += self.learningRate * \
                    (
                        reward + self.discount * np.max(self.qTable[nextState[0]][nextState[1]][nextPresentIndex]) - \
                        self.qTable[state[0]][state[1]][presentIndex][action]
                    )
                
                state, collectedPresents = nextState, nextPresents
                t += 1

            self.epsilon = max(self.minEpsilon, self.epsilon * (1 - self.epsilonDecay))

            if episode % 1000 == 0:
                print(f'Episode {episode}')
                self.env.render(screen, cellSize)
                pygame.time.wait(350)
        
        self.save_table()

    def test_agent(self, screen, cellSize):
        state, collectedPresents = self.env.reset()
        done = False
        steps = 0

        while not done:
            handle_pygame_events()

            presentIndex = int(''.join(['1' if (i, j) in collectedPresents else '0' for (i, j) in self.env.presentStates]), 2)
            action = np.argmax(self.qTable[state[0]][state[1]][presentIndex])

            nextState, nextPresents, reward, done = self.env.step(action)

            self.env.render(screen, cellSize)
            pygame.time.wait(500)
            state, collectedPresents = nextState, nextPresents
            
            steps += 1
        
        return collectedPresents, steps
   
    def save_table(self):
        
        os.makedirs('training', exist_ok=True)

        with open(os.path.join('training', 'q_table.pkl'), 'wb') as f:
            pickle.dump(self.qTable, f)

    def load_table(self):

        try:
            with open(os.path.join('training', 'q_table.pkl'), 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            print("Arquivo q_table.pkl não encontrado.")
            return None