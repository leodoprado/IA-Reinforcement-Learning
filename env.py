import os
import numpy as np
import random
import pygame
import pickle

class Env:
    def __init__(self, size=10, nBombs=3, nRewards=2, nWalls=5):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.startState = (0, 0)
        self.goalState = (size-1, size-1)
        self.nPresents = nRewards

        self.bombsStates = self.place_random(nBombs)
        self.presentStates = self.place_random(nRewards, exclude=self.bombsStates)
        self.wallStates = self.place_random(nWalls, exclude=self.bombsStates + self.presentStates)
        self.save_env()
        
        self.presentsCollected = set()

        for i, j in self.bombsStates:
            self.grid[i][j] = 1
        for i, j in self.presentStates:
            self.grid[i][j] = 2
        for i, j in self.wallStates:
            self.grid[i][j] = 3

        self.images = {
            "bomberman": pygame.image.load("images/bomberman.png"),
            "goal": pygame.image.load("images/goal.png"),
            "bomb": pygame.image.load("images/bomb.png"),
            "reward": pygame.image.load("images/reward.png"),
            "wall": pygame.image.load("images/rock.png"),
        }

    def place_random(self, num_items, exclude=[]):
        items = []
        while len(items) < num_items:
            i, j = random.randint(0, self.size-1), random.randint(0, self.size-1)
            
            if (i, j) not in items and (i, j) not in exclude and (i, j) != self.startState and (i, j) != self.goalState:
                items.append((i, j))
        
        return items
    

    def reset(self):
        self.currentState = self.startState
        self.presentsCollected = set()
        return self.currentState, tuple(self.presentsCollected)
    

    def step(self, action):
        i, j = self.currentState
        if action == 0: # CIMA
            i = max(i-1, 0)
        elif action == 1: # BAIXO
            i = min(i+1, self.size-1)
        elif action == 2: # ESQUERDA
            j = max(j-1, 0)
        elif action == 3: # DIREITA
            j = min(j+1, self.size-1)
        
        if (i, j) in self.wallStates:
            i, j = self.currentState
        
        self.currentState = (i, j)
        
        if self.currentState == self.goalState:
            if len(self.presentsCollected) == len(self.presentStates):
                reward = 10
                done = True
            else:
                reward = -3
                done = True
        elif self.currentState in self.bombsStates:
            reward = -10
            done = True
        elif self.currentState in self.presentStates and self.currentState not in self.presentsCollected:
            self.presentsCollected.add(self.currentState)
            reward = 2
            done = False
        else:
            reward = -0.1
            done = False
        
        return self.currentState, tuple(self.presentsCollected), reward, done


    def render(self, screen, cellSize=60):
        screen.fill((200, 200, 200))

        for i in range(self.size):
            for j in range(self.size):
                if (i, j) == self.currentState:
                    image = self.images["bomberman"]
                elif (i, j) == self.goalState:
                    image = self.images["goal"]
                elif self.grid[i][j] == 1:
                    image = self.images["bomb"]
                elif self.grid[i][j] == 2:
                    if (i, j) not in self.presentsCollected:
                        image = self.images["reward"]
                    else:
                        continue
                elif self.grid[i][j] == 3:
                    image = self.images["wall"]
                else:
                    continue 

                screen.blit(image, (j * cellSize, i * cellSize, cellSize, cellSize))

        pygame.display.flip()

    def save_env(self):

        gridData = {
            'bombsStates': self.bombsStates,
            'presentsStates': self.presentStates,
            'wallStates': self.wallStates
        }

        os.makedirs('training', exist_ok=True)

        with open(os.path.join('training', 'q_env.pkl'), 'wb') as f:
            pickle.dump(gridData, f)

    def load_env(self):
        try:
            with open(os.path.join('training', 'q_env.pkl'), 'rb') as f:
                gridData = pickle.load(f)
                return gridData['bombsStates'], gridData['presentsStates'], gridData['wallsStates']
        except FileNotFoundError:
            print("Arquivo q_env.pkl não encontrado.")
            return None

    def save_qtable(self, q_table):
        os.makedirs('training', exist_ok=True)

        with open(os.path.join('training', 'q_table.pkl'), 'wb') as f:
            pickle.dump(q_table, f)

    def load_qtable(self):
        try:
            with open(os.path.join('training', 'q_table.pkl'), 'rb') as f:
                q_table = pickle.load(f)
                return q_table
        except FileNotFoundError:
            print("Arquivo q_table.pkl não encontrado.")
            return None