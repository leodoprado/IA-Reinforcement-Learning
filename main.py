from events import init_pygame, quit_pygame
from env import Env
from agent import Agent

if __name__ == '__main__':
    size = 10
    bombs = 8
    presents = 5
    walls = 8

    env = Env(size, bombs, presents, walls)
    agent = Agent(env)

    screen, cellSize = init_pygame(env)

    agent.train_agent(screen, cellSize)

    collectedPresents, steps = agent.test_agent(screen, cellSize)

    print(f'=============================================================')
    print(f'Recompensas Coletadas: {len(collectedPresents)}/{presents}')
    print(f'Quantidade de Passos: {steps}')
    print(f'=============================================================')

    quit_pygame()