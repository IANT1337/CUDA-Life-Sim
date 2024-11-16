import pygame
import cupy as cp
import numpy as np
from numba.cuda.random import create_xoroshiro128p_states
from simulation.cell import Cell
from simulation.organic import Organic
from utils.random_points import generate_random_points_in_circle

class Simulation:
    def __init__(self, config):
        self.config = config
        self.screen_width = config['screen_width']
        self.screen_height = config['screen_height']
        self.init_pygame()
        self.init_simulation()

        self.threads_per_block = 256
        self.blocks_per_grid = (sum(cell_config['num_cells'] for cell_config in config['cells']) + (self.threads_per_block - 1)) // self.threads_per_block
        self.rng_states = create_xoroshiro128p_states(self.threads_per_block * self.blocks_per_grid, seed=np.random.randint(1, 1e6))

    def init_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('Cell Simulation')

    def init_simulation(self):
        self.cells = [Cell(self.config, cell_config) for cell_config in self.config['cells']]
        self.organic = Organic(self.config)

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            self.update()
            self.render()
        pygame.quit()

    def update(self):
        prey_positions = None
        predator_positions = None
        for cell in self.cells:
            if cell.is_carnivore:
                predator_positions = cell.positions
            else:
                prey_positions = cell.positions

        for cell in self.cells:
            cell.update_positions(
                self.blocks_per_grid, self.threads_per_block,
                prey_positions, predator_positions, self.organic.positions, self.rng_states
            )
        self.organic.update()

    def render(self):
        self.screen.fill((0, 0, 0))
        self.organic.render(self.screen)
        for cell in self.cells:
            cell.render(self.screen)
        pygame.display.flip()