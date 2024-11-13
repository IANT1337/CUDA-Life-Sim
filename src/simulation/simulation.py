import pygame
import cupy as cp
import numpy as np
from numba import cuda
from simulation.prey import Prey
from simulation.predator import Predator
from simulation.organic import Organic
from utils.random_points import generate_random_points_in_circle
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

class Simulation:
    def __init__(self, config):
        self.config = config
        self.screen_width = config['screen_width']
        self.screen_height = config['screen_height']
        self.num_preys = config['num_preys']
        self.num_predators = config['num_predators']
        self.init_pygame()
        self.init_simulation()

        self.threads_per_block = 256
        self.blocks_per_grid = (self.num_preys + self.num_predators + (self.threads_per_block - 1)) // self.threads_per_block
        self.rng_states = create_xoroshiro128p_states(self.threads_per_block * self.blocks_per_grid, seed=np.random.randint(1, 1e6))



    def init_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('Prey vs Predator Simulation')

    def init_simulation(self):
        self.prey = Prey(self.config)
        self.predator = Predator(self.config)
        self.organic = Organic(self.config)
        self.frame_count = 0
        self.organics_spawn_interval = 5
        self.organics_spawn_count = 50
        self.circle_center_x = self.screen_width // 2
        self.circle_center_y = self.screen_height // 2
        self.circle_radius = min(self.screen_width, self.screen_height) // 4

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

        self.prey.update_positions(
            self.blocks_per_grid, self.threads_per_block,
            self.prey.positions, self.predator.positions, self.prey.speeds, self.prey.vision_ranges,
            self.screen_width, self.screen_height, self.rng_states, self.organic.positions,
            self.prey.energy, self.prey.max_energy, self.prey.reproduce_flags
        )
        self.predator.update_positions(
            self.blocks_per_grid, self.threads_per_block,
            self.predator.positions, self.prey.positions, self.predator.speeds, self.predator.vision_ranges,
            self.screen_width, self.screen_height, self.rng_states, self.predator.energy,
            self.predator.max_energy, self.predator.reproduce_flags
        )
        self.organic.update()
        self.frame_count += 1

    def render(self):
        self.screen.fill((0, 0, 0))
        self.organic.render(self.screen)
        self.prey.render(self.screen)
        self.predator.render(self.screen)
        pygame.display.flip()