import cupy as cp
from numba import cuda, float32
from numba.cuda.random import xoroshiro128p_uniform_float32
import math
import pygame
from utils.movement import calculate_new_position

from utils.movement import calculate_new_position

class Predator:
    def __init__(self, config):
        self.positions = cp.stack((
            cp.random.uniform(0, config['screen_width'], config['num_predators']),
            cp.random.uniform(0, config['screen_height'], config['num_predators'])
        ), axis=1)
        self.energy = cp.full(config['num_predators'], config['predator_energy'])
        self.speeds = cp.array(cp.clip(
            cp.random.normal(config['predator_speed'], 0, size=config['num_predators']),
            0.1, None
        ))
        self.vision_ranges = cp.array(cp.clip(
            cp.random.normal(config['predator_vision_range'], 10, size=config['num_predators']),
            0, config['screen_width']
        ))
        self.reproduce_flags = cp.zeros(config['num_predators'], dtype=cp.int32)
        self.max_energy = 1000
        pass
    
    @staticmethod
    @cuda.jit
    def update_positions_kernel(predator_positions, prey_positions, predator_speeds, predator_vision_ranges, screen_width, screen_height, rng_states, predator_energy, max_predator_energy, predator_reproduce_flags):
        idx = cuda.grid(1)
        # Energy management constants
        predator_energy_consumption = 0.2
        predator_energy_gain = 60.0
        reproduction_threshold_predator = max_predator_energy * 0.8  # 80% threshold for predator reproduction
        # Update predator positions, handle energy depletion, prey consumption, and set reproduction flag
        if idx < predator_positions.shape[0]:
            if predator_positions[idx, 0] == -1:
                return
            # Deplete predator energy
            predator_energy[idx] -= predator_energy_consumption
            if predator_energy[idx] <= 0:
                # Mark predator as inactive if energy is depleted
                predator_positions[idx, 0] = -1
                predator_positions[idx, 1] = -1
                return
            # Set reproduction flag for predator if energy exceeds threshold
            if predator_energy[idx] >= reproduction_threshold_predator:
                predator_reproduce_flags[idx] = 1  # Mark for reproduction
            # Use individual predator's speed and vision range
            predator_speed = predator_speeds[idx]
            predator_vision_range = predator_vision_ranges[idx]
            # Look for the nearest prey within vision range
            nearest_prey_idx = -1
            min_dist = float('inf')
            for j in range(prey_positions.shape[0]):
                if prey_positions[j, 0] == -1:  # Skip inactive prey
                    continue
                dist_to_prey = (predator_positions[idx, 0] - prey_positions[j, 0]) ** 2 + (predator_positions[idx, 1] - prey_positions[j, 1]) ** 2
                if dist_to_prey < min_dist and dist_to_prey <= predator_vision_range ** 2:
                    min_dist = dist_to_prey
                    nearest_prey_idx = j
            # Move towards prey if one is within range
            if nearest_prey_idx != -1:
                new_x, new_y = calculate_new_position(predator_positions[idx], prey_positions[nearest_prey_idx], predator_speed, screen_width, screen_height, 1)
            else:
                # Random wandering if no predators or organics are nearby
                random_angle = xoroshiro128p_uniform_float32(rng_states, idx) * 2 * math.pi
                random_distance = xoroshiro128p_uniform_float32(rng_states, idx) * predator_vision_range
                target_x = predator_positions[idx, 0] + random_distance * math.cos(random_angle)
                target_y = predator_positions[idx, 1] + random_distance * math.sin(random_angle)
                target_position = cuda.local.array(2, dtype=float32)
                target_position[0] = target_x
                target_position[1] = target_y
                new_x, new_y = calculate_new_position(prey_positions[idx], target_position, predator_speed, screen_width, screen_height, 1)
            predator_positions[idx, 0] = new_x
            predator_positions[idx, 1] = new_y
            # Consume prey if within range
            if nearest_prey_idx != -1 and min_dist < 3:
                predator_energy[idx] += predator_energy_gain
                prey_positions[nearest_prey_idx, 0] = -1
                prey_positions[nearest_prey_idx, 1] = -1
            pass
        cuda.syncthreads()

    def update_positions(self, blocks_per_grid, threads_per_block, *args):
        Predator.update_positions_kernel[blocks_per_grid, threads_per_block](*args)
    def render(self, screen):
        predator_positions_host = cp.asnumpy(self.positions)
        predator_positions_host = predator_positions_host[predator_positions_host[:, 0] != -1]
        for pos in predator_positions_host:
            pygame.draw.rect(screen, (255, 0, 0), (pos[0], pos[1], 1, 1))