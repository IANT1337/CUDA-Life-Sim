import cupy as cp
from numba import cuda, float32
from numba.cuda.random import xoroshiro128p_uniform_float32
import math
import pygame
from utils.movement import calculate_new_position

class Prey:
    def __init__(self, config):
        self.positions = cp.stack((
            cp.random.uniform(0, config['screen_width'], config['num_preys']),
            cp.random.uniform(0, config['screen_height'], config['num_preys'])
        ), axis=1)
        self.energy = cp.full(config['num_preys'], config['prey_energy'])
        self.speeds = cp.array(cp.clip(
            cp.random.normal(config['prey_speed'], 2, size=config['num_preys']),
            0.05, None
        ))
        self.vision_ranges = cp.array(cp.clip(
            cp.random.normal(config['prey_vision_range'], 10, size=config['num_preys']),
            0, config['screen_width']
        ))
        self.reproduce_flags = cp.zeros(config['num_preys'], dtype=cp.int32)
        self.max_energy = 1000
        pass

    @staticmethod    
    @cuda.jit
    def update_positions_kernel(prey_positions, predator_positions, prey_speeds, prey_vision_ranges, screen_width, screen_height, rng_states, organic_positions, prey_energy, max_prey_energy, prey_reproduce_flags):
        idx = cuda.grid(1)
        # Energy management constants
        prey_energy_consumption = 0.1
        prey_energy_gain = 60.0
        reproduction_threshold_prey = max_prey_energy * 0.8  # 80% threshold for prey reproduction
        # Update prey positions, handle energy depletion, organic consumption, and set reproduction flag
        if idx < prey_positions.shape[0]:
            if prey_positions[idx, 0] == -1:
                return
            # Deplete prey energy
            prey_energy[idx] -= prey_energy_consumption
            if prey_energy[idx] <= 0:
                # Mark prey as inactive if energy is depleted
                prey_positions[idx, 0] = -1
                prey_positions[idx, 1] = -1
                return
            # Set reproduction flag for prey if energy exceeds threshold
            if prey_energy[idx] >= reproduction_threshold_prey:
                prey_reproduce_flags[idx] = 1  # Mark for reproduction
            # Use individual prey's speed and vision range
            prey_speed = prey_speeds[idx]
            prey_vision_range = prey_vision_ranges[idx]
            # Check for the nearest predator within vision range, constrained to local pixels
            nearest_predator_idx = -1
            min_dist_to_predator = float('inf')
            local_pixel_range = prey_vision_range  # Define the range of local pixels to search within
            for j in range(predator_positions.shape[0]):
                if abs(prey_positions[idx, 0] - predator_positions[j, 0]) <= local_pixel_range and abs(prey_positions[idx, 1] - predator_positions[j, 1]) <= local_pixel_range:
                    dist_to_predator = (prey_positions[idx, 0] - predator_positions[j, 0]) ** 2 + (prey_positions[idx, 1] - predator_positions[j, 1]) ** 2
                    if dist_to_predator < min_dist_to_predator and dist_to_predator <= prey_vision_range ** 2:
                        min_dist_to_predator = dist_to_predator
                        nearest_predator_idx = j
            # If a predator is nearby, run away
            if nearest_predator_idx != -1:
                new_x, new_y = calculate_new_position(prey_positions[idx], predator_positions[nearest_predator_idx], prey_speed, screen_width, screen_height, 0)
            else:
                # If no predator is nearby, search for the nearest organic within vision range
                nearest_organic_idx = -1
                min_dist_to_organic = float('inf')
                for j in range(organic_positions.shape[0]):
                    if organic_positions[j, 0] == -1:  # Skip inactive organics
                        continue
                    dist_to_organic = (prey_positions[idx, 0] - organic_positions[j, 0]) ** 2 + (prey_positions[idx, 1] - organic_positions[j, 1]) ** 2
                    if dist_to_organic < min_dist_to_organic and dist_to_organic <= prey_vision_range ** 2:
                        min_dist_to_organic = dist_to_organic
                        nearest_organic_idx = j
                # Move toward the nearest organic if one is in range
                if nearest_organic_idx != -1:
                    new_x, new_y = calculate_new_position(prey_positions[idx], organic_positions[nearest_organic_idx], prey_speed, screen_width, screen_height, 1)
                else:
                    # Random wandering if no predators or organics are nearby
                    random_angle = xoroshiro128p_uniform_float32(rng_states, idx) * 2 * math.pi
                    random_distance = xoroshiro128p_uniform_float32(rng_states, idx) * prey_vision_range
                    target_x = prey_positions[idx, 0] + random_distance * math.cos(random_angle)
                    target_y = prey_positions[idx, 1] + random_distance * math.sin(random_angle)
                    target_position = cuda.local.array(2, dtype=float32)
                    target_position[0] = target_x
                    target_position[1] = target_y
                    new_x, new_y = calculate_new_position(prey_positions[idx], target_position, prey_speed, screen_width, screen_height, 1)
            
            prey_positions[idx, 0] = new_x
            prey_positions[idx, 1] = new_y
            # Consume organic if within range
            if nearest_organic_idx != -1 and min_dist_to_organic < 2:
                prey_energy[idx] += prey_energy_gain
                organic_positions[nearest_organic_idx, 0] = -1
                organic_positions[nearest_organic_idx, 1] = -1
            pass
        cuda.syncthreads()

    def update_positions(self, blocks_per_grid, threads_per_block, *args):
        Prey.update_positions_kernel[blocks_per_grid, threads_per_block](*args)

    def render(self, screen):
        prey_positions_host = cp.asnumpy(self.positions)
        prey_positions_host = prey_positions_host[prey_positions_host[:, 0] != -1]
        for pos in prey_positions_host:
            pygame.draw.rect(screen, (0, 255, 0), (pos[0], pos[1], 1, 1))