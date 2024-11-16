import cupy as cp
from numba import cuda
import pygame
from utils.movement import calculate_new_position

class Cell:
    def __init__(self, config, cell_config):
        self.config = config  # Store the config parameter as an attribute
        self.name = cell_config['name']
        self.positions = cp.stack((
            cp.random.uniform(0, config['screen_width'], cell_config['num_cells']),
            cp.random.uniform(0, config['screen_height'], cell_config['num_cells'])
        ), axis=1)
        self.energy = cp.full(cell_config['num_cells'], cell_config['energy'])
        self.speeds = cp.array(cp.clip(
            cp.random.normal(cell_config['speed'], 2, size=cell_config['num_cells']),
            0.05, None
        ))
        self.vision_ranges = cp.array(cp.clip(
            cp.random.normal(cell_config['vision_range'], 10, size=cell_config['num_cells']),
            0, config['screen_width']
        ))
        self.reproduce_flags = cp.zeros(cell_config['num_cells'], dtype=cp.int32)
        self.max_energy = cell_config['max_energy']
        self.is_carnivore = cell_config['is_carnivore']
        self.color = tuple(cell_config['color'])  # Read the color from the config

    @staticmethod
    @cuda.jit
    def update_positions_kernel(cell_positions, target_positions, cell_speeds, cell_vision_ranges, screen_width, screen_height, rng_states, organic_positions, cell_energy, max_cell_energy, cell_reproduce_flags, is_carnivore):
        idx = cuda.grid(1)
        # Energy management constants
        cell_energy_consumption = 0.1
        cell_energy_gain = 60.0
        reproduction_threshold_cell = max_cell_energy * 0.8  # 80% threshold for cell reproduction
        
        # Update cell positions, handle energy depletion, organic consumption, and set reproduction flag
        if idx < cell_positions.shape[0]:
            if cell_positions[idx, 0] == -1:
                return
            # Deplete cell energy
            cell_energy[idx] -= cell_energy_consumption
            if cell_energy[idx] <= 0:
                # Mark cell as inactive if energy is depleted
                cell_positions[idx, 0] = -1
                cell_positions[idx, 1] = -1
                return
            # Set reproduction flag for cell if energy exceeds threshold
            if cell_energy[idx] >= reproduction_threshold_cell:
                cell_reproduce_flags[idx] = 1  # Mark for reproduction
            # Use individual cell's speed and vision range
            cell_speed = cell_speeds[idx]
            cell_vision_range = cell_vision_ranges[idx]
            fear_distance = cell_vision_range*4  # Distance to move away from danger before resuming normal behavior
            # Check for the nearest target within vision range, constrained to local pixels
            nearest_target_idx = -1
            min_dist_to_target = float('inf')
            local_pixel_range = cell_vision_range  # Define the range of local pixels to search within
            for j in range(target_positions.shape[0]):
                if abs(cell_positions[idx, 0] - target_positions[j, 0]) <= local_pixel_range and abs(cell_positions[idx, 1] - target_positions[j, 1]) <= local_pixel_range:
                    dist_to_target = (cell_positions[idx, 0] - target_positions[j, 0]) ** 2 + (cell_positions[idx, 1] - target_positions[j, 1]) ** 2
                    if dist_to_target < min_dist_to_target and dist_to_target <= cell_vision_range ** 2:
                        min_dist_to_target = dist_to_target
                        nearest_target_idx = j
            # Check for the nearest predator within vision range, constrained to local pixels
            nearest_predator_idx = -1
            min_dist_to_predator = float('inf')
            if not is_carnivore:
                for j in range(target_positions.shape[0]):
                    if abs(cell_positions[idx, 0] - target_positions[j, 0]) <= local_pixel_range and abs(cell_positions[idx, 1] - target_positions[j, 1]) <= local_pixel_range:
                        dist_to_predator = (cell_positions[idx, 0] - target_positions[j, 0]) ** 2 + (cell_positions[idx, 1] - target_positions[j, 1]) ** 2
                        if dist_to_predator < min_dist_to_predator and dist_to_predator <= fear_distance ** 2:
                            min_dist_to_predator = dist_to_predator
                            nearest_predator_idx = j
            # If a predator is nearby, move away from it
            if nearest_predator_idx != -1:
                new_x, new_y = calculate_new_position(cell_positions[idx], target_positions[nearest_predator_idx], cell_speed, screen_width, screen_height, 0, rng_states, idx, cell_vision_range)
            elif nearest_target_idx != -1:
                # If a target is nearby, move towards or away based on carnivore status
                direction = 1 if is_carnivore else 0
                new_x, new_y = calculate_new_position(cell_positions[idx], target_positions[nearest_target_idx], cell_speed, screen_width, screen_height, direction, rng_states, idx, cell_vision_range)
            elif organic_positions is not None and not is_carnivore:
                # If no target is nearby and the cell is not a carnivore, search for the nearest organic within vision range
                nearest_organic_idx = -1
                min_dist_to_organic = float('inf')
                for j in range(organic_positions.shape[0]):
                    if organic_positions[j, 0] == -1:  # Skip inactive organics
                        continue
                    dist_to_organic = (cell_positions[idx, 0] - organic_positions[j, 0]) ** 2 + (cell_positions[idx, 1] - organic_positions[j, 1]) ** 2
                    if dist_to_organic < min_dist_to_organic and dist_to_organic <= cell_vision_range ** 2:
                        min_dist_to_organic = dist_to_organic
                        nearest_organic_idx = j
                # Move toward the nearest organic if one is in range
                if nearest_organic_idx != -1:
                    new_x, new_y = calculate_new_position(cell_positions[idx], organic_positions[nearest_organic_idx], cell_speed, screen_width, screen_height, 1, rng_states, idx, cell_vision_range)
                else:
                    # Random wandering if no targets or organics are nearby
                    new_x, new_y = calculate_new_position(
                        cell_positions[idx], cell_positions[idx], cell_speed, screen_width, screen_height, 2, rng_states, idx, cell_vision_range
                    )
            else:
                # Random wandering if no targets are nearby and the cell is a carnivore
                new_x, new_y = calculate_new_position(
                    cell_positions[idx], cell_positions[idx], cell_speed, screen_width, screen_height, 2, rng_states, idx, cell_vision_range
                )
            cell_positions[idx, 0] = new_x
            cell_positions[idx, 1] = new_y
            # Consume organic if within range and the cell is not a carnivore
            if not is_carnivore and nearest_organic_idx != -1 and min_dist_to_organic < 2:
                cell_energy[idx] += cell_energy_gain
                organic_positions[nearest_organic_idx, 0] = -1
                organic_positions[nearest_organic_idx, 1] = -1
            # Consume prey if within range and the cell is a carnivore
            if is_carnivore and nearest_target_idx != -1 and min_dist_to_target < 2:
                cell_energy[idx] += cell_energy_gain
                target_positions[nearest_target_idx, 0] = -1
                target_positions[nearest_target_idx, 1] = -1
        cuda.syncthreads()

    def update_positions(self, blocks_per_grid, threads_per_block, prey_positions, predator_positions, organic_positions, rng_states):
        if self.is_carnivore:
            target_positions = prey_positions
        else:
            target_positions = predator_positions

        Cell.update_positions_kernel[blocks_per_grid, threads_per_block](
            self.positions, target_positions, self.speeds, self.vision_ranges,
            self.config['screen_width'], self.config['screen_height'], rng_states,
            organic_positions, self.energy, self.max_energy, self.reproduce_flags, self.is_carnivore
        )

    def render(self, screen):
        cell_positions_host = cp.asnumpy(self.positions)
        cell_positions_host = cell_positions_host[cell_positions_host[:, 0] != -1]
        for pos in cell_positions_host:
            pygame.draw.rect(screen, self.color, (pos[0], pos[1], 1, 1))