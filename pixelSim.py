import pygame
import numpy as np
import json
import cupy as cp
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import math

# Load configuration from JSON
with open('animalConfig.json') as config_file:
    config = json.load(config_file)

# Simulation parameters
screen_width = config['screen_width']
screen_height = config['screen_height']
num_preys = config['num_preys']
num_predators = config['num_predators']
num_organics = config['num_organics']  # New parameter
prey_speed = config['prey_speed']
predator_speed = config['predator_speed']
prey_vision_range = config['prey_vision_range']
predator_vision_range = config['predator_vision_range']
max_prey_energy = 1000
max_predator_energy = 1000
# Initialize organics positions
organic_positions = cp.random.randint(0, screen_width, size=(num_organics, 2))

# Settings for spawning new organics
organics_spawn_interval = 5  # Number of frames between organic spawns
organics_spawn_count = 50       # Number of organics to spawn each interval
frame_count = 0                # Counter to keep track of frames

# Define the circle parameters
circle_center_x = screen_width // 2
circle_center_y = screen_height // 2
circle_radius = min(screen_width, screen_height) // 4  # Example: quarter of the smaller dimension

# Function to generate random points within a circle
def generate_random_points_in_circle(center_x, center_y, radius, count):
    angles = cp.random.uniform(0, 2 * math.pi, count)
    radii = cp.sqrt(cp.random.uniform(0, 1, count)) * radius  # sqrt to ensure uniform distribution within circle
    x_coords = center_x + radii * cp.cos(angles)
    y_coords = center_y + radii * cp.sin(angles)
    return cp.stack((x_coords, y_coords), axis=1).astype(cp.int32)

# Initialize energy for prey and predators
initial_prey_energy = config['prey_energy']
initial_predator_energy = config['predator_energy']
prey_energy = cp.full(num_preys, initial_prey_energy)
predator_energy = cp.full(num_predators, initial_predator_energy)


# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Prey vs Predator Simulation')

# Initialize prey and predator positions on GPU with uniform float distribution
prey_positions = cp.stack((
    cp.random.uniform(0, screen_width, num_preys),
    cp.random.uniform(0, screen_height, num_preys)
), axis=1)

predator_positions = cp.stack((
    cp.random.uniform(0, screen_width, num_predators),
    cp.random.uniform(0, screen_height, num_predators)
), axis=1)

# Initialize random states for CUDA with a variable seed for randomness
threads_per_block = 1000
blocks = (max(num_preys, num_predators) + (threads_per_block - 1)) // threads_per_block
rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed=np.random.randint(1, 1e6))


# Define reproduction threshold (e.g., 80% of max energy)
reproduction_threshold_percent = 0.8
max_population = 1000  # Max limit for prey/predators to prevent excessive growth

# Initialize reproduction flags for prey and predators
prey_reproduce_flags = cp.zeros(num_preys, dtype=cp.int32)
predator_reproduce_flags = cp.zeros(num_predators, dtype=cp.int32)


# Mutation ranges for genome traits
prey_speed_mutation = 2  # Max variation 
prey_vision_mutation = 10  # Max variation 
predator_speed_mutation = 0
predator_vision_mutation = 10

# Generate individual genomes for each prey and predator
prey_speeds = np.clip(
    np.random.normal(config['prey_speed'], prey_speed_mutation, size=num_preys), 
    0.1, None  # Ensure minimum speed > 0
)
prey_vision_ranges = np.clip(
    np.random.normal(config['prey_vision_range'], prey_vision_mutation, size=num_preys),
    0, screen_width  # Ensure vision range is within screen bounds
)

predator_speeds = np.clip(
    np.random.normal(config['predator_speed'], predator_speed_mutation, size=num_predators),
    0.1, None
)
predator_vision_ranges = np.clip(
    np.random.normal(config['predator_vision_range'], predator_vision_mutation, size=num_predators),
    0, screen_width
)

# Transfer genomes to GPU
prey_speeds_gpu = cp.array(prey_speeds)
prey_vision_ranges_gpu = cp.array(prey_vision_ranges)
predator_speeds_gpu = cp.array(predator_speeds)
predator_vision_ranges_gpu = cp.array(predator_vision_ranges)

@cuda.jit
def update_positions(prey_positions, predator_positions, prey_speeds, predator_speeds, prey_vision_ranges, predator_vision_ranges, screen_width, screen_height, rng_states, organic_positions, prey_energy, predator_energy, max_prey_energy, max_predator_energy, prey_reproduce_flags, predator_reproduce_flags):
    idx = cuda.grid(1)

    # Energy management constants
    prey_energy_consumption = 0.1
    prey_energy_gain = 60.0
    predator_energy_consumption = 0.2
    predator_energy_gain = 60.0
    reproduction_threshold_prey = max_prey_energy * 0.8  # 80% threshold for prey reproduction
    reproduction_threshold_predator = max_predator_energy * 0.8  # 80% threshold for predator reproduction

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
            direction_x = prey_positions[idx, 0] - predator_positions[nearest_predator_idx, 0]
            direction_y = prey_positions[idx, 1] - predator_positions[nearest_predator_idx, 1]
            norm = (direction_x ** 2 + direction_y ** 2) ** 0.5
            if norm > 0:
                direction_x /= norm
                direction_y /= norm
            new_x = prey_positions[idx, 0] + direction_x * prey_speed
            new_y = prey_positions[idx, 1] + direction_y * prey_speed

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
                direction_x = organic_positions[nearest_organic_idx, 0] - prey_positions[idx, 0]
                direction_y = organic_positions[nearest_organic_idx, 1] - prey_positions[idx, 1]
                norm = (direction_x ** 2 + direction_y ** 2) ** 0.5
                if norm > 0:
                    direction_x /= norm
                    direction_y /= norm
                new_x = prey_positions[idx, 0] + direction_x * prey_speed
                new_y = prey_positions[idx, 1] + direction_y * prey_speed
            else:
                # Random wandering if no predators or organics are nearby
                random_angle = xoroshiro128p_uniform_float32(rng_states, idx) * 2 * np.pi
                random_distance = xoroshiro128p_uniform_float32(rng_states, idx) * prey_vision_range
                target_x = prey_positions[idx, 0] + random_distance * math.cos(random_angle)
                target_y = prey_positions[idx, 1] + random_distance * math.sin(random_angle)
                direction_x = target_x - prey_positions[idx, 0]
                direction_y = target_y - prey_positions[idx, 1]
                norm = (direction_x ** 2 + direction_y ** 2) ** 0.5
                if norm > 0:
                    direction_x /= norm
                    direction_y /= norm
                new_x = prey_positions[idx, 0] + direction_x * prey_speed
                new_y = prey_positions[idx, 1] + direction_y * prey_speed

        # Apply boundary conditions
        if new_x < 0 or new_x >= screen_width:
            direction_x = -direction_x
        if new_y < 0 or new_y >= screen_height:
            direction_y = -direction_y
        prey_positions[idx, 0] = max(0, min(new_x, screen_width - 1))
        prey_positions[idx, 1] = max(0, min(new_y, screen_height - 1))

        # Consume organic if within range
        if nearest_organic_idx != -1 and min_dist_to_organic < 2:
            prey_energy[idx] += prey_energy_gain
            organic_positions[nearest_organic_idx, 0] = -1
            organic_positions[nearest_organic_idx, 1] = -1

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
            direction_x = prey_positions[nearest_prey_idx, 0] - predator_positions[idx, 0]
            direction_y = prey_positions[nearest_prey_idx, 1] - predator_positions[idx, 1]
            norm = (direction_x ** 2 + direction_y ** 2) ** 0.5
            if norm > 0:
                direction_x /= norm
                direction_y /= norm
            new_x = predator_positions[idx, 0] + direction_x * predator_speed
            new_y = predator_positions[idx, 1] + direction_y * predator_speed
        else:
            # Random wandering if no prey in range
            random_angle = xoroshiro128p_uniform_float32(rng_states, idx) * 2 * np.pi
            random_distance = xoroshiro128p_uniform_float32(rng_states, idx) * predator_vision_range
            target_x = predator_positions[idx, 0] + random_distance * math.cos(random_angle)
            target_y = predator_positions[idx, 1] + random_distance * math.sin(random_angle)
            direction_x = target_x - predator_positions[idx, 0]
            direction_y = target_y - predator_positions[idx, 1]
            norm = (direction_x ** 2 + direction_y ** 2) ** 0.5
            if norm > 0:
                direction_x /= norm
                direction_y /= norm
            new_x = predator_positions[idx, 0] + direction_x * predator_speed
            new_y = predator_positions[idx, 1] + direction_y * predator_speed

        # Apply boundary conditions
        if new_x < 0 or new_x >= screen_width:
            direction_x = -direction_x
        if new_y < 0 or new_y >= screen_height:
            direction_y = -direction_y
        predator_positions[idx, 0] = max(0, min(new_x, screen_width - 1))
        predator_positions[idx, 1] = max(0, min(new_y, screen_height - 1))

        # Consume prey if within range
        if nearest_prey_idx != -1 and min_dist < 3:
            predator_energy[idx] += predator_energy_gain
            prey_positions[nearest_prey_idx, 0] = -1
            prey_positions[nearest_prey_idx, 1] = -1

    cuda.syncthreads()




# Main loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update positions
    threadsperblock = 256
    blockspergrid = (num_preys + num_predators + (threadsperblock - 1)) // threadsperblock
    # Update positions within the main loop
    update_positions[blockspergrid, threadsperblock](
        prey_positions, predator_positions, 
        prey_speeds_gpu, predator_speeds_gpu, 
        prey_vision_ranges_gpu, predator_vision_ranges_gpu, 
        screen_width, screen_height, rng_states,
        organic_positions, prey_energy, predator_energy, 
        max_prey_energy, max_predator_energy,
        prey_reproduce_flags, predator_reproduce_flags
    )


    # # Process prey reproduction
    # reproduce_indices = cp.where(prey_reproduce_flags == 1)[0]
    # for idx in reproduce_indices:
    #     # Generate a random nearby position for the new prey
    #     offset_x = (cp.random.rand() - 0.5) * 10
    #     offset_y = (cp.random.rand() - 0.5) * 10
    #     new_x = cp.array([max(0, min(prey_positions[idx, 0] + offset_x, screen_width - 1))], dtype=cp.int32)
    #     new_y = cp.array([max(0, min(prey_positions[idx, 1] + offset_y, screen_height - 1))], dtype=cp.int32)
        
    #     # Create new position as a 2D cupy array
    #     new_position = cp.concatenate((new_x, new_y), axis=None).reshape(1, 2)

    #     # Append new prey if population limit allows
    #     if prey_positions.shape[0]:
    #         prey_positions = cp.concatenate((prey_positions, new_position), axis=0)
    #         new_energy = (prey_energy[idx] // 2).astype(cp.int32)  # Explicitly cast to int32
    #         prey_energy[idx] = (prey_energy[idx] // 2).astype(cp.int32)  # Reduce parent energy by half as int32
    #         prey_energy = cp.concatenate((prey_energy, cp.array([new_energy])))

    # Reset prey reproduction flags
    prey_reproduce_flags[:] = 0

    # # Process predator reproduction similarly
    # reproduce_indices = cp.where(predator_reproduce_flags == 1)[0]
    # for idx in reproduce_indices:
    #     # Generate a random nearby position for the new predator
    #     offset_x = (cp.random.rand() - 0.5) * 10
    #     offset_y = (cp.random.rand() - 0.5) * 10
    #     new_x = cp.array([max(0, min(predator_positions[idx, 0] + offset_x, screen_width - 1))], dtype=cp.int32)
    #     new_y = cp.array([max(0, min(predator_positions[idx, 1] + offset_y, screen_height - 1))], dtype=cp.int32)
        
    #     # Create new position as a 2D cupy array
    #     new_position = cp.concatenate((new_x, new_y), axis=None).reshape(1, 2)

    #     # Append new predator if population limit allows
    #     if predator_positions.shape[0]:
    #         predator_positions = cp.concatenate((predator_positions, new_position), axis=0)
    #         new_energy = (predator_energy[idx] // 2).astype(cp.int32)  # Explicitly cast to int32
    #         predator_energy[idx] = (predator_energy[idx] // 2).astype(cp.int32)  # Reduce parent energy by half as int32
    #         predator_energy = cp.concatenate((predator_energy, cp.array([new_energy])))

    # Reset predator reproduction flags
    predator_reproduce_flags[:] = 0
    # Copy data from GPU to CPU for rendering
    prey_positions_host = cp.asnumpy(prey_positions)
    predator_positions_host = cp.asnumpy(predator_positions)
    organic_positions_host = cp.asnumpy(organic_positions)
    # Remove caught prey
    prey_positions_host = prey_positions_host[prey_positions_host[:, 0] != -1]
    predator_positions_host = predator_positions_host[predator_positions_host[:, 0] != -1]
    organic_positions_host = organic_positions_host[organic_positions_host[:, 0] != -1]

    # Render
    screen.fill((0, 0, 0))
    for pos in organic_positions_host:
        pygame.draw.rect(screen, (0, 0, 255), (pos[0], pos[1], 1, 1))
    for pos in prey_positions_host:
        if pos[0] != -1:  # Only draw active prey
            pygame.draw.rect(screen, (0, 255, 0), (pos[0], pos[1], 1, 1))
    for pos in predator_positions_host:
        if pos[0] != -1:  # Only draw active predators
            pygame.draw.rect(screen, (255, 0, 0), (pos[0], pos[1], 1, 1))

    pygame.display.flip()


    # Update prey positions on GPU
    prey_positions = cp.array(prey_positions_host)

    # Check if it’s time to spawn new organics
    frame_count += 1
    if frame_count % organics_spawn_interval == 0:
        # Generate new organic positions within the circle
        new_organics = generate_random_points_in_circle(circle_center_x, circle_center_y, circle_radius, organics_spawn_count)
        
        # Expand the organic_positions array to include new organics
        organic_positions = cp.concatenate((organic_positions, new_organics), axis=0)


# Quit Pygame
pygame.quit()