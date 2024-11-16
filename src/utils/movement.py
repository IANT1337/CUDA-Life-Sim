from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float32
import math

@cuda.jit(device=True)
def calculate_new_position(current_position, target_position, speed, screen_width, screen_height, direction, rng_states=None, idx=None, vision_range=None):
    if direction == 2:  # Random wandering
        random_angle = xoroshiro128p_uniform_float32(rng_states, idx) * 2 * math.pi
        random_distance = xoroshiro128p_uniform_float32(rng_states, idx) * vision_range
        wander_target_x = current_position[0] + random_distance * math.cos(random_angle)
        wander_target_y = current_position[1] + random_distance * math.sin(random_angle)
        direction_x = wander_target_x - current_position[0]
        direction_y = wander_target_y - current_position[1]
    else:
        direction_x = target_position[0] - current_position[0]
        direction_y = target_position[1] - current_position[1]

    norm = (direction_x ** 2 + direction_y ** 2) ** 0.5
    if norm > 0:
        direction_x /= norm
        direction_y /= norm
    
    if direction == 0:  # Move away from the target
        direction_x = -direction_x
        direction_y = -direction_y
    
    # Calculate the new position
    new_x = current_position[0] + direction_x * speed
    new_y = current_position[1] + direction_y * speed

    # Check if the new position overshoots the target
    if direction != 2:  # Only check for overshooting when not wandering
        dist_to_target = ((new_x - current_position[0]) ** 2 + (new_y - current_position[1]) ** 2) ** 0.5
        if dist_to_target > norm:
            new_x = target_position[0]
            new_y = target_position[1]

    # Apply boundary conditions
    if new_x < 0 or new_x >= screen_width:
        direction_x = -direction_x
    if new_y < 0 or new_y >= screen_height:
        direction_y = -direction_y
    new_x = max(0, min(new_x, screen_width - 1))
    new_y = max(0, min(new_y, screen_height - 1))
    return new_x, new_y