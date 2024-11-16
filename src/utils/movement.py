from numba import cuda

@cuda.jit(device=True)
def calculate_new_position(current_position, target_position, speed, screen_width, screen_height, direction):
    direction_x = target_position[0] - current_position[0]
    direction_y = target_position[1] - current_position[1]
    norm = (direction_x ** 2 + direction_y ** 2) ** 0.5
    if norm > 0:
        direction_x /= norm
        direction_y /= norm
    
    if direction == 0:  # Move away from the target
        direction_x = -direction_x
        direction_y = -direction_y
    
    new_x = current_position[0] + direction_x * speed
    new_y = current_position[1] + direction_y * speed

    # Apply boundary conditions
    if new_x < 0 or new_x >= screen_width:
        direction_x = -direction_x
    if new_y < 0 or new_y >= screen_height:
        direction_y = -direction_y
    new_x = max(0, min(new_x, screen_width - 1))
    new_y = max(0, min(new_y, screen_height - 1))

    return new_x, new_y