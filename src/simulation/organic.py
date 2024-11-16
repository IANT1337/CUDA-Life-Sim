import cupy as cp
from utils.random_points import generate_random_points_in_circle
import pygame

class Organic:
    def __init__(self, config):
        self.positions = cp.random.randint(0, config['screen_width'], (config['num_organics'], 2))
        self.spawn_interval = config.get('organics_spawn_interval', 5)
        self.spawn_count = config.get('organics_spawn_count', 50)
        self.frame_count = 0
        self.circle_center_x = config['screen_width'] // 2
        self.circle_center_y = config['screen_height'] // 2
        self.circle_radius = min(config['screen_width'], config['screen_height']) // 4

    def update(self):
        self.frame_count += 1
        if self.frame_count % self.spawn_interval == 0:
            new_organics = generate_random_points_in_circle(
                self.circle_center_x, self.circle_center_y, self.circle_radius, self.spawn_count
            )
            self.positions = cp.concatenate((self.positions, new_organics), axis=0)

    def render(self, screen):
        organic_positions_host = cp.asnumpy(self.positions)
        organic_positions_host = organic_positions_host[organic_positions_host[:, 0] != -1]
        for pos in organic_positions_host:
            pygame.draw.rect(screen, (0, 0, 255), (pos[0], pos[1], 1, 1))