import copy
import functools
import logging
import sys, math

import warnings

warnings.filterwarnings("ignore",
                        message="pkg_resources is deprecated as an API",
                        category=UserWarning,
                        module="pygame\\.pkgdata")

import numpy as np
import pygame

from gymnasium.spaces import Box, Tuple, Dict
from gymnasium.utils import colorize, seeding
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
from scipy.signal import convolve2d

from configs.arena_config import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] In %(pathname)s:%(lineno)d:\n%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


##########################################
#          Assistance functions          #
##########################################

def calculate_circle(min_x, min_y, max_x, max_y):
    """
    Calculate the circle that fits perfectly within a bounding box.

    Parameters:
    min_x (float): The minimum x value of the bounding box.
    max_x (float): The maximum x value of the bounding box.
    min_y (float): The minimum y value of the bounding box.
    max_y (float): The maximum y value of the bounding box.

    Returns:
    tuple: A tuple containing the center coordinates (x, y) and the radius of the circle.
    """
    # Calculate the center of the bounding box
    x_center = (min_x + max_x) / 2
    y_center = (min_y + max_y) / 2

    # Calculate the radius of the circle
    radius = min(max_x - min_x, max_y - min_y) / 2

    return ((x_center, y_center), radius)


def circle_transformation(circle_a, circle_b):
    """
    Calculate the transformation from one circle to another.

    Parameters:
    circle_a (tuple): A tuple (x_a, y_a, r_a) representing Circle A's center and radius.
    circle_b (tuple): A tuple (x_b, y_b, r_b) representing Circle B's center and radius.

    Returns:
    tuple: A tuple containing the translation vector (dx, dy) and the scaling factor.
    """
    scale_factor = 0.99
    (x_a, y_a), r_a = circle_a
    (x_b, y_b), r_b = circle_b

    # Scaling
    scale = r_b / r_a

    x_a *= scale * scale_factor
    y_a *= scale * scale_factor

    # Translation vector
    dx = x_b - x_a
    dy = y_b - y_a

    return (dx, dy), scale * scale_factor

def compute_distance(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2)

def is_rectangle_in_circle(x, y, length, circle_center, circle_radius, fully_contained=True):
    """
    Check if a pygame.Rect is completely or partially contained within a circle.

    Parameters:
    rect (pygame.Rect): The rectangle to check.
    circle_center (tuple): The (x, y) coordinates of the center of the circle.
    circle_radius (float): The radius of the circle.

    Returns:
    bool: if fully_contained = True, return True if rect fully contained within circle
          if fully_contained = False, return True if rect partially contained within circle.
    """
    rect = pygame.Rect(x - length / 2., y - length / 2.,
                       length, length)
    rect_corners = [
        (rect.left, rect.top),  # Top-left
        (rect.left, rect.bottom),  # Bottom-left
        (rect.right, rect.top),  # Top-right
        (rect.right, rect.bottom)  # Bottom-right
    ]

    for x, y in rect_corners:
        dx = x - circle_center[0]
        dy = y - circle_center[1]
        distance = math.sqrt(dx ** 2 + dy ** 2)

        if fully_contained and distance > circle_radius:
            return False
        elif not fully_contained and distance <= circle_radius:
            return True

    # fully_contained == True => all corners have been checked and False was not returned, all corners in circle
    # fully_contained == False => all corners have been checked and True was not returned, no corners in circle
    return fully_contained


def to_polar_coordinates(pos, arena_center, arena_radius, norm=True):
    """
    Converts an (x, y) position into normalized polar coordinates (r, θ).

    Parameters:
    - pos: tuple, (x, y) position of the agent.
    - arena_center: tuple, (x, y) center of the arena.
    - arena_radius: float, radius of the circular arena.

    Returns:
    - r_normalized: float, radial distance normalized to [0, 1].
    - theta_normalized: float, angle in radians normalized to [-1, 1].
    """
    # Calculate raw polar coordinates
    dx = pos[0] - arena_center[0]
    dy = pos[1] - arena_center[1]
    r = np.sqrt(dx ** 2 + dy ** 2)  # Radial distance
    theta = np.arctan2(dy, dx)  # Angle in radians

    if norm is True:
        # Normalize radial distance to [0, 1]
        r_normalized = r / arena_radius

        # Normalize theta to [-1, 1] (tanh-friendly)
        theta_normalized = theta / np.pi  # θ in [-π, π] becomes [-1, 1]

        return r_normalized, theta_normalized
    else:
        return r, theta


def from_polar_coordinates(polar_coords, arena_center, arena_radius, norm=True):
    """
    Converts normalized polar coordinates (r, θ) back to Cartesian coordinates (x, y).

    Parameters:
    - polar_coords: tuple, (r, θ) in normalized polar coordinates
    - arena_center: tuple, (x, y) center of the arena
    - arena_radius: float, radius of the circular arena
    - norm: boolean, True if input is normalized (default)

    Returns:
    - tuple: (x, y) Cartesian coordinates
    """
    # Unpack polar coordinates
    r_norm, theta_norm = polar_coords

    if norm:
        # Denormalize coordinates
        r = r_norm * arena_radius  # Convert [0,1] back to [0,arena_radius]
        theta = theta_norm * np.pi  # Convert [-1,1] back to [-π, π]
    else:
        r, theta = r_norm, theta_norm

    # Convert polar to Cartesian coordinates
    x = arena_center[0] + r * np.cos(theta)
    y = arena_center[1] + r * np.sin(theta)

    return x, y


##########################################
#            Helper class for efficient distance detection            #
##########################################
class SpatialGrid:
    def __init__(self, cell_size, width, height):
        self.cell_size = cell_size
        self.width = width
        self.height = height
        self.nx = int(np.ceil(width / cell_size))
        self.ny = int(np.ceil(height / cell_size))
        self.grid = [[[] for _ in range(self.ny)] for _ in range(self.nx)]

    def _get_cell_index(self, x, y):
        return min(int(x / self.cell_size), self.nx - 1), min(int(y / self.cell_size), self.ny - 1)

    def add_item(self, idx, x, y):
        i, j = self._get_cell_index(x, y)
        self.grid[i][j].append(idx)

    def remove_item(self, idx, x, y):
        i, j = self._get_cell_index(x, y)
        self.grid[i][j].remove(idx)

    def query_radius(self, x, y, radius):
        i_center, j_center = self._get_cell_index(x, y)
        cell_radius = int(np.ceil(radius / self.cell_size))

        # Get neighboring cells
        item_ids = []
        for i in range(max(0, i_center - cell_radius), min(self.nx, i_center + cell_radius + 1)):
            for j in range(max(0, j_center - cell_radius), min(self.ny, j_center + cell_radius + 1)):
                item_ids.extend(self.grid[i][j])
        return item_ids

##########################################
#            Ant Environment             #
##########################################

class Ant():
    """Agent class for the ant"""

    def __init__(
            self,
            pos,
            theta=None,
            arena_center=(SCREEN_W / 2.0, SCREEN_H / 2.0),
            arena_radius=min(SCREEN_W, SCREEN_H) / 2.0 - min(SCREEN_W, SCREEN_H) * BOUNDARY_SCALE
    ):
        self.pos = vec2d(*pos)
        self.speed = 0.0
        self.theta = theta if theta is not None else 0.0
        self.traveled_distance = 0

        # Timing counters for smooth turning
        self.turning_time_left = 0.0
        self.turning_time_right = 0.0

        self.max_turn_duration = 0.2  # Time to reach full turn rate in seconds

        self.vision_range = MAX_VISION_RANGE

        # Initialize distance attributes for each vision segment
        self.wall_distances = {}
        self.nearest_food_distances = {}
        self.nearest_food_angles = {}
        # areas delimited by vision segments and vision ranges (portion of a pie slice)
        self.density_areas = {}
        # idx for food_sources that the ant can handle_food
        self.food_in_eating_range = []

        self.carried_food = 0
        self.current_food = 0
        self.current_energy = ANT_LIFE
        self.in_nest = False

        self.total_reward = 0
        # Arena information
        self.arena_center = arena_center
        self.arena_radius = arena_radius
        self.wall_collision = False

        # Pheromone sensing per segment
        self.pheromone_sensors = np.zeros((len(vision_segments), PHEROMONE_TYPES))  # 3 segments x 3 types
        self.pheromone_deposit_type = None  # Current deposit type

    def _compute_distance_to_wall(self, angle):
        """
        Compute the normalized distance from the ant's current position to the arena wall
        along a specified angle, scaled between 0 (very close) and 1 (beyond VISION_RANGE).

        Parameters:
        - angle (float): The angle (in radians) along which to compute the distance.

        Returns:
        - float: Normalized distance to the wall (0.0 to 1.0).
                Returns 0.0 if the wall is at zero distance,
                1.0 if the wall is beyond VISION_RANGE.
        """
        x0, y0 = self.pos.x, self.pos.y
        Cx, Cy = self.arena_center
        R = self.arena_radius

        dx = math.cos(angle)
        dy = math.sin(angle)

        # Quadratic equation coefficients: t^2 + 2bt + c = 0
        a = 1
        b = (dx * (x0 - Cx) + dy * (y0 - Cy))
        c = (x0 - Cx) ** 2 + (y0 - Cy) ** 2 - R ** 2

        discriminant = b ** 2 - c
        if discriminant < 0:
            # No intersection; should not happen if the ant is inside the arena
            return 1.0  # Treat as beyond VISION_RANGE

        sqrt_discriminant = math.sqrt(discriminant)
        t1 = -b + sqrt_discriminant
        t2 = -b - sqrt_discriminant

        # We need the positive t value
        t = t1 if t1 >= 0 else t2

        if t < 0:
            # Both intersections are behind the ant
            return 1.0  # Treat as beyond VISION_RANGE

        # Scale the distance: 0.0 (very close) to 1.0 (beyond VISION_RANGE)
        if t <= 0:
            scaled_distance = 0.0  # Extremely close to the wall
        elif t >= self.vision_range:
            scaled_distance = 1.0  # Wall is beyond VISION_RANGE
        else:
            scaled_distance = t / self.vision_range

        return scaled_distance


    def _detect_distances(self, detected_food):
        """
        Detect distances to walls and ants.
        Returns distances as positive values for ants and negative values for walls, prioritizing the closest object.
        """
        for idx, (angle_range, _, _) in enumerate(vision_segments):
            # Calculate central angle once
            central_angle = ((self.theta + sum(angle_range) / 2) % (2 * math.pi))

            # Initialize scaled distances
            food_distance_scaled = 1.0 # outside of vision range
            food_angle = -1 # outside allowed angle range(0, 2 * math.pi), to show nothing is detected

            # Get wall distance
            # compute_distance_to_wall already returns the distance scaled
            wall_distance_scaled = self._compute_distance_to_wall(central_angle)

            # Get ant distance
            food_in_segment = detected_food.get(idx, [])
            if food_in_segment:
                # Use generator expression instead of list comprehension for efficiency
                raw_food_distance, food_angle, food_radius = min(
                    ((math.hypot(food[0] - self.pos.x, food[1] - self.pos.y) - food[3], food[2], food[3]) # distance to food center minus food radius
                     for food in food_in_segment),
                    key=lambda x: x[0],
                    default=None
                )

                if raw_food_distance is not None and raw_food_distance <= self.vision_range:
                    food_distance_scaled = raw_food_distance / self.vision_range

            # Store all scores at once
            self.wall_distances[idx] = wall_distance_scaled
            self.nearest_food_distances[idx] = food_distance_scaled
            self.nearest_food_angles[idx] = food_angle

    def _normalize_food_density(self, food_density_areas: dict, max_food_size: int, max_value: float = 1.0):
        """
        Scales the number of detected food in each density area using an exponential function,
        ensuring that the scaled value reaches approximately max_value when the maximum food size is detected.

        Parameters:
        - food_density_areas (dict):
            A dictionary mapping segment names and vision ranges to food quantity detected.

        - max_food_size (int):
            Max quantity for food resource.

        - max_value (float, optional):
            The maximum value that the scaled output can reach for each segment.
            Default is 1.0.

        Assigns:
        - food_density_areas (dict): density scaled between 0.0 and 1.0
        """
        # Constant k to ensure the scaled value reaches ~99% of max_value at adjusted_fraction = 1
        k = 4.60517  # k = -ln(0.01)

        for segment_idx in range(len(vision_segments)):
            for range_idx in range(len(view_ranges)):
                food_quantity = food_density_areas[segment_idx][range_idx]["food"]

                # Normalize food_quantity by max_food_size
                fraction_detected = food_quantity / max_food_size if max_food_size > 0 else 0.0

                # Adjust fraction_detected so that food_quantity larger or equal to food size corresponds to an adjusted fraction of 1
                adjusted_fraction = min(fraction_detected, 1.0)

                # Apply exponential scaling
                scaled_value = max_value * (1 - math.exp(-k * adjusted_fraction))

                food_density_areas[segment_idx][range_idx]["food"] = scaled_value

        return food_density_areas

    def _get_fitting_vision_range_index(self, distance_squared: float, vision_ranges: list, food_size):
        for i in range(len(vision_ranges)):
            if distance_squared <= (vision_ranges[i] + food_size) ** 2:
                return i
        return None

    def _detect_nest(self, nest):
        dx = nest.pos.x - self.pos.x
        dy = nest.pos.y - self.pos.y
        angle_to_nest = math.atan2(dy, dx) % (2 * math.pi)
        distance = math.sqrt(dx ** 2 + dy ** 2)
        ant_in_nest = nest.is_ant_in_nest(self)

        return distance, angle_to_nest, ant_in_nest

    def _get_segment_boundaries(self):
        segment_boundaries = []
        for idx, (angle_range, _, _) in enumerate(vision_segments):
            start = (self.theta + angle_range[0]) % (2 * math.pi)
            stop = (self.theta + angle_range[1]) % (2 * math.pi)
            segment_boundaries.append((start, stop, idx))
        return segment_boundaries

    def _detect_nearby_food(self, food_sources, food_grid: SpatialGrid):
        """
        Detects food sources within vision range and categorizes them into vision segments.

        Parameters:
        - food_sources (list of tuples): The (x, y, quantity) positions of food sources and their quantity.

        Returns:
        - dict: Maps segment names to lists of food positions within that segment.

         Assigns:
            - collected_food int: food intersecting ants body
        """
        # Pre-calculate common values
        pos_x, pos_y = self.pos.x, self.pos.y

        # Query grid instead of scanning all food
        if food_grid:
            food_ids = food_grid.query_radius(pos_x, pos_y, self.vision_range)
        else:
            food_ids = range(len(food_sources))

        # Initialize detected_food with empty lists using dict comprehension
        detected_food = {segment_idx: [] for segment_idx in range(len(vision_segments))}
        # Initialize density_areas to count number of ants in area
        density_areas = {segment_idx: [{'food': 0} for _ in range(len(view_ranges))]
                         for segment_idx in range(len(vision_segments))}
        # Store food that can be eaten
        food_in_eating_range = []
        # Pre-calculate segment boundaries once
        segment_boundaries = self._get_segment_boundaries()

        # Process each food

        for i in food_ids:
            (food_x, food_y, quantity) = food_sources[i]
            # Use faster distance calculation without sqrt for initial check
            dx = food_x - pos_x
            dy = food_y - pos_y
            distance_squared = dx * dx + dy * dy
            food_size = int(max(2, quantity * FOOD_SIZE_SCALING))
            food_radius = food_size / 2
            eating_range_squared = (ANT_DIM.x + food_radius) ** 2
            if distance_squared <= eating_range_squared:
                #Mark food as eaten
                food_in_eating_range.append(i)

            fitting_view_range_idx = self._get_fitting_vision_range_index(distance_squared, view_ranges, food_radius)

            if distance_squared <= (self.vision_range + food_radius) ** 2:
                # Only calculate angle for ants within range
                angle_to_food = math.atan2(dy, dx) % (2 * math.pi)
                # normalized angle
                norm_angle_to_food = angle_to_food / (2 * math.pi)

                # Find matching segment
                for start_angle, stop_angle, segment_idx in segment_boundaries:
                    if start_angle < stop_angle:
                        if start_angle <= angle_to_food <= stop_angle:
                            detected_food[segment_idx].append((food_x, food_y, norm_angle_to_food, food_radius))
                            density_areas[segment_idx][fitting_view_range_idx]['food'] += quantity
                            break
                    elif angle_to_food >= start_angle or angle_to_food <= stop_angle:
                        # Angle wraps around 2π boundary
                        detected_food[segment_idx].append((food_x, food_y, norm_angle_to_food, food_radius))
                        density_areas[segment_idx][fitting_view_range_idx]['food'] += quantity
                        break

        return detected_food, density_areas, food_in_eating_range


    def _detect_pheromones(self, pheromone_grid, cell_size):
        """Detect pheromones in each vision segment"""
        # Initialize sensor array: 2 ranges x 3 segments x 3 pheromone types
        sensors = np.zeros((len(pheromone_ranges), len(vision_segments), PHEROMONE_TYPES))

        if pheromone_grid is None:
            return sensors

        # Get grid position
        grid_x = int(self.pos.x / cell_size)
        grid_y = int(self.pos.y / cell_size)

        # Calculate search range
        max_range = PHEROMONE_DETECTION_MAX_RANGE
        grid_range = int(max_range / cell_size) + 1

        # Get grid bounds
        min_i = max(0, grid_y - grid_range)
        max_i = min(pheromone_grid.shape[0], grid_y + grid_range + 1)
        min_j = max(0, grid_x - grid_range)
        max_j = min(pheromone_grid.shape[1], grid_x + grid_range + 1)

        # Pre-calculate segment boundaries once
        segment_boundaries = self._get_segment_boundaries()
        ranges_squared = [r**2 for r in pheromone_ranges]
        max_range_squared = max(ranges_squared)
        dim_squared = ANT_DIM.x ** 2
        for i in range(min_i, max_i):
            for j in range(min_j, max_j):
                # Calculate world position of cell center
                cell_x = (j + 0.5) * cell_size
                cell_y = (i + 0.5) * cell_size

                if not np.sum(pheromone_grid[i, j]) >= 0.1:
                    continue

                # Calculate vector from ant to cell
                dx = cell_x - self.pos.x
                dy = cell_y - self.pos.y
                # distance = math.sqrt(dx ** 2 + dy ** 2)
                distance_squared = dx ** 2 + dy ** 2

                # Skip if beyond vision range
                # skip if directly under ant
                if distance_squared > max_range_squared or distance_squared < dim_squared:
                    continue

                # angle relative to ant
                angle = math.atan2(dy, dx) % (2 * math.pi)

                # angle relative to ant's heading
                relative_angle = (angle - self.theta + math.pi) % (2 * math.pi) - math.pi

                # Early skip if outside vision cone
                if not (vision_segments[0][0][0] <= relative_angle <= vision_segments[-1][0][-1]):
                    continue

                # Determine which range bin the cell belongs to
                range_idx = None
                for idx, r in enumerate(ranges_squared):
                    if distance_squared <= r:
                        range_idx = idx
                        break

                # Skip if not in any range bin
                if range_idx is None:
                    continue

                for start_angle, stop_angle, segment_idx in segment_boundaries:
                    if start_angle < stop_angle:
                        if start_angle <= angle <= stop_angle:
                            sensors[range_idx, segment_idx] += pheromone_grid[i, j]
                            break
                    elif angle >= start_angle or angle <= stop_angle:
                        # Angle wraps around 2π boundary
                        sensors[range_idx, segment_idx] += pheromone_grid[i, j]
                        break

        return sensors

    def get_polar_position(self):
        """
        Returns the agent's position in normalized polar coordinates.

        Returns:
        - r: float, radial distance normalized to [0, 1].
        - theta: float, angle normalized to [-1, 1].
        """
        r, theta = to_polar_coordinates(
            pos=(self.pos.x, self.pos.y),
            arena_center=self.arena_center,
            arena_radius=self.arena_radius,
            norm=True
        )
        return r, theta

    # def _turn(self):
    #     self.theta += self.theta_dot
    #     self.theta = self.theta % (2 * math.pi)

    def _move(self):
        """
        Move an agent from its current position (x, y) according to desired_speed
        and angle theta using matrix multiplication.
        """
        # Calculate the desired direction of travel (rotate to angle theta)
        direction = np.array([np.cos(self.theta), np.sin(self.theta)]) * self.desired_speed
        # Set the desired position based on direction and speed relative to timestep
        desired_pos = np.add(np.array(self.pos), direction)
        # If leaving the circle, push agent back into circle.
        self.wall_collision = not is_rectangle_in_circle(desired_pos[0], desired_pos[1], ANT_DIM.x, self.arena_center, self.arena_radius)
        if not self.wall_collision:
            self.traveled_distance += compute_distance(self.pos.x, self.pos.y, desired_pos[0], desired_pos[1])
            self.pos = vec2d(desired_pos[0], desired_pos[1])


    def handle_food(self, food_sources, nest, agent, decrease_food_source=True):
        self.current_food = 0
        if self.in_nest and self.carried_food > 0:
            nest.register_food(self.carried_food, agent)
            self.carried_food = 0
        else:
            for food_idx in self.food_in_eating_range:
                if self.carried_food >= ANT_CARRY_CAPACITY: break
                x, y, quantity = food_sources[food_idx]
                initial_quantity = quantity
                # if self.current_energy < ANT_LIFE:
                #     extracted = min(quantity, ANT_LIFE - self.current_energy)
                #     self.current_energy += extracted
                #     self.current_food += extracted
                #     quantity -= extracted
                if quantity > 0 and self.carried_food < ANT_CARRY_CAPACITY:
                    extracted = min(quantity, ANT_CARRY_CAPACITY - self.carried_food)
                    self.carried_food += extracted
                    self.current_food += extracted
                    quantity -= extracted
                if not decrease_food_source:
                    quantity = initial_quantity
                food_sources[food_idx] = (x, y, quantity)


    def handle_health(self, turn_left, turn_right):
        '''
        Remove from the agents food, then health at every timestep
        Refresh energy if ant in nest
        :return:
        '''
        if self.in_nest and self.current_energy < ANT_LIFE and self.carried_food > 0:
            self.current_energy = ANT_LIFE
            return

        health_lost = speed_penalty_square(self.speed) + rotation_penalty_square(turn_right, turn_left) + int(self.wall_collision) # decrease extra if colliding with wall
        # if self.carried_food >= health_lost:
        #     self.carried_food -= health_lost
        # else:
        #     self.current_energy -= (health_lost - self.carried_food)
        #     self.carried_food = 0

        if self.current_energy >= health_lost:
            self.current_energy -= health_lost
        else:
            self.carried_food -= health_lost
            if self.carried_food < 0: #subtract the remaining health
                self.current_energy += self.carried_food
                self.carried_food = 0

    def place_pheromone(self, pheromone_action, pheromone_grid, grid_width, grid_height, cell_size):
        grid_x = int(self.pos.x / cell_size)
        grid_y = int(self.pos.y / cell_size)

        if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
            # Determine deposit type based on action value
            if pheromone_action < -0.5:
                # Type 1 (red)
                pheromone_grid[grid_y, grid_x, 0] = PHEROMONE_DEPOSIT
                self.pheromone_deposit_type = 0
            elif pheromone_action < 0:
                # Type 2 (green)
                pheromone_grid[grid_y, grid_x, 1] = PHEROMONE_DEPOSIT
                self.pheromone_deposit_type = 1
            elif pheromone_action < 0.5:
                # No deposit
                self.pheromone_deposit_type = None
            else:
                # Type 3 (blue)
                pheromone_grid[grid_y, grid_x, 2] = PHEROMONE_DEPOSIT
                self.pheromone_deposit_type = 2

    def set_action(self, action):
        acceleration, turn_left, turn_right = action

        self.desired_speed = self.speed
        # self.desired_turn_speed = 0

        self.desired_speed += acceleration * AGENT_ACCELERATION
        self.desired_speed = np.clip(self.desired_speed, -AGENT_SPEED, AGENT_SPEED)

        # If turn_left is negative, it means turn right
        rotation = (turn_right - turn_left) * TURN_RATE
        self.theta += rotation

        # If no turn, reset the turn timers
        if turn_left == turn_right:
            self.turning_time_left = 0.0
            self.turning_time_right = 0.0

        return [acceleration, turn_left, turn_right]

    def get_obs(self, food_sources=None, nest=None, pheromone_grid=None, cell_size=None, time=0, food_grid=None):
        """
        Generate the observation vector for the neural network, incorporating
        composite scores that encode both wall and ant distances for each vision segment.

        Parameters:
        - others (list, optional): A list of other ant agents to detect.

        Returns:
        - list: The observation vector containing normalized positions, velocities,
                vision segment data, and pheromone levels.
        """
        if food_sources is not None:
            detected_food, density_areas, food_in_eating_range = self._detect_nearby_food(food_sources, food_grid)
            self.food_in_eating_range = food_in_eating_range
            self.density_areas = self._normalize_food_density(density_areas, MAX_FOOD_QUANTITY)
            self._detect_distances(detected_food)

        # Add pheromone sensor values (flattened: 3 segments * 3 types = 9 values)
        # if pheromone_grid is not None and cell_size is not None:
        #     self.pheromone_sensors = self._detect_pheromones(pheromone_grid, cell_size)
        #     # clip value to max detectable
        #     numpy.clip(self.pheromone_sensors, 0, MAX_DETECTABLE_PHEROMONE_CONCENTRATION)
        #     #normalize concentrations
        #     self.pheromone_sensors /= MAX_DETECTABLE_PHEROMONE_CONCENTRATION

        if nest is not None:
            (nest_distance, nest_angle, ant_in_nest) = self._detect_nest(nest)
        else:
            nest_distance, nest_angle, ant_in_nest = 1., -1., False

        self.in_nest = ant_in_nest

        # Get the agent's position in polar coordinates
        r_norm, theta_norm = self.get_polar_position()

        # Initialize the observation vector with basic state information
        result = [
            # r_norm,  # Normalized radial position
            # theta_norm,  # Normalized angular position
            self.speed / AGENT_SPEED,  # Normalized current speed
            self.theta / (2 * math.pi),  # Normalized current orientation
            time / (TIME_LIMIT * SIM_FPS),  # HEARTBEAT
            self.traveled_distance / MAX_TRAVEL_PER_EPISODE, # normalized travel distance
            self.current_energy / ANT_LIFE, # normalized health
            self.carried_food / ANT_CARRY_CAPACITY, # normalize food carry
            int(ant_in_nest), # 1 if ant is in nest, 0 otherwise
            int(self.current_food > 0 or (self.carried_food > 0 and ant_in_nest)), # reward sensor
            int(int(self.wall_collision)), # penalty sensor
            nest_distance / (self.arena_radius * 2), # normalize to arena diameter, the maximum distance between ant and nest
            nest_angle / (2 * math.pi), # normalized angle to nest
        ]

        # result.extend(self.pheromone_sensors.flatten())

        # add scaled nearest wall distances per vision segment
        result.extend([self.wall_distances[i] for i in range(len(self.wall_distances))])
        # add nearest food distances per vision segment
        result.extend([self.nearest_food_distances[i]  for i in range(len(self.nearest_food_distances))])
        # add normalized angle to nearest food per vision segment
        result.extend([self.nearest_food_angles[i]  for i in range(len(self.nearest_food_angles))])

        # Include the food density for each food detection area

        # for segment_idx in range(len(vision_segments)):
        #     # result.append(density_areas[segment_idx][0]['food'])
        #     for range_idx in range(len(view_ranges)):
        #         result.append(density_areas[segment_idx][range_idx]['food'])

        return result

    def update(self, noise=0.0):
        self.pos = vec2d(
            self.pos.x + np.random.randn() * noise,
            self.pos.y - np.random.randn() * noise
        )
        self.theta += (np.random.randn() * noise)
        self.theta = self.theta % (2 * math.pi)

        self.speed = self.desired_speed

        # self._turn()
        self._move()

class Nest:
    """Class for the ant nest"""
    def __init__(self, pos, radius=10):
        self.pos = vec2d(*pos)
        self.radius = radius
        self.collected_food = 0.
        self.food_register = {}

    def is_ant_in_nest(self, ant: Ant):
        return is_rectangle_in_circle(ant.pos.x, ant.pos.y, ANT_DIM.x, self.pos, self.radius, False)

    def register_food(self, quantity, agent):
        self.food_register[agent] = quantity
        self.collected_food += quantity

    def clear_food_register(self):
        self.food_register = {}

def speed_penalty_exponential(speed):
    """
    Exponential speed punishment
    :param speed: agents current abs speed
    :return: how much the reward should diminish
    """
    return (SPEED_PENALTY_BASE ** int(speed)) - 1

def speed_penalty_square(speed):
    """
    Square speed punishment
    :param speed: agents current abs speed
    :return: how much the reward should diminish
    """
    return ((abs(speed) ** 2) / (AGENT_SPEED ** 2)) / 4

def rotation_penalty_square(turn_right, turn_left):
    """
    Penalty based on how much the agent changes direction
    :param turn_right: how much the agent wants to turn right.(-1, 1)
    :param turn_left: how much the agent wants to turn left.(-1, 1)
    :return: how much the reward should diminish
    """
    # if turn_right 1 and turn_left -1, max value in abs would be 2
    return (abs(turn_right - turn_left) ** 2) / (2 ** 2)



class AntColonyEnv(ParallelEnv):
    metadata = {
        'name': 'ant-colony-v0',
        'render_modes': ['human', 'rgb_array'],
        'render_fps': SIM_FPS
    }

    def __init__(self, nr_of_ants=10, nr_of_food=20, render_mode=None):
        self.np_random, seed = self.seed(seed=69)

        self.ants = None
        self.nest = None
        self.possible_agents = [f"ant_{i}" for i in range(nr_of_ants)]
        self.nr_of_ants = nr_of_ants
        self.nr_of_food = nr_of_food

        self.food_sources = None
        self.viewer = None
        self.state = None
        self.noise = 0

        self.t = 0
        self.t_limit = TIME_LIMIT

        self.time_offset = None
        self.actions = []

        assert render_mode is None or render_mode in type(self).metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None
        self.terminations = {}
        self.truncations = {}

        arena_center = (int(SCREEN_W / 2.0), int(SCREEN_H / 2.0))
        arena_radius = int(min(SCREEN_W, SCREEN_H) / 2.0 - min(SCREEN_W, SCREEN_H) * BOUNDARY_SCALE)

        # circular arena
        self.ant_arena = arena_sizes( vec2d(arena_center[0], arena_center[1]), arena_radius)

        # Pheromone grid setup
        self.cell_size = ANT_DIM.x  # pheromone is ant size
        self.grid_width = int(SCREEN_W / self.cell_size)
        self.grid_height = int(SCREEN_H / self.cell_size)
        self.pheromone_grid = np.zeros((self.grid_height, self.grid_width, PHEROMONE_TYPES),
                                       dtype=np.float32)

        # Add spatial grid
        self.food_grid = None
        # add food radius to ensure that center of food is in square
        self.food_cell_size = MAX_VISION_RANGE + (MAX_FOOD_QUANTITY * FOOD_SIZE_SCALING / 2)

        # Arena mask to stop pheromones diffusing outside of arena
        # self.arena_mask = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        # center_x = int(SCREEN_W / 2)
        # center_y = int(SCREEN_H / 2)
        # y, x = np.ogrid[-center_y:self.grid_height - center_y, -center_x:self.grid_width - center_x]
        # mask_radius = (min(self.grid_height, self.grid_width) / 2) * 0.98  # Slightly smaller than arena
        # self.arena_mask = x * x + y * y <= mask_radius * mask_radius

        # Diffusion kernel
        self.diffusion_kernel = np.array([
            [0.01, 0.01, 0.01],
            [0.01, .92, 0.01],  # Center keeps 92%, neighbors get 1% each
            [0.01, 0.01, 0.01]
        ])

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        float_max = np.finfo(np.float32).max
        float_min = np.finfo(np.float32).min
        env_obs_spaces = {
            # 'r_norm': Box(0.0,1.0),
            # 'theta_norm': Box(-1.0, 1.0),
            'speed': Box(-1., 1., (1,), np.float32),
            'theta': Box(0.0, 1., (1,), np.float32),
            "heartbeat": Box(0., 1., (1,), np.float32),
            "travel_distance": Box(0., 1., (1,), np.float32),
            "current_energy": Box(0., 1., (1, ) , np.float32),
            "carried_food": Box(0., 1., (1, ) , np.float32),
            "ant_in_nest": Box(0., 1., (1, ) , np.float32),
            "reward_sensor": Box(0., 1., (1,), np.float32),
            "penalty_sensor": Box(0., 1., (1,), np.float32),
            "distance_to_nest": Box(0.0, 1., (1, ), np.float32),
            "angle_to_nest": Box(0.0, 1., (1,), np.float32),
            # 'pheromone_sensors': Box(0.0, 1.0, (len(pheromone_ranges) * len(vision_segments) * PHEROMONE_TYPES,)),
            'vision_segments_wall_distances': Box(0.0, 1.0, (len(vision_segments), np.float32)),
            'vision_segments_food_distances': Box(0.0, 1.0,(len(vision_segments), np.float32)),
            'vision_segments_food_angles': Box(-1.0, 1., (len(vision_segments),)),
            # 'density_areas': Box(0.0, 1.0,(len(vision_segments) * len(view_ranges),)),
            # used when testing just the closest density sensors
            # 'density_areas': Box(0.0, 1.0, (len(vision_segments),)),
        }

        return Dict(env_obs_spaces)

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Box(-1.0, 1.0, (4,), dtype=np.float32)

    def seed(self, seed=None):
        return seeding.np_random(seed)

    def _angle_difference(self, angle1, angle2):
        """
        Calculate the smallest difference between two angles (radians).
        """
        diff = (angle2 - angle1) % (2 * math.pi)
        if diff > math.pi:
            diff -= 2 * math.pi
        return diff

    def _reward_function(self, agent, actions=None):
        """
        Calculate the reward of the ant based on whether their closest view range detects food
        """
        ant = self.ants[agent]
        # negative reward if agent died
        if self.terminations[agent]:
            ant.total_reward -= 100
            return -100

        # turn_left, turn_right = actions[1:3]
        nest_reward = self.nest.food_register.get(agent, 0)
        reward = nest_reward * 5 + ant.current_food * 0.8
        ant.total_reward += reward

        return reward

    def get_observations(self, others=None):
        return {a: self.ants[a].get_obs(others, self.nest, self.pheromone_grid, self.cell_size, self.t, self.food_grid) for a in self.agents}

    def _destroy(self):
        self.ants = {}
        self.food_sources = []
        self.other_ants_angles = []

        self.viewer = None
        self.state = None

        self.window = None
        self.clock = None

    def _respawn_food(self, respawn_eaten_food=True):
        rng = np.random.default_rng()
        max_food_draw_radius = (self.ant_arena.arena_radius * 2) * 0.5 - (MAX_FOOD_QUANTITY * FOOD_SIZE_SCALING) * 2
        min_food_draw_radius = (self.ant_arena.arena_radius * 2) * 0.3
        # max_food_draw_radius = self.nest.radius * 8 - (MAX_FOOD_QUANTITY * FOOD_SIZE_SCALING) * 2

        def get_rand_food_params():
            while True:
                x, y = rng.uniform(-max_food_draw_radius, max_food_draw_radius, size=2)
                quantity = int(rng.uniform(MAX_FOOD_QUANTITY - 10, MAX_FOOD_QUANTITY))
                # return int(self.ant_arena.arena_center.x + x), int(self.ant_arena.arena_center.y + y), quantity
                x, y = int(self.nest.pos.x + x), int(self.nest.pos.y + y)
                if (compute_distance(x, y, self.nest.pos.x, self.nest.pos.y) > min_food_draw_radius and
                        is_rectangle_in_circle(x, y, quantity * FOOD_SIZE_SCALING / 2, self.ant_arena.arena_center, self.ant_arena.arena_radius)):
                    return x, y, quantity

        def place_food_mirroring_nest():
            """Returns food parameters with a food location that mirrors the position of the nest over arena origin."""
            nest_r, nest_theta = to_polar_coordinates(self.nest.pos, self.ant_arena.arena_center, self.ant_arena.arena_radius)
            food_theta = (nest_theta + 1) - 2 if nest_theta + 1 > 1 else nest_theta + 1
            quantity = int(rng.uniform(MAX_FOOD_QUANTITY - 10, MAX_FOOD_QUANTITY))
            food_x, food_y = from_polar_coordinates((nest_r, food_theta), self.ant_arena.arena_center, self.ant_arena.arena_radius, True)
            return food_x, food_y, quantity

        if self.food_sources is None or len(self.food_sources) == 0:
            self.food_sources = []
            for i in range(self.nr_of_food):
                x, y, quantity = get_rand_food_params()
                if self.nr_of_food == 1:
                    x, y, quantity = place_food_mirroring_nest()
                self.food_sources.append((x, y, quantity))
                # Add to food grid
                self.food_grid.add_item(i, x, y)
        elif respawn_eaten_food:
            for i in range(len(self.food_sources)):
                x, y, quantity = self.food_sources[i]
                if quantity <= 0:
                    # Remove consumed food from grid
                    self.food_grid.remove_item(i, x, y)
                    x, y, quantity = get_rand_food_params()
                    self.food_sources[i] = (x, y, quantity)
                    # Add to food grid
                    self.food_grid.add_item(i, x, y)

    def _apply_arena_mask(self):
        """Zero out pheromones outside the circular arena"""

        # Apply mask to all pheromone channels
        for i in range(PHEROMONE_TYPES):
            self.pheromone_grid[:, :, i] *= self.arena_mask

    def _diffuse_pheromones(self):
        """Apply diffusion to pheromone grid using convolution"""
        # Create a copy to avoid modifying during convolution
        diffused_grid = np.zeros_like(self.pheromone_grid)

        # Apply convolution to each pheromone type separately
        for i in range(PHEROMONE_TYPES):
            channel = self.pheromone_grid[:, :, i]
            # Apply convolution with boundary='fill' to prevent edge artifacts
            diffused = convolve2d(
                channel,
                self.diffusion_kernel,
                mode='same',
                boundary='fill',
                fillvalue=0
            )
            diffused_grid[:, :, i] = diffused

        # Update the pheromone grid
        self.pheromone_grid = diffused_grid
        # # Optional: Apply arena mask to zero out pheromones outside arena
        # self._apply_arena_mask()


    def reset(self, seed=None, options=None):
        self._destroy()

        self.t = 0  # timestep reset

        self.actions = []

        self.time_offset = 0.0

        nr_of_agents = options.get('nr_of_agents', len(self.possible_agents)) if options else len(self.possible_agents)
        self.agents = copy.deepcopy(self.possible_agents[:nr_of_agents])

        self.ants = {}
        rng = np.random.default_rng()

        # spawn nest
        # max_nest_draw_radius = self.ant_arena.arena_radius * 0.20 - NEST_RADIUS * 2
        # draw in center
        max_nest_draw_radius = self.ant_arena.arena_radius * 0.1
        min_nest_draw_radius = self.ant_arena.arena_radius * 0.0
        while True:
            x, y = rng.uniform(-max_nest_draw_radius, max_nest_draw_radius, size=2)
            x, y = (int(self.ant_arena.arena_center.x + x), int(self.ant_arena.arena_center.y + y))
            if is_rectangle_in_circle(x, y, int(NEST_RADIUS), self.ant_arena.arena_center, self.ant_arena.arena_radius)\
                    and compute_distance(x, y, self.ant_arena.arena_center.x, self.ant_arena.arena_center.y) > min_nest_draw_radius:
                self.nest = Nest((x, y), int(NEST_RADIUS))
                break

        # spawn ants in nest
        max_ant_draw_radius = self.nest.radius
        for agent in self.agents:
            while True:
                x, y = rng.uniform(-max_ant_draw_radius, max_ant_draw_radius, size=2)
                x, y = (int(self.nest.pos.x + x), int(self.nest.pos.y + y))
                theta = rng.uniform(0.0, 2 * math.pi)
                if is_rectangle_in_circle(x, y, ANT_DIM.x, self.nest.pos, self.nest.radius):
                    self.ants[agent] = Ant((x, y), theta)
                    break

        # spawn ants randomly
        # max_ant_draw_radius = self.ant_arena.arena_radius * 0.80 - ANT_DIM.x * 2
        # for agent in self.agents:
        #     while True:
        #         x, y = rng.uniform(-max_ant_draw_radius, max_ant_draw_radius, size=2)
        #         x, y = (int(self.ant_arena.arena_center.x + x), int(self.ant_arena.arena_center.y + y))
        #         theta = rng.uniform(0.0, 2 * math.pi)
        #         if is_rectangle_in_circle(x, y, self.ant_arena.arena_center, self.ant_arena.arena_radius):
        #             self.ants[agent] = Ant((x, y), theta)
        #             break

        # Initialize spatial grid
        self.food_grid = SpatialGrid(
            cell_size=self.food_cell_size,
            width=SCREEN_W,
            height=SCREEN_H
        )

        # spawn food
        self._respawn_food(SHOULD_FOOD_RESPAWN)

        self.pheromone_grid = np.zeros((self.grid_height, self.grid_width, PHEROMONE_TYPES),
                                       dtype=np.float32)

        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}

        if self.food_sources is not None:
            observations = self.get_observations(self.food_sources)
        else:
            observations = self.get_observations()

        infos = {a: {
            'wall_distances': self.ants[a].wall_distances,
            'nearest_food_distances': self.ants[a].nearest_food_distances,
            'nearest_food_angles': self.ants[a].nearest_food_angles,
        } for a in self.agents}

        if self.render_mode == 'human':
            self._render_frame

        return observations, infos

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def step(self, actions):
        """
        Each step, take the given actions and return observations, rewards, terminations, truncations
        and any other additional information if necessary.
        """

        # Apply pheromone decay
        self.pheromone_grid *= PHEROMONE_DECAY

        # Apply diffusion every 15 time steps
        # if self.t % 15 == 0:
        #     self._diffuse_pheromones()

        # Pygame controls and resources if render_mode == 'human'
        if self.render_mode == "human":
            self._render_frame

        # don't respawn food for pheromone training
        self._respawn_food(SHOULD_FOOD_RESPAWN)
        for agent, action in actions.items():
            self.ants[agent].set_action(action[:3])
            self.ants[agent].update()
            self.ants[agent].place_pheromone(action[3], self.pheromone_grid, self.grid_width, self.grid_height, self.cell_size)
            self.ants[agent].handle_health(action[1], action[2])
            self.ants[agent].handle_food(self.food_sources, self.nest, agent, SHOULD_FOOD_DECREASE)
        self.t += 1

        # terminate ants that have run out of energy
        self.terminations = {a: self.ants[a].current_energy <= 0 for a in self.agents}

        # Determine if the episode is done
        self.truncations = {a: self.t >= self.t_limit for a in self.agents}

        # Get observations of other ants in the arena
        if self.food_sources is not None:
            observations = self.get_observations(self.food_sources)
        else:
            observations = self.get_observations()

        # Initialize reward value
        rewards = {agent: self._reward_function(agent, action) for agent, action in actions.items()}
        # clear the food used to compute rewards this episode
        self.nest.clear_food_register()

        # Additional information (if necessary)
        infos = {a: {
            'wall_distances': self.ants[a].wall_distances,
            'nearest_ant_distances': self.ants[a].nearest_food_distances,
            'nearest_food_angles': self.ants[a].nearest_food_angles,
        } for a in self.agents}

        self.agents = [agent for agent in self.agents if not self.terminations[agent]]

        return observations, rewards, self.terminations, self.truncations, infos

    @staticmethod
    def _draw_text(canvas: pygame.Surface, font: pygame.font, x: int, y: int, text: str, color: tuple[int, int, int]):
        text = font.render(text, True, color)
        text_rect = text.get_rect()
        text_rect.center = (x,y)
        canvas.blit(text, text_rect)

    def render(self, mode=None):
        if mode is not None:
            self.render_mode = mode
        # if self.render_mode == 'rgb_array':
        return self._render_frame

    def draw_arena(self, canvas: pygame.Surface):
        # Draw the circular arena on the canvas
        pygame.draw.circle(
            canvas,
            (230, 230, 230),
            (int(self.ant_arena.arena_center.x), int(self.ant_arena.arena_center.y)),  # center of the arena
            int(self.ant_arena.arena_radius)  # Radius
        )

    def highlight_best_ant(self, canvas: pygame.Surface):
        # Draw a glow around the best performing ant
        best_ant = max(self.agents, key=lambda agent: self.ants[agent].total_reward)
        best_ant = self.ants[best_ant]
        pygame.draw.circle(
            canvas,
            (225, 225, 140),
            (int(best_ant.pos.x), int(best_ant.pos.y)),
            MAX_VISION_RANGE
        )

    def draw_pheromone_grid(self, canvas: pygame.Surface, render_pheromones=True, render_ant_grid_vision=True):
        # Calculate arena bounding square
        arena_center_x, arena_center_y = self.ant_arena.arena_center
        arena_radius = self.ant_arena.arena_radius
        bounding_left = arena_center_x - arena_radius
        bounding_right = arena_center_x + arena_radius
        bounding_top = arena_center_y - arena_radius
        bounding_bottom = arena_center_y + arena_radius

        # Calculate grid indices within bounding square
        grid_left = max(0, int(bounding_left / self.cell_size))
        grid_right = min(self.grid_width, int(bounding_right / self.cell_size) + 1)
        grid_top = max(0, int(bounding_top / self.cell_size))
        grid_bottom = min(self.grid_height, int(bounding_bottom / self.cell_size) + 1)

        def draw_grid_lines():
            # Draw grid lines only within the bounding square
            grid_color = (180, 180, 180)  # Semi-transparent gray

            # Vertical lines
            for i in range(grid_left, grid_right + 1):
                x = int(i * self.cell_size)
                pygame.draw.line(
                    canvas,
                    grid_color,
                    (x, bounding_top),
                    (x, bounding_bottom),
                    1
                )

            # Horizontal lines
            for j in range(grid_top, grid_bottom + 1):
                y = int(j * self.cell_size)
                pygame.draw.line(
                    canvas,
                    grid_color,
                    (bounding_left, y),
                    (bounding_right, y),
                    1
                )

        def draw_pheromones():
            # Draw pheromone concentrations only within bounding square
            for i in range(grid_top, grid_bottom):
                for j in range(grid_left, grid_right):
                    pheromones = self.pheromone_grid[i, j]
                    total_pheromone = np.sum(pheromones)

                    if total_pheromone > 0.01:  # Only draw significant concentrations
                        x = int((j + 0.5) * self.cell_size)
                        y = int((i + 0.5) * self.cell_size)

                        # Skip if outside arena circle
                        dx = x - arena_center_x
                        dy = y - arena_center_y
                        distance = math.sqrt(dx ** 2 + dy ** 2)
                        if distance > arena_radius:
                            continue

                        # Create blended color based on pheromone types
                        r = min(255, int(pheromones[0] * 255))
                        g = min(255, int(pheromones[1] * 255))
                        b = min(255, int(pheromones[2] * 255))
                        alpha = min(255, int(total_pheromone * 100))

                        # Draw concentration circle
                        pygame.draw.circle(
                            canvas,
                            (r, g, b, alpha),
                            (x, y),
                            int(self.cell_size / 2)
                        )

        def draw_ant_grid_vision():
            for agent in self.agents:
                ant = self.ants[agent]
                # Get grid position
                grid_x = int(ant.pos.x / self.cell_size)
                grid_y = int(ant.pos.y / self.cell_size)

                # Calculate search range
                max_range = PHEROMONE_DETECTION_MAX_RANGE
                grid_range = int(max_range / self.cell_size) + 1

                # Get grid bounds
                min_i = max(0, grid_y - grid_range)
                max_i = min(self.pheromone_grid.shape[0], grid_y + grid_range + 1)
                min_j = max(0, grid_x - grid_range)
                max_j = min(self.pheromone_grid.shape[1], grid_x + grid_range + 1)

                # Pre-calculate segment boundaries once
                segment_boundaries = ant._get_segment_boundaries()

                for i in range(min_i, max_i):
                    for j in range(min_j, max_j):
                        # Calculate world position of cell center
                        cell_x = (j + 0.5) * self.cell_size
                        cell_y = (i + 0.5) * self.cell_size

                        r, g, b = 255, 0, 0
                        if np.sum(self.pheromone_grid[i, j]) >= 0.1:
                            r, g, b = 0, 255, 0

                        # Calculate vector from ant to cell
                        dx = cell_x - ant.pos.x
                        dy = cell_y - ant.pos.y
                        distance = math.sqrt(dx ** 2 + dy ** 2)

                        # Skip if beyond vision range
                        # skip if directly under ant
                        if distance > max_range or distance < ANT_DIM.x:
                            continue

                        # Determine which range bin the cell belongs to
                        range_idx = None
                        for idx, pheromone_range in enumerate(pheromone_ranges):
                            if distance <= pheromone_range:
                                range_idx = idx
                                break

                        # Skip if not in any range bin
                        if range_idx is None:
                            continue

                        # diferentiate colours between ranges by blue
                        b = (200 * range_idx) % 256
                        colour = (r, g, b)

                        angle = math.atan2(dy, dx) % (2 * math.pi)

                        for start_angle, stop_angle, segment_idx in segment_boundaries:
                            if start_angle < stop_angle:
                                if start_angle <= angle <= stop_angle:
                                    pygame.draw.rect(
                                        canvas,
                                        colour,
                                        (
                                            j * self.cell_size,
                                            i * self.cell_size,
                                            ANT_DIM.x,
                                            ANT_DIM.x
                                        )
                                    )
                                    break
                            elif angle >= start_angle or angle <= stop_angle:
                                # Angle wraps around 2π boundary
                                pygame.draw.rect(
                                    canvas,
                                    colour,
                                    (
                                        j * self.cell_size,
                                        i * self.cell_size,
                                        ANT_DIM.x,
                                        ANT_DIM.x
                                    )
                                )
                                break

        draw_grid_lines()
        if render_ant_grid_vision:
            draw_ant_grid_vision()
        if render_pheromones:
            draw_pheromones()

    def draw_food_grid(self, canvas: pygame.Surface):
        for agent in self.agents:
            ant = self.ants[agent]
            grid_i, grid_j = self.food_grid._get_cell_index(ant.pos.x, ant.pos.y)

            cell_size = self.food_cell_size
            pygame.draw.rect(
                canvas,
                (128, 0, 128),
                (grid_i * cell_size, grid_j * cell_size, cell_size, cell_size)
            )

            i_center, j_center = self.food_grid._get_cell_index(ant.pos.x, ant.pos.y)
            cell_radius = int(np.ceil(MAX_VISION_RANGE / cell_size))

            for i in range(max(0, i_center - cell_radius), min(self.food_grid.nx, i_center + cell_radius + 1)):
                for j in range(max(0, j_center - cell_radius), min(self.food_grid.ny, j_center + cell_radius + 1)):
                    pygame.draw.rect(
                        canvas,
                        (0, 0, 255),
                        (i * cell_size, j * cell_size, cell_size, cell_size),
                        3
                    )

    def draw_nest(self, canvas: pygame.Surface, font: pygame.font):
        ## DRAW NEST
        pygame.draw.circle(
            canvas,
            (160, 110, 30),
            (int(self.nest.pos.x), int(self.nest.pos.y)),
            self.nest.radius
        )

        self._draw_text(
            canvas,
            font,
            self.nest.pos.x,
            self.nest.pos.y,
            f"{int(self.nest.collected_food)}",
            (0, 0, 0)
        )

    def draw_food(self, canvas: pygame.Surface, font: pygame.font):
        # Draw food positions and size
        if self.food_sources is not None:
            for (x, y, quantity) in self.food_sources:
                # scale down the food_size
                # food_size is the diameter
                food_size = int(max(2, quantity * FOOD_SIZE_SCALING))
                pygame.draw.circle(
                    canvas,
                    (0, 102, 0),
                    (x, y),
                    food_size / 2
                )

                self._draw_text(
                    canvas,
                    font,
                    x,
                    y,
                    f"{int(quantity)}",
                    (255, 255, 255)
                )

    def draw_ant_vision(self, canvas: pygame.Surface, font: pygame.font, draw_food_density=True):
        for segment_idx, ((angle_range_start, angle_range_end), colour, _) in enumerate(vision_segments):
            for agent in self.agents:
                ant = self.ants[agent]
                # Calculate start and stop angles, normalized to 0 to 2*pi
                start_angle = (ant.theta + angle_range_start) % (2 * math.pi)
                stop_angle = (ant.theta + angle_range_end) % (2 * math.pi)
                central_angle = (ant.theta + (angle_range_start + angle_range_end) / 2) % (2 * math.pi)

                # Draw lines delimiting the vision segments
                pygame.draw.line(
                    canvas,
                    colour,
                    (int(ant.pos.x), int(ant.pos.y)),
                    (
                        int(ant.pos.x + np.cos(start_angle) * MAX_VISION_RANGE),
                        int(ant.pos.y + np.sin(start_angle) * MAX_VISION_RANGE)
                    )
                )

                pygame.draw.line(
                    canvas,
                    colour,
                    (int(ant.pos.x), int(ant.pos.y)),
                    (
                        int(ant.pos.x + np.cos(stop_angle) * MAX_VISION_RANGE),
                        int(ant.pos.y + np.sin(stop_angle) * MAX_VISION_RANGE)
                    )
                )

                if draw_food_density:
                    for vision_idx, view_range in enumerate(view_ranges):
                        # compute distance to the center of each area
                        draw_distance = view_range / 2 if vision_idx <= 0 \
                            else view_ranges[vision_idx - 1] + (view_range - view_ranges[vision_idx - 1]) / 2
                        # display food density
                        self._draw_text(
                            canvas,
                            font,
                            int(ant.pos.x + np.cos(central_angle) * draw_distance),
                            int(ant.pos.y + np.sin(central_angle) * draw_distance),
                            f"{round(ant.density_areas[segment_idx][vision_idx]['food'], 2)}",
                            (0, 0, 0)
                        )

                        # Draw arcs delimiting the vision ranges and density areas
                        pygame.draw.arc(
                            canvas,
                            colour,
                            (int(ant.pos.x - view_range), int(ant.pos.y - view_range), view_range * 2,
                             view_range * 2),
                            # To align to pygame where angles increase clockwise, we needed to inverse the values
                            -stop_angle,
                            -start_angle
                        )

                if DRAW_PHEROMONE_VISION:
                    # draw arc for the pheromone detection range
                    for range in pheromone_ranges:
                        pygame.draw.arc(
                            canvas,
                            (255, 165, 0),# orange
                            (int(ant.pos.x - range), int(ant.pos.y - range), range * 2,
                             range * 2),
                            # To align to pygame where angles increase clockwise, we needed to inverse the values
                            -stop_angle,
                            -start_angle
                        )

    def draw_ants(self, canvas: pygame.Surface, font: pygame.font, draw_ant_vision=True):
        # Draw nearest ant distance indicators
        for agent in self.agents:
            ant = self.ants[agent]
            if draw_ant_vision:
                for segment_idx, distance in ant.nearest_food_distances.items():
                    angle = ant.nearest_food_angles[segment_idx]
                    angle = angle * (2 * math.pi) # denormalize angle, from (0, 1) interval
                    if distance is not None and distance < 1.0:
                        # Calculate the central angle of the segment
                        angle_range, segment_color, _ = vision_segments[segment_idx]

                        # Calculate end point based on nearest food distance
                        end_x = int(ant.pos.x + math.cos(angle) * distance * MAX_VISION_RANGE)
                        end_y = int(ant.pos.y + math.sin(angle) * distance * MAX_VISION_RANGE)

                        # Draw a line from the ant to the nearest food
                        pygame.draw.line(
                            canvas,
                            segment_color,  # segment color for nearest ant distance
                            (int(ant.pos.x), int(ant.pos.y)),
                            (end_x, end_y),
                            1
                        )

            if DRAW_NEST_SENSORS:
                # Draw distance indicator to nest
                distance, angle_to_nest, _ = ant._detect_nest(self.nest)
                nest_x = int(ant.pos.x + math.cos(angle_to_nest) * distance)
                nest_y = int(ant.pos.y + math.sin(angle_to_nest) * distance)
                pygame.draw.line(
                    canvas,
                    (0, 0, 0),  # black color for nearest ant distance
                    (int(ant.pos.x), int(ant.pos.y)),
                    (nest_x, nest_y),
                    1
                )

            # Draw agent last; to ensure visibility.
            pygame.draw.rect(
                canvas,
                (0, 0, 180),
                (
                    ant.pos.x - ANT_DIM.x / 2.,
                    ant.pos.y - ANT_DIM.y / 2.,
                    ANT_DIM.x,
                    ANT_DIM.y
                )
            )

            pygame.draw.line(
                canvas,
                (0, 0, 180),
                (int(ant.pos.x), int(ant.pos.y)),
                (
                    int(ant.pos.x + np.cos(ant.theta) * ANT_DIM.x * 3),
                    int(ant.pos.y + np.sin(ant.theta) * ANT_DIM.x * 3)
                )
            )

            backwards_l, backwards, backwards_r = ((ant.theta - math.pi + sign * math.pi / 3) % (2 * math.pi) for sign
                                                   in [-1, 0, 1])
            # draw current energy
            self._draw_text(
                canvas,
                font,
                int(ant.pos.x + np.cos(backwards_l) * ANT_DIM.x * 4),
                int(ant.pos.y + np.sin(backwards_l) * ANT_DIM.x * 4),
                f"{int(ant.current_energy)}",
                (0, 0, 0)
            )

            # draw current reward
            self._draw_text(
                canvas,
                font,
                int(ant.pos.x + np.cos(backwards) * ANT_DIM.x * 4),
                int(ant.pos.y + np.sin(backwards) * ANT_DIM.x * 4),
                f"{int(ant.total_reward)}",
                (0, 0, 0)
            )

            # draw current food
            self._draw_text(
                canvas,
                font,
                int(ant.pos.x + np.cos(backwards_r) * ANT_DIM.x * 4),
                int(ant.pos.y + np.sin(backwards_r) * ANT_DIM.x * 4),
                f"{int(ant.carried_food)}",
                (0, 0, 0)
            )

    @property
    def _render_frame(self):
        if self.window is None and self.render_mode == 'human':
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption('Ant Colony Evolution')
            self.window = pygame.display.set_mode(
                (SCREEN_W, SCREEN_H)
            )

        if self.clock is None and self.render_mode == 'human':
            self.clock = pygame.time.Clock()
            return

        # Fill the canvas with purple as the outer background
        canvas = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        canvas.fill((200, 190, 210))  # Outer purple background
        font = pygame.font.SysFont('Arial', 9)

        self.draw_arena(canvas)

        if HIGHLIGHT_BEST_ANT:
            self.highlight_best_ant(canvas)

        if DRAW_FOOD_GRID:
            self.draw_food_grid(canvas)

        self.draw_pheromone_grid(canvas, DRAW_PHEROMONES, DRAW_PHEROMONE_VISION)

        self.draw_nest(canvas, font)

        ### THEN DRAW FOOD SOURCES AT THEIR CURRENT POSITIONS

        self.draw_food(canvas, font)

        if DRAW_ANT_VISION:
            self.draw_ant_vision(canvas, font, DRAW_FOOD_DENSITY)

        # Draw wall distance indicators
        # for segment_idx, distance in self.ant.wall_distances.items():
        #     if distance is not None:
        #         # Calculate the central angle of the segment
        #         angle_range, _ = vision_segments[segment_idx]
        #         central_angle = (angle_range[0] + angle_range[1]) / 2
        #         central_angle = (self.ant.theta + central_angle) % (2 * math.pi)

        #         # Calculate end point based on wall distance
        #         end_x = int(self.ant.pos.x + math.cos(central_angle) * distance)
        #         end_y = int(self.ant.pos.y + math.sin(central_angle) * distance)

        #         # Draw a line from the ant to the wall
        #         pygame.draw.line(
        #             canvas,
        #             (255, 0, 0),  # Red color for wall distance
        #             (int(self.ant.pos.x), int(self.ant.pos.y)),
        #             (end_x, end_y),
        #             1
        #         )

        self.draw_ants(canvas, font, DRAW_ANT_VISION)

        if self.render_mode == 'human':
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )


if __name__ == "__main__":
    env = AntColonyEnv(render_mode='human', nr_of_ants=1, nr_of_food=5)

    while True:
        total_reward = 0
        observations, infos = env.reset()

        manual_mode = True
        manual_action = [0.0, 0.0, 0.0, -1.0]

        terminations = [False]
        truncations = [False]
        done = False
        while env.agents and not (all(terminations) or any(truncations)):
            if manual_mode:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            done = True
                        if event.key == pygame.K_r:
                            env.reset()
                        if event.key == pygame.K_UP:    manual_action[0:3] = [0.1, 0.0, 0.0]
                        if event.key == pygame.K_DOWN:  manual_action[0:3] = [-0.1, 0.0, 0.0]
                        if event.key == pygame.K_LEFT:  manual_action[1], manual_action[2] = 0.5, 0.0
                        if event.key == pygame.K_RIGHT: manual_action[1], manual_action[2] = 0.0, 0.5
                    elif event.type == pygame.KEYUP:
                        if event.key == pygame.K_UP:    manual_action[0] = 0.0
                        if event.key == pygame.K_DOWN:  manual_action[0] = 0.0
                        if event.key == pygame.K_LEFT:  manual_action[1] = 0.0
                        if event.key == pygame.K_RIGHT:  manual_action[2] = 0.0

                actions = {agent: manual_action for agent in env.agents}
            else:
                actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            if done is True:
                break
            observations, rewards, terminations, truncations, infos = env.step(actions)
            total_reward += sum(rewards.values())
            terminations = [terminations[a] for a in terminations]
            truncations = [truncations[a] for a in truncations]

        print('Cumulative score:', total_reward)
    env.close()
