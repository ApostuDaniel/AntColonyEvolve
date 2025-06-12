import copy
import functools
import logging
import sys, math
from collections import namedtuple
from pickle import FALSE

import numpy as np
import pygame

from gymnasium.spaces import Box, Tuple, Dict
from gymnasium.utils import colorize, seeding
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] In %(pathname)s:%(lineno)d:\n%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

SIM_FPS = 40  # Simulation environment FPS

SCREEN_W = 1200
SCREEN_H = 1200
BOUNDARY_SCALE = 0.02

vec2d = namedtuple('vec2d', ['x', 'y'])
arena_sizes = namedtuple('arena_sizes', ['arena_center', 'arena_radius'])

# Global parameters for agent control
TIMESTEP = 1. / SIM_FPS  # Not sure if this will be necessary, given the fixed FPS?
# TIME_LIMIT = SIM_FPS * 60   # 60 seconds
TIME_LIMIT = SIM_FPS * 10  # 30 seconds

ANT_DIM = vec2d(5, 5)
AGENT_SPEED = 20.00  # Maximum ant speed observed in data (pixels)
AGENT_ACCELERATION = 10.00
TURN_RATE = 2 * 0.3575  # Maximum turning rate observed in data (radians)

SPEED_PENALTY_BASE = 1.5 ** (1 / AGENT_SPEED)

MAX_VISION_RANGE = 90
MIDDLE_VISION_RANGE = 60
MIN_VISION_RANGE = 30

MAX_FOOD_QUANTITY = 50
FOOD_SIZE_SCALING = 0.5


DRAW_ANT_VISION = True
# DRAW_PHEROMONES = False
# OBSERVE_OTHERS = True  # Whether to include other ants in the simulation
# LOAD_ANTS = True  # Whether to load real ant data or use empty list
# LOAD_PHEROMONES = False  # Whether to load pheromone data or calculate in real-time


vision_segments = [
    # Front arc: Directly in front of the agent
    # Bright red, intense and strong
    ((-math.pi / 2, -3 * math.pi / 10), (255, 0, 0), 'forward_l1'),
    # Vivid orange, warm and energetic
    ((-3 * math.pi / 10, -math.pi / 10), (255, 165, 0), 'forward_l2'),
    # Bright yellow, cheerful and vibrant
    ((-math.pi / 10, math.pi / 10), (255, 255, 0), 'forward'),
    # Bright green, fresh and lively
    ((math.pi / 10, 3 * math.pi / 10), (0, 255, 0), 'forward_r2'),
    # Sky blue, cool and calm
    ((3 * math.pi / 10, math.pi / 2), (0, 0, 255), 'forward_r1'),
    # Remove backwards detection zones
    # # Deep indigo, mysterious and strong
    # ((-9 * math.pi / 6, -7 * math.pi / 6), (75, 0, 130), 'backward_r1'),
    # # Soft violet, dreamy and delicate
    # ((-7 * math.pi / 6, -5 * math.pi / 6), (238, 130, 238), 'backward'),
    # # Soft pink, gentle and warm
    # ((-5 * math.pi / 6, -math.pi / 2), (255, 192, 203), 'backward_l1'),
]

view_ranges = [MIN_VISION_RANGE, MIDDLE_VISION_RANGE, MAX_VISION_RANGE]


REWARD_TYPE = 'coverage'  # 'action', 'coverage', 'aggregation'
TRACK_TRAIL = 'all'  # 'all', 'fade', 'none'
MOVEMENT_THRESHOLD = 10
FADE_DURATION = 5  # seconds

##########################################
#          Assistance functions          #
##########################################

def find_bounding_box(data):
    # Separate all x and y values into slices
    all_x_values = data[[col for col in data.columns if 'x' in col]]
    all_y_values = data[[col for col in data.columns if 'y' in col]]
    # Calculating the minimum and maximum for x and y values efficiently
    min_x = all_x_values.min(axis=None)
    max_x = all_x_values.max(axis=None)
    min_y = all_y_values.min(axis=None)
    max_y = all_y_values.max(axis=None)

    return min_x, min_y, max_x, max_y


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


def apply_transform_scale(data, trans, scale):
    data[[col for col in data.columns if 'x' in col]] = data[[col for col in data.columns if 'x' in col]].transform(
        np.vectorize(lambda x: np.round((x * scale) + trans[0]))
    )
    data[[col for col in data.columns if 'y' in col]] = data[[col for col in data.columns if 'y' in col]].transform(
        np.vectorize(lambda x: np.round((x * scale) + trans[1]))
    )


def euclidean_distances(data):
    a = np.array(data)
    b = a.reshape(a.shape[0], 1, a.shape[1])
    distances = np.sqrt(np.einsum('ijk, ijk->ij', a - b, a - b))
    np.fill_diagonal(distances, np.NaN)

    return distances


def is_rectangle_in_circle(x, y, circle_center, circle_radius):
    """
    Check if a pygame.Rect is completely contained within a circle.

    Parameters:
    rect (pygame.Rect): The rectangle to check.
    circle_center (tuple): The (x, y) coordinates of the center of the circle.
    circle_radius (float): The radius of the circle.

    Returns:
    bool: True if the rectangle is completely contained within the circle, False otherwise.
    """
    rect = pygame.Rect(x - ANT_DIM.x / 2., y - ANT_DIM.y / 2.,
                       ANT_DIM.x, ANT_DIM.y)
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

        if distance > circle_radius:
            return False

    return True


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
        # self.theta_dot = 0.0
        self.trail = []

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

        self.collected_food = 0
        self.total_food = 0
        # Arena information
        self.arena_center = arena_center
        self.arena_radius = arena_radius
        self.wall_collision = False

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

    # def _detect_distances(self, detected_ants):
    #     """
    #     Identify the distance to the wall of the arena and the distance to the nearest ant
    #     in each cone of vision, scaling such that closer objects (walls or ants) are
    #     closer to 1 and farther objects (or beyond the vision range) are closer to 0.

    #     Parameters:
    #     - detected_ants (dict): A dictionary mapping segment names to lists of detected ants.
    #     """

    #     for idx, (angle_range, _) in enumerate(vision_segments):
    #         start_angle, stop_angle = angle_range
    #         segment_name = self._get_segment_name(idx)

    #         # Calculate the central angle of the vision segment
    #         central_angle = (start_angle + stop_angle) / 2
    #         # Adjust the central angle based on the agent's current orientation
    #         central_angle = (self.theta + central_angle) % (2 * math.pi)

    #         # Compute distance to the wall along the central angle
    #         raw_wall_distance = self._compute_distance_to_wall(central_angle)

    #         # Flip scaling for wall distances: Closer walls = 1, farther walls = scaled to 0
    #         if raw_wall_distance is not None and raw_wall_distance <= self.vision_range:
    #             wall_distance = max(1.0 - (raw_wall_distance / self.vision_range), 0.0)
    #         else:
    #             wall_distance = 0.0  # Default for no wall detection (out of range)

    #         self.wall_distances[segment_name] = wall_distance

    #         # Find the minimum distance to the nearest ant in this segment
    #         ants_in_segment = detected_ants.get(segment_name, [])
    #         if ants_in_segment:
    #             distances = [
    #                 math.sqrt((ant[0] - self.pos.x) ** 2 + (ant[1] - self.pos.y) ** 2)
    #                 for ant in ants_in_segment
    #             ]
    #             raw_ant_distance = min(distances)
    #         else:
    #             raw_ant_distance = None  # Default for no ants in range

    #         # Flip scaling for ant distances: Closer ants = 1, farther ants = scaled to 0
    #         if raw_ant_distance is not None and raw_ant_distance <= self.vision_range:
    #             ant_distance = max(1.0 - (raw_ant_distance / self.vision_range), 0.0)
    #         else:
    #             ant_distance = 0.0  # Default for no detection or out of range

    #         self.nearest_ant_distances[segment_name] = ant_distance

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
                raw_food_distance, food_angle = min(
                    ((math.hypot(food[0] - self.pos.x, food[1] - self.pos.y), food[2])
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

    def _get_fitting_vision_range_index(self, distance: float, vision_ranges: list):
        for i in range(len(vision_ranges)):
            if distance <= vision_ranges[i]:
                return i
        return None

    def _detect_nearby_food(self, food_sources):
        """
        Detects food sources within vision range and categorizes them into vision segments.
        Marks them if they have come in contact with the ant.

        Parameters:
        - food_sources (list of tuples): The (x, y, quantity) positions of food sources and their quantity.

        Returns:
        - dict: Maps segment names to lists of food positions within that segment.

         Assigns:
            - collected_food int: food intersecting ants body
        """
        # Pre-calculate common values
        vision_range_squared = self.vision_range ** 2
        view_ranges_squared = [view_range ** 2 for view_range in view_ranges]

        # Initialize detected_food with empty lists using dict comprehension
        detected_food = {segment_idx: [] for segment_idx in range(len(vision_segments))}
        # Initialize density_areas to count number of ants in area
        density_areas = {segment_idx: [{'food': 0} for _ in range(len(view_ranges))]
                         for segment_idx in range(len(vision_segments))}

        # Pre-calculate segment boundaries once
        segment_boundaries = []
        for idx, (angle_range, _, _) in enumerate(vision_segments):
            start = (self.theta + angle_range[0]) % (2 * math.pi)
            stop = (self.theta + angle_range[1]) % (2 * math.pi)
            segment_boundaries.append((start, stop, idx))

        # Process each food
        pos_x, pos_y = self.pos.x, self.pos.y
        self.collected_food = 0
        for i, (food_x, food_y, quantity, was_eaten) in enumerate(food_sources):
            # Use faster distance calculation without sqrt for initial check
            dx = food_x - pos_x
            dy = food_y - pos_y
            distance_squared = dx * dx + dy * dy
            food_size = int(max(2, quantity * FOOD_SIZE_SCALING))
            eating_range_squared = ((ANT_DIM.x + food_size) / 2) ** 2
            if distance_squared <= eating_range_squared and was_eaten == False:
                #Mark food as eaten
                self.collected_food += quantity
                food_sources[i] = (food_x, food_y, quantity, True)

            fitting_view_range_idx = self._get_fitting_vision_range_index(distance_squared, view_ranges_squared)

            if distance_squared <= vision_range_squared:
                # Only calculate angle for ants within range
                angle_to_food = math.atan2(dy, dx) % (2 * math.pi)

                # Find matching segment
                for start_angle, stop_angle, segment_idx in segment_boundaries:
                    if start_angle < stop_angle:
                        if start_angle <= angle_to_food <= stop_angle:
                            detected_food[segment_idx].append((food_x, food_y, angle_to_food))
                            density_areas[segment_idx][fitting_view_range_idx]['food'] += quantity
                            break
                    elif angle_to_food >= start_angle or angle_to_food <= stop_angle:
                        # Angle wraps around 2π boundary
                        detected_food[segment_idx].append((food_x, food_y, angle_to_food))
                        density_areas[segment_idx][fitting_view_range_idx]['food'] += quantity
                        break

        return detected_food, density_areas

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
        self.wall_collision = not is_rectangle_in_circle(desired_pos[0], desired_pos[1], self.arena_center, self.arena_radius)
        if not self.wall_collision:
            self.pos = vec2d(desired_pos[0], desired_pos[1])

    def set_action(self, action):
        acceleration, turn_left, turn_right = action

        self.desired_speed = self.speed
        # self.desired_turn_speed = 0

        # # Accelerate if direction > 0.5 and decelerate otherwise, emulates FORWARD and BACKWARD movement
        # if direction < 0.5:
        #     self.desired_speed = max(-AGENT_SPEED, self.desired_speed - acceleration * AGENT_ACCELERATION)
        # elif direction > 0.5:
        #     self.desired_speed = min(AGENT_SPEED, self.desired_speed + acceleration * AGENT_ACCELERATION)

        self.desired_speed += acceleration * AGENT_ACCELERATION
        self.desired_speed = np.clip(self.desired_speed, -AGENT_SPEED, AGENT_SPEED)

        # Another movement variant(without acceleration)
        # if direction < 0.5:
        #     self.desired_speed = -(AGENT_SPEED * acceleration)
        # elif direction > 0.5:
        #     self.desired_speed = AGENT_SPEED * acceleration

        # rotation = -TURN_RATE * turn_left if turn_left > turn_right else TURN_RATE * turn_right if turn_right > turn_left else 0

        # If turn_left is negative, it means turn right
        rotation = (turn_right - turn_left) * TURN_RATE
        self.theta += rotation

        # If no turn, reset the turn timers
        if turn_left == turn_right:
            self.turning_time_left = 0.0
            self.turning_time_right = 0.0

        return [acceleration, turn_left, turn_right]

    def get_obs(self, others=None):
        """
        Generate the observation vector for the neural network, incorporating
        composite scores that encode both wall and ant distances for each vision segment.

        Parameters:
        - others (list, optional): A list of other ant agents to detect.

        Returns:
        - list: The observation vector containing normalized positions, velocities,
                vision segment data, and pheromone levels.
        """
        if others is not None:
            detected_food, density_areas = self._detect_nearby_food(others)
            self.density_areas = self._normalize_food_density(density_areas, MAX_FOOD_QUANTITY)
            self._detect_distances(detected_food)

        # Get the agent's position in polar coordinates
        r_norm, theta_norm = self.get_polar_position()

        # Initialize the observation vector with basic state information
        result = [
            r_norm,  # Normalized radial position
            theta_norm,  # Normalized angular position
            self.speed / AGENT_SPEED,  # Current speed
            self.theta,  # Current orientation
        ]

        # add scaled nearest wall distances per vision segment
        result.extend([self.wall_distances[i] for i in range(len(self.wall_distances))])
        # add nearest food distances per vision segment
        result.extend([self.nearest_food_distances[i]  for i in range(len(self.nearest_food_distances))])
        # # add angle to nearest food per vision segment
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
        # self.theta_dot = self.desired_turn_speed

        # self._turn()
        self._move()


def speed_penalty_eq(speed):
    """
    Exponential speed punishment
    :param speed: agents current abs speed
    :return: how much the reward should diminish
    """
    return (SPEED_PENALTY_BASE ** int(speed)) - 1


class AntColonyEnv(ParallelEnv):
    metadata = {
        'name': 'ant-colony-v0',
        'render_modes': ['human', 'rgb_array'],
        'render_fps': SIM_FPS
    }

    def __init__(self, nr_of_ants=10, nr_of_food=20, render_mode=None):
        self.np_random, seed = self.seed(seed=69)

        self.ants = None
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

        arena_center = (int(SCREEN_W / 2.0), int(SCREEN_H / 2.0))
        arena_radius = int(min(SCREEN_W, SCREEN_H) / 2.0 - min(SCREEN_W, SCREEN_H) * BOUNDARY_SCALE)

        # circular arena
        self.ant_arena = arena_sizes( vec2d(arena_center[0], arena_center[1]), arena_radius)

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        float_max = np.finfo(np.float32).max
        float_min = np.finfo(np.float32).min
        env_obs_spaces = {
            'r_norm': Box(0.0,1.0),
            'theta_norm': Box(-1.0, 1.0),
            'speed': Box(-AGENT_SPEED, AGENT_SPEED, (1,), np.float32),
            'theta': Box(0.0, 2 * math.pi, (1,), np.float32),
            'vision_segments_wall_distances': Box(0.0, 1.0, (len(vision_segments), np.float32)),
            'vision_segments_food_distances': Box(0.0, 1.0,(len(vision_segments), np.float32)),
            'vision_segments_food_angles': Box(-1.0, 2 * math.pi, (len(vision_segments),)),
            # 'density_areas': Box(0.0, 1.0,(len(vision_segments) * len(view_ranges),)),
            # used when testing just the closest density sensors
            # 'density_areas': Box(0.0, 1.0, (len(vision_segments),)),
        }

        return Dict(env_obs_spaces)

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Box(-1.0, 1.0, (3,), dtype=np.float32)

    def seed(self, seed=None):
        return seeding.np_random(seed)

    # def _get_interval_data(self, trail, start_time, interval=False):
    #     theta = -1
    #     threshold = 5
    #     time = start_time + 1
    #     time_offset = 0
    #     distance = 0
    #     while theta < 0 and time != len(trail):
    #         try:
    #             dx, dy = trail[time] - trail[start_time]
    #         except IndexError:
    #             break
    #         if dx == 0 and dy == 0:
    #             time_offset += 1
    #         else:
    #             current_distance = np.sqrt(dx ** 2 + dy ** 2)
    #             if current_distance > threshold:
    #                 theta = math.atan2(dy, dx) % (2 * math.pi)
    #                 distance = current_distance
    #         time += 1
    #
    #     return distance, theta, (time - start_time), time_offset
    #
    # def _get_angle_from_trajectory(self, trail, start_time):
    #     _, angle, _, _ = self._get_interval_data(trail, start_time)
    #     return angle

    # def _identify_overlapping_grid_cells(
    #         self,
    #         pos: tuple,
    #         bounding_box: int,
    #         grid_size: tuple = (SCREEN_W, SCREEN_H),
    #         coarse_grid_size: tuple = (20, 20)
    # ) -> list:
    #     """
    #     Identify which cells in the discretized grid an ant overlaps with based on its position and bounding box.
    #
    #     Parameters:
    #     - x (float): X-coordinate of the ant in simulation space (0 <= x < grid_width).
    #     - y (float): Y-coordinate of the ant in simulation space (0 <= y < grid_height).
    #     - bounding_box (int): Size of the ant's bounding box in pixels (assumed square).
    #     - grid_size (tuple): Size of the simulation grid in pixels (height, width). Default is (900, 900).
    #     - coarse_grid_size (tuple): Size of the discretized grid (height, width). Default is (20, 20).
    #
    #     Returns:
    #     - List[Tuple[int, int]]: List of (row, column) indices of grid cells overlapped by the ant.
    #     """
    #     x, y = pos
    #
    #     fine_height, fine_width = grid_size
    #     coarse_height, coarse_width = coarse_grid_size
    #
    #     # Calculate discretization factors
    #     factor_y = fine_height / coarse_height
    #     factor_x = fine_width / coarse_width
    #
    #     # Calculate bounding box boundaries in simulation space
    #     half_box = bounding_box / 2
    #     x_min = max(x - half_box, 0)
    #     x_max = min(x + half_box, fine_width - 1)
    #     y_min = max(y - half_box, 0)
    #     y_max = min(y + half_box, fine_height - 1)
    #
    #     # Map simulation space boundaries to coarse grid indices
    #     row_min = int(math.floor(y_min / factor_y))
    #     row_max = int(math.floor(y_max / factor_y))
    #     col_min = int(math.floor(x_min / factor_x))
    #     col_max = int(math.floor(x_max / factor_x))
    #
    #     # Ensure indices are within coarse grid bounds
    #     row_min = max(row_min, 0)
    #     row_max = min(row_max, coarse_height - 1)
    #     col_min = max(col_min, 0)
    #     col_max = min(col_max, coarse_width - 1)
    #
    #     # Generate list of overlapping grid cells
    #     overlapping_cells = []
    #     for row in range(row_min, row_max + 1):
    #         for col in range(col_min, col_max + 1):
    #             overlapping_cells.append((row, col))
    #
    #     return overlapping_cells


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
        reward = ant.collected_food - int(ant.wall_collision) - speed_penalty_eq(ant.speed)
        ant.total_food += reward

        return reward

    def get_observations(self, others=None):
        return {a: self.ants[a].get_obs(others) for a in self.agents}

    def _track_trail(self, pos: vec2d):
        self.ant_trail.append(pos)

    def _destroy(self):
        self.ants = {}
        self.ant_trail = []
        self.food_sources = []
        self.other_ants_angles = []

        self.viewer = None
        self.state = None

        self.window = None
        self.clock = None

    def _respawn_food(self, replacement_indices=None):
        rng = np.random.default_rng()
        max_food_draw_radius = self.ant_arena.arena_radius * 0.80 - (MAX_FOOD_QUANTITY * FOOD_SIZE_SCALING) * 2

        def get_rand_food_params():
            x, y = rng.uniform(-max_food_draw_radius, max_food_draw_radius, size=2)
            quantity = int(rng.uniform(MAX_FOOD_QUANTITY - 10, MAX_FOOD_QUANTITY))
            return int(self.ant_arena.arena_center.x + x), int(self.ant_arena.arena_center.y + y), quantity, False

        if replacement_indices is not None:
            for i in replacement_indices:
                self.food_sources[i] = get_rand_food_params()
        else:
            self.food_sources = []
            for _ in range(self.nr_of_food):
                self.food_sources.append(get_rand_food_params())


    def reset(self, seed=None, options=None):
        self._destroy()

        self.t = 0  # timestep reset

        # self.arena_mask = np.zeros((50, 50), dtype=bool)
        #
        # # Create arena mask for valid cells
        # center_x = 25  # Center of 50x50 grid
        # center_y = 25
        # y, x = np.ogrid[-center_y:50 - center_y, -center_x:50 - center_x]
        # mask_radius = (min(50, 50) / 2) * 0.98  # Slightly smaller than arena
        # self.arena_mask = x * x + y * y <= mask_radius * mask_radius

        self.actions = []

        self.time_offset = 0.0

        self.agents = copy.deepcopy(self.possible_agents)

        self.ants = {}

        rng = np.random.default_rng()
        max_ant_draw_radius = self.ant_arena.arena_radius * 0.80 - ANT_DIM.x * 2
        for agent in self.agents:
            while True:
                x, y = rng.uniform(-max_ant_draw_radius, max_ant_draw_radius, size=2)
                x, y = (int(self.ant_arena.arena_center.x + x), int(self.ant_arena.arena_center.y + y))
                theta = rng.uniform(0.0, 2 * math.pi)
                if is_rectangle_in_circle(x, y, self.ant_arena.arena_center, self.ant_arena.arena_radius):
                    self.ants[agent] = Ant((x, y), theta)
                    break

        self._respawn_food()

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

        # Pygame controls and resources if render_mode == 'human'
        if self.render_mode == "human":
            self._render_frame

        eaten_food = [ i for i, (_, _, _,was_eaten) in enumerate(self.food_sources) if was_eaten]
        self._respawn_food(list(eaten_food))
        for agent, action in actions.items():
            self.ants[agent].set_action(action)
            self.ants[agent].update()

        self.t += 1

        # Get observations of other ants in the arena
        if self.food_sources is not None:
            observations = self.get_observations(self.food_sources)
        else:
            observations = self.get_observations()

        first_ant = list(self.ants.keys())[0]
        self._track_trail(self.ants[first_ant].pos)

        terminations = {a: False for a in self.agents}

        # Determine if the episode is done
        if self.t >= self.t_limit:
            truncations = {a: True for a in self.agents}
        else:
            truncations = {a: False for a in self.agents}

        # Initialize reward value
        rewards = {agent: self._reward_function(agent, action) for agent, action in actions.items()}

        # Additional information (if necessary)
        infos = {a: {
            'wall_distances': self.ants[a].wall_distances,
            'nearest_ant_distances': self.ants[a].nearest_food_distances,
            'nearest_food_angles': self.ants[a].nearest_food_angles,
        } for a in self.agents}

        return observations, rewards, terminations, truncations, infos

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

        # Draw the circular arena on the canvas
        pygame.draw.circle(
            canvas,
            (230, 230, 230),
            (int(self.ant_arena.arena_center.x), int(self.ant_arena.arena_center.y)),  # center of the arena
            int(self.ant_arena.arena_radius)  # Radius
        )

        # Draw a glow around the best performing ant
        best_ant = max(self.agents, key=lambda agent: self.ants[agent].total_food)
        best_ant = self.ants[best_ant]
        pygame.draw.circle(
            canvas,
            (225, 225, 140),
            (int(best_ant.pos.x), int(best_ant.pos.y)),
            MAX_VISION_RANGE
        )

        ### DRAW TRAILS FIRST

        # Draw ant trail
        trail_length = len(self.ant_trail)
        if TRACK_TRAIL == 'all':
            self.ant_trail_segment = self.ant_trail
        elif TRACK_TRAIL == 'fade':
            if trail_length > FADE_DURATION * SIM_FPS:
                self.ant_trail_segment = self.ant_trail[trail_length - FADE_DURATION * SIM_FPS:]
        else:
            self.ant_trail_segment = []
        for pos in self.ant_trail_segment:
            pygame.draw.rect(
                canvas,
                (180, 180, 220),
                (
                    pos.x - ANT_DIM.x / 2.,
                    pos.y - ANT_DIM.y / 2.,
                    ANT_DIM.x,
                    ANT_DIM.y
                )
            )

        ### THEN DRAW FOOD SOURCES AT THEIR CURRENT POSITIONS

        # Draw food positions and size
        if self.food_sources is not None:
            try:
                for food_source in self.food_sources:
                    # scale down the food size
                    food_size = int(max(2, food_source[2] * FOOD_SIZE_SCALING))
                    pygame.draw.rect(
                        canvas,
                        (180, 180, 180),
                        (
                            food_source[0] - food_size / 2.,
                            food_source[1] - food_size / 2.,
                            food_size,
                            food_size
                        )
                    )

            except IndexError:
                print(food_source)
                logger.error("End of time series reached for other ants.")
            except TypeError:
                print(food_source)
                logger.error("Cannot draw ant with provided coordinates.")

        if DRAW_ANT_VISION:
            font = pygame.font.SysFont('Arial', 9)
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

                    for vision_idx, view_range in enumerate(view_ranges):
                        # compute distance to the center of each area
                        draw_distance =  view_range / 2 if vision_idx <= 0 \
                            else view_ranges[vision_idx - 1] + (view_range - view_ranges[vision_idx - 1]) / 2
                        # display number of entities in each density area
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

        # Draw nearest ant distance indicators
        for agent in self.agents:
            ant = self.ants[agent]
            for segment_idx, distance in ant.nearest_food_distances.items():
                angle = ant.nearest_food_angles[segment_idx]
                if distance is not None and distance != 1.0:
                    # Calculate the central angle of the segment
                    angle_range, segment_color, _ = vision_segments[segment_idx]

                    # Calculate end point based on nearest ant distance
                    end_x = int(ant.pos.x + math.cos(angle) * distance * MAX_VISION_RANGE)
                    end_y = int(ant.pos.y + math.sin(angle) * distance * MAX_VISION_RANGE)

                    # Draw a line from the ant to the nearest ant
                    pygame.draw.line(
                        canvas,
                        segment_color,  # black color for nearest ant distance
                        (int(ant.pos.x), int(ant.pos.y)),
                        (end_x, end_y),
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
    env = AntColonyEnv(render_mode='human')

    total_reward = 0
    observations, infos = env.reset()

    manual_mode = True
    manual_action = [0.0, 0.0, 0.0]

    terminations = [False]
    truncations = [False]
    done = False
    while env.agents and not any(terminations) and not any(truncations):
        if manual_mode:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        done = True
                    if event.key == pygame.K_r:
                        env.reset()
                    if event.key == pygame.K_UP:    manual_action[0:3] = [0.5, 0.0, 0.0]
                    if event.key == pygame.K_DOWN:  manual_action[0:3] = [-0.5, 0.0, 0.0]
                    if event.key == pygame.K_LEFT:  manual_action[1], manual_action[2] = 0.7, 0.0
                    if event.key == pygame.K_RIGHT: manual_action[1], manual_action[2] = 0.0, 0.7
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
        terminations = [terminations[a] for a in terminations]
        truncations = [truncations[a] for a in truncations]

    env.close()
    print('Cumulative score:', total_reward)