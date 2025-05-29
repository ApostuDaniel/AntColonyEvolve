import logging
import sys, math
from collections import namedtuple
import numpy as np
import pygame
import random
import lzma
import os
from pathlib import Path
from typing import Dict, Any, Optional

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import colorize, seeding

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] In %(pathname)s:%(lineno)d:\n%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

SIM_FPS = 30  # Simulation environment FPS

SCREEN_W = 900
SCREEN_H = 900
BOUNDARY_SCALE = 0.02

vec2d = namedtuple('vec2d', ['x', 'y'])
arena_sizes = namedtuple('arena_sizes', ['arena_center', 'arena_radius'])

# Global parameters for agent control
TIMESTEP = 1. / SIM_FPS  # Not sure if this will be necessary, given the fixed FPS?
# TIME_LIMIT = SIM_FPS * 60   # 60 seconds
TIME_LIMIT = SIM_FPS * 30  # 60 seconds

ANT_DIM = vec2d(5, 5)
AGENT_SPEED = 120.00  # Maximum ant speed observed in data (pixels)
TURN_RATE = 0.3575  # Maximum turning rate observed in data (radians)
VISION_RANGE = 100  # No idea what is a reasonable value for this.

DRAW_ANT_VISION = True
# DRAW_PHEROMONES = False
# OBSERVE_OTHERS = True  # Whether to include other ants in the simulation
# LOAD_ANTS = True  # Whether to load real ant data or use empty list
# LOAD_PHEROMONES = False  # Whether to load pheromone data or calculate in real-time

vision_segments = [
    # Front arc: Directly in front of the agent
    # Bright red, intense and strong
    ((-math.pi / 2, -3 * math.pi / 10), (255, 0, 0)),
    # Vivid orange, warm and energetic
    ((-3 * math.pi / 10, -math.pi / 10), (255, 165, 0)),
    # Bright yellow, cheerful and vibrant
    ((-math.pi / 10, math.pi / 10), (255, 255, 0)),
    # Bright green, fresh and lively
    ((math.pi / 10, 3 * math.pi / 10), (0, 255, 0)),
    # Sky blue, cool and calm
    ((3 * math.pi / 10, math.pi / 2), (0, 0, 255)),
    # Deep indigo, mysterious and strong
    ((-9 * math.pi / 6, -7 * math.pi / 6), (75, 0, 130)),
    # Soft violet, dreamy and delicate
    ((-7 * math.pi / 6, -5 * math.pi / 6), (238, 130, 238)),
    # Soft pink, gentle and warm
    ((-5 * math.pi / 6, -math.pi / 2), (255, 192, 203)),
]

REWARD_TYPE = 'coverage'  # 'action', 'coverage', 'aggregation'
TRACK_TRAIL = 'all'  # 'all', 'fade', 'none'
MOVEMENT_THRESHOLD = 10
FADE_DURATION = 5  # seconds

##########################################
#          Assistance functions          #
##########################################


def translate_data_to_sim_space(data, arena_dim):
    data_len = len(data)
    logger.info(msg=f"Ant trail data loaded. Total records: {data_len}")
    arena_bb = find_bounding_box(data)
    origin_arena = calculate_circle(*arena_bb)

    translation, scale = circle_transformation(origin_arena, arena_dim)

    logger.info(msg=f"Processing data now. This will take a while...")
    apply_transform_scale(data, translation, scale)
    logger.info(msg=f"Finished processing.")

    logger.info(msg=f"Translation: {translation}, scale: {scale}")
    logger.info(
        msg=f"Original: ({origin_arena[0][0] + translation[0]}, {origin_arena[0][1] + translation[1]}), scale: {origin_arena[1] * scale}")
    logger.info(msg=f"Simulated: {arena_dim[0]}, scale: {arena_dim[1]}")

    return data


def load_analysis_results(results_dir: str = 'results/state_space') -> Dict[str, Any]:
    """
    Load the saved analysis results from the specified directory.

    Args:
        results_dir: Directory containing the analysis output files

    Returns:
        Dictionary containing all analysis results:
            - state_vectors: Raw state vectors
            - reduced_states: PCA-reduced state representations
            - transition_probabilities: State transition matrix
            - common_paths: Common behavioral sequences
            - behavioral_motifs: Identified behavioral patterns

    Raises:
        FileNotFoundError: If any of the required files are missing
    """
    results_path = Path(results_dir)

    try:
        # Load each component
        state_vectors = np.load(results_path / 'state_vectors.npy')
        reduced_states = np.load(results_path / 'reduced_states.npy')
        transition_probabilities = np.load(results_path / 'transition_probabilities.npy')
        common_paths = np.load(results_path / 'common_paths.npy', allow_pickle=True)
        behavioral_motifs = np.load(results_path / 'behavioral_motifs.npy', allow_pickle=True)

        # Package results
        results = {
            'state_vectors': state_vectors,
            'reduced_states': reduced_states,
            'transition_probabilities': transition_probabilities,
            'common_paths': common_paths,
            'behavioral_motifs': behavioral_motifs
        }

        # Print summary of loaded data
        print("\nLoaded analysis results:")
        print(f"State vectors shape: {state_vectors.shape}")
        print(f"Reduced states shape: {reduced_states.shape}")
        print(f"Transition matrix shape: {transition_probabilities.shape}")
        print(f"Number of common paths: {len(common_paths)}")
        print(f"Number of behavioral motifs: {len(behavioral_motifs)}")

        return results

    except FileNotFoundError as e:
        print(f"Error: Could not find required analysis files in {results_dir}")
        print(f"Make sure you have run the analysis code first: {e}")
        raise
    except Exception as e:
        print(f"Error loading analysis results: {e}")
        raise

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
        self.theta_dot = 0.0
        self.trail = []

        # Timing counters for smooth turning
        self.turning_time_left = 0.0
        self.turning_time_right = 0.0

        self.max_turn_duration = 0.2  # Time to reach full turn rate in seconds

        # Detection scalar:
        # num of ants in cone, or distance to closes ant
        self.V_f_l1 = None
        self.V_f_l2 = None
        self.V_f = None
        self.V_f_r2 = None
        self.V_f_r1 = None
        self.V_b_r = None
        self.V_b = None
        self.V_b_l = None
        self.vision_range = VISION_RANGE

        # Initialize distance attributes for each vision segment
        self.composite_scores = {}
        self.wall_distances = {}
        self.nearest_ant_distances = {}

        self.pheromone = 0.0

        # Arena information
        self.arena_center = arena_center
        self.arena_radius = arena_radius

    def _get_segment_name(self, idx):
        """
        Returns the segment name based on its index.

        Parameters:
        - idx (int): Index of the vision segment.

        Returns:
        - str: Name of the vision segment.
        """
        segment_names = [
            'forward_l1', 'forward_l2', 'forward', 'forward_r2',
            'forward_r1', 'backward_r', 'backward', 'backward_l'
        ]
        return segment_names[idx]

    def _compute_distance_to_wall(self, angle):
        """
        Compute the normalized distance from the ant's current position to the arena wall
        along a specified angle, scaled between 1 (very close) and 0 (beyond VISION_RANGE).

        Parameters:
        - angle (float): The angle (in radians) along which to compute the distance.

        Returns:
        - float: Normalized distance to the wall (1.0 to 0.0).
                Returns 1.0 if the wall is at zero distance,
                0.0 if the wall is beyond VISION_RANGE.
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
            return 0.0  # Treat as beyond VISION_RANGE

        sqrt_discriminant = math.sqrt(discriminant)
        t1 = -b + sqrt_discriminant
        t2 = -b - sqrt_discriminant

        # We need the positive t value
        t = t1 if t1 >= 0 else t2

        if t < 0:
            # Both intersections are behind the ant
            return 0.0  # Treat as beyond VISION_RANGE

        # Scale the distance: 1.0 (very close) to 0.0 (beyond VISION_RANGE)
        if t <= 0:
            scaled_distance = 1.0  # Extremely close to the wall
        elif t >= self.vision_range:
            scaled_distance = 0.0  # Wall is beyond VISION_RANGE
        else:
            scaled_distance = 1.0 - (t / self.vision_range)

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

    def _detect_distances(self, detected_ants):
        """
        Detect distances to walls and ants, and compute a hierarchical composite score for each segment.
        Returns distances as positive values for ants and negative values for walls, prioritizing the closest object.
        """
        for idx, (angle_range, _) in enumerate(vision_segments):
            segment_name = self._get_segment_name(idx)

            # Calculate central angle once
            central_angle = ((self.theta + sum(angle_range) / 2) % (2 * math.pi))

            # Initialize scaled distances
            wall_distance_scaled = 0.0
            ant_distance_scaled = (0.0,0)

            # Get wall distance
            # compute_distance_to_wall already returns the distance scaled
            # but in the opposite way, 1.0 means very close and 0.0 means very far

            raw_wall_distance = self._compute_distance_to_wall(central_angle)
            wall_distance_scaled = 1.0 - raw_wall_distance

            # Get ant distance
            ants_in_segment = detected_ants.get(segment_name, [])
            if ants_in_segment:
                # Use generator expression instead of list comprehension for efficiency
                raw_ant_distance = min(
                    ((math.hypot(ant[0] - self.pos.x, ant[1] - self.pos.y), ant[2])
                     for ant in ants_in_segment),
                    key=lambda x: x[0],
                    default=None
                )

                if raw_ant_distance is not None and raw_ant_distance[0] <= self.vision_range:
                    ant_distance_scaled = 1.0 - (raw_ant_distance[0] / self.vision_range), raw_ant_distance[1]

            # TODO: remove composite scores
            # Determine composite score based on closest object
            if ant_distance_scaled[0] > 0 and (wall_distance_scaled == 0 or
                                            raw_ant_distance[0] < raw_wall_distance):
                composite_score = ant_distance_scaled[0]  # Positive for ants
            elif wall_distance_scaled > 0:
                composite_score = -wall_distance_scaled  # Negative for walls
            else:
                composite_score = 0.0

            # Store all scores at once
            self.composite_scores[segment_name] = composite_score
            self.wall_distances[segment_name] = wall_distance_scaled
            self.nearest_ant_distances[segment_name] = ant_distance_scaled

    def _detect_vision(self, detected_ants: dict, total_colony_size: int, max_value: float = 1.0):
        """
        Scales the number of detected ants in each vision segment using an exponential function,
        ensuring that the scaled value reaches approximately max_value when 20% of the colony is detected
        in a single segment.

        Parameters:
        - detected_ants (dict):
            A dictionary mapping segment names to lists of detected ants.

        - total_colony_size (int):
            The total number of ants in the colony.

        - max_value (float, optional):
            The maximum value that the scaled output can reach for each segment.
            Default is 1.0.

        Assigns:
        - self.V_f_l1, self.V_f_l2, self.V_f, self.V_f_r2,
        self.V_f_r1, self.V_b_r, self.V_b, self.V_b_l:
            The scaled vision values for each segment.
        """
        # Constant k to ensure the scaled value reaches ~99% of max_value at adjusted_fraction = 1
        k = 4.60517  # k = -ln(0.01)

        vision_counts = [
            len(detected_ants['forward_l1']),
            len(detected_ants['forward_l2']),
            len(detected_ants['forward']),
            len(detected_ants['forward_r2']),
            len(detected_ants['forward_r1']),
            len(detected_ants['backward_r']),
            len(detected_ants['backward']),
            len(detected_ants['backward_l'])
        ]

        scaled_vision = []
        for ants_detected in vision_counts:
            # Normalize ants_detected by total_colony_size
            fraction_detected = ants_detected / total_colony_size if total_colony_size > 0 else 0.0

            # Adjust fraction_detected so that 20% of the colony corresponds to an adjusted fraction of 1
            adjusted_fraction = min(fraction_detected / 0.2, 1.0)

            # Apply exponential scaling
            scaled_value = max_value * (1 - math.exp(-k * adjusted_fraction))

            scaled_vision.append(scaled_value)

        # Assign the scaled values to the instance variables
        (
            self.V_f_l1, self.V_f_l2, self.V_f, self.V_f_r2,
            self.V_f_r1, self.V_b_r, self.V_b, self.V_b_l
        ) = scaled_vision

    def _detect_nearby_ants(self, other_ants):
        """
        Detects other ants within vision range and categorizes them into vision segments.

        Parameters:
        - other_ants (list of tuples): The (x, y) positions of other ants.

        Returns:
        - dict: Maps segment names to lists of ant positions within that segment.
        """
        # Pre-calculate common values
        vision_range_squared = self.vision_range ** 2
        segment_names = ('forward_l1', 'forward_l2', 'forward', 'forward_r2',
                         'forward_r1', 'backward_r', 'backward', 'backward_l')

        # Initialize detected_ants with empty lists using dict comprehension
        detected_ants = {name: [] for name in segment_names}

        # Pre-calculate segment boundaries once
        segment_boundaries = []
        for idx, (angle_range, _) in enumerate(vision_segments):
            start = (self.theta + angle_range[0]) % (2 * math.pi)
            stop = (self.theta + angle_range[1]) % (2 * math.pi)
            segment_boundaries.append((start, stop, segment_names[idx]))

        # Process each ant
        pos_x, pos_y = self.pos.x, self.pos.y
        for ant_x, ant_y in other_ants:
            # Use faster distance calculation without sqrt for initial check
            dx = ant_x - pos_x
            dy = ant_y - pos_y
            distance_squared = dx * dx + dy * dy

            if distance_squared <= vision_range_squared:
                # Only calculate angle for ants within range
                angle_to_ant = math.atan2(dy, dx) % (2 * math.pi)

                # Find matching segment
                for start_angle, stop_angle, segment_name in segment_boundaries:
                    if start_angle < stop_angle:
                        if start_angle <= angle_to_ant <= stop_angle:
                            detected_ants[segment_name].append((ant_x, ant_y, angle_to_ant))
                            break
                    elif angle_to_ant >= start_angle or angle_to_ant <= stop_angle:
                        # Angle wraps around 2π boundary
                        detected_ants[segment_name].append((ant_x, ant_y, angle_to_ant))
                        break

        return detected_ants

    def set_pheromone(self, pheromone):
        self.pheromone = pheromone

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

    def _turn(self):
        self.theta += self.theta_dot
        self.theta = self.theta % (2 * math.pi)

    def _move(self):
        """
        Move an agent from its current position (x, y) according to desired_speed
        and angle theta using matrix multiplication.
        """
        # Calculate the desired direction of travel (rotate to angle theta)
        direction = np.array([np.cos(self.theta), np.sin(self.theta)]) * self.desired_speed
        # Set the desired position based on direction and speed relative to timestep
        desired_pos = np.add(np.array(self.pos), direction)
        # If leaving the cirle, push agent back into circle.
        if is_rectangle_in_circle(desired_pos[0], desired_pos[1], self.arena_center, self.arena_radius):
            self.pos = vec2d(desired_pos[0], desired_pos[1])

    def set_action(self, action, distance=None, rotation=None):
        forward = False
        backward = False
        turn_left = False
        turn_right = False

        if action[0] > 0.25: forward = True
        if action[1] > 0.25: backward = True
        if action[2] > 0.25: turn_left = True
        if action[3] > 0.25: turn_right = True

        self.desired_speed = 0
        self.desired_turn_speed = 0

        if (forward and (not backward) and distance is not None):
            # self.desired_speed = AGENT_SPEED # * TIMESTEP
            self.desired_speed = distance
        if (backward and (not forward) and distance is not None):
            # self.desired_speed = -AGENT_SPEED # * TIMESTEP
            self.desired_speed = -distance

        rotation = -TURN_RATE / 2 if turn_left else TURN_RATE / 2 if turn_right else 0
        self.theta += rotation
        # if rotation is not None:
        #     self.theta = rotation
        # else:
        #     self.theta = -1

        # If no turn, reset the turn timers
        if not turn_left and not turn_right:
            self.turning_time_left = 0.0
            self.turning_time_right = 0.0

        return [int(x) for x in [forward, backward, turn_left, turn_right]]

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
            detected_ants = self._detect_nearby_ants(others)
            self._detect_vision(detected_ants, len(others))
            self._detect_distances(detected_ants)

        # Get the agent's position in polar coordinates
        r_norm, theta_norm = self.get_polar_position()

        # Initialize the observation vector with basic state information
        result = [
            r_norm,  # Normalized radial position
            theta_norm,  # Normalized angular position
            self.speed,  # Current speed
            self.theta,  # Current orientation
            self.theta_dot,  # Angular velocity
            self.V_f_l1,  # Vision segment: Forward Left 1
            self.V_f_l2,  # Vision segment: Forward Left 2
            self.V_f,  # Vision segment: Forward
            self.V_f_r2,  # Vision segment: Forward Right 2
            self.V_f_r1,  # Vision segment: Forward Right 1
            self.V_b_r,  # Vision segment: Backward Right
            self.V_b,  # Vision segment: Backward
            self.V_b_l,  # Vision segment: Backward Left
            self.pheromone  # Current pheromone level
        ]

        # Include composite scores for each vision segment
        # Ensure that the segments are ordered consistently
        for idx, (angle_range, _) in enumerate(vision_segments):
            segment_name = self._get_segment_name(idx)
            composite_score = self.composite_scores.get(segment_name, 0.0)
            result.append(composite_score)

        return result

    def update(self, noise=0.0):
        self.pos = vec2d(
            self.pos.x + np.random.randn() * noise,
            self.pos.y - np.random.randn() * noise
        )
        self.theta += (np.random.randn() * noise)
        self.theta = self.theta % (2 * math.pi)

        self.speed = self.desired_speed
        self.theta_dot = self.desired_turn_speed

        # self._turn()
        self._move()


class AntDynamicsEnv(gym.Env):
    metadata = {
        'name': 'ant-colony-v0',
        'render_modes': ['human', 'rgb_array'],
        'render_fps': SIM_FPS
    }

    def __init__(self, render_mode=None):
        self.np_random, seed = self.seed(seed=69)

        self.ant = None

        self.target_data = {
            'angle': [],
            'distance': [],
            'action': []
        }

        self.other_ants = None
        self.viewer = None
        self.state = None
        self.noise = 0

        self.t = 0
        self.t_limit = TIME_LIMIT

        self.time_offset = None
        self.snapshot_interval_sec = 1.0  # As per your pheromone data generation
        self.frames_per_snapshot = int(self.snapshot_interval_sec * SIM_FPS)

        self.actions = []

        assert render_mode is None or render_mode in type(self).metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

        arena_center = (SCREEN_W / 2.0, SCREEN_H / 2.0),
        arena_radius = min(SCREEN_W, SCREEN_H) / 2.0 - min(SCREEN_W, SCREEN_H) * BOUNDARY_SCALE

        # circular arena
        self.ant_arena = arena_sizes( vec2d(SCREEN_W / 2.0, SCREEN_H / 2.0), min(SCREEN_W, SCREEN_H) / 2.0 - min(SCREEN_W, SCREEN_H) * BOUNDARY_SCALE)

        # Determine the number of additional observations
        num_segments = len(vision_segments)
        additional_obs = num_segments * 2  # wall_distance and nearest_ant_distance per segment

        # Calculate the total number of observation elements
        base_obs_length = 14  # Number of elements in the initial 'result' list
        total_obs_length = base_obs_length + additional_obs

        high = np.array([np.finfo(np.float32).max] * total_obs_length)

        # Update the observation space to include additional observations
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            shape=(total_obs_length,),
            dtype=np.float32
        )

    def seed(self, seed=None):
        return seeding.np_random(seed)

    @property
    def _snapshot_index(self):
        return ((self.t + self.time_offset) // self.frames_per_snapshot)

    def _get_interval_data(self, trail, start_time, interval=False):
        theta = -1
        threshold = 5
        time = start_time + 1
        time_offset = 0
        distance = 0
        while theta < 0 and time != len(trail):
            try:
                dx, dy = trail[time] - trail[start_time]
            except IndexError:
                break
            if dx == 0 and dy == 0:
                time_offset += 1
            else:
                current_distance = np.sqrt(dx ** 2 + dy ** 2)
                if current_distance > threshold:
                    theta = math.atan2(dy, dx) % (2 * math.pi)
                    distance = current_distance
            time += 1

        return distance, theta, (time - start_time), time_offset

    def _get_angle_from_trajectory(self, trail, start_time):
        _, angle, _, _ = self._get_interval_data(trail, start_time)
        return angle

    def _identify_overlapping_grid_cells(
            self,
            pos: tuple,
            bounding_box: int,
            grid_size: tuple = (SCREEN_W, SCREEN_H),
            coarse_grid_size: tuple = (20, 20)
    ) -> list:
        """
        Identify which cells in the discretized grid an ant overlaps with based on its position and bounding box.

        Parameters:
        - x (float): X-coordinate of the ant in simulation space (0 <= x < grid_width).
        - y (float): Y-coordinate of the ant in simulation space (0 <= y < grid_height).
        - bounding_box (int): Size of the ant's bounding box in pixels (assumed square).
        - grid_size (tuple): Size of the simulation grid in pixels (height, width). Default is (900, 900).
        - coarse_grid_size (tuple): Size of the discretized grid (height, width). Default is (20, 20).

        Returns:
        - List[Tuple[int, int]]: List of (row, column) indices of grid cells overlapped by the ant.
        """
        x, y = pos

        fine_height, fine_width = grid_size
        coarse_height, coarse_width = coarse_grid_size

        # Calculate discretization factors
        factor_y = fine_height / coarse_height
        factor_x = fine_width / coarse_width

        # Calculate bounding box boundaries in simulation space
        half_box = bounding_box / 2
        x_min = max(x - half_box, 0)
        x_max = min(x + half_box, fine_width - 1)
        y_min = max(y - half_box, 0)
        y_max = min(y + half_box, fine_height - 1)

        # Map simulation space boundaries to coarse grid indices
        row_min = int(math.floor(y_min / factor_y))
        row_max = int(math.floor(y_max / factor_y))
        col_min = int(math.floor(x_min / factor_x))
        col_max = int(math.floor(x_max / factor_x))

        # Ensure indices are within coarse grid bounds
        row_min = max(row_min, 0)
        row_max = min(row_max, coarse_height - 1)
        col_min = max(col_min, 0)
        col_max = min(col_max, coarse_width - 1)

        # Generate list of overlapping grid cells
        overlapping_cells = []
        for row in range(row_min, row_max + 1):
            for col in range(col_min, col_max + 1):
                overlapping_cells.append((row, col))

        return overlapping_cells


    def _angle_difference(self, angle1, angle2):
        """
        Calculate the smallest difference between two angles (radians).
        """
        diff = (angle2 - angle1) % (2 * math.pi)
        if diff > math.pi:
            diff -= 2 * math.pi
        return diff

    def _reward_function(self, actions=None):
        """
        Calculate the reward given the focal ant and the accuracy of its behaviour
        over the trial, given the source data as the ground truth.
        """
        if REWARD_TYPE.lower() == 'action':
            return 1
        elif REWARD_TYPE.lower() == 'coverage':
            return 1
        else:
            return -69

    def get_observations(self, others=None):
        return self.ant.get_obs(others)

    def _track_trail(self, pos: vec2d):
        self.ant_trail.append(pos)

    def _destroy(self):
        self.ant = None
        self.ant_trail = []
        self.target_trail = []
        self.target_data = None
        self.target_angles = []
        self.target_distances = []
        self.other_ants = []
        self.other_ants_angles = []

        self.viewer = None
        self.state = None

        self.window = None
        self.clock = None

    def reset(self):
        self._destroy()

        self.t = 0  # timestep reset

        self.arena_mask = np.zeros((50, 50), dtype=bool)

        # Create arena mask for valid cells
        center_x = 25  # Center of 50x50 grid
        center_y = 25
        y, x = np.ogrid[-center_y:50 - center_y, -center_x:50 - center_x]
        mask_radius = (min(50, 50) / 2) * 0.98  # Slightly smaller than arena
        self.arena_mask = x * x + y * y <= mask_radius * mask_radius

        self.actions = []

        self.time_offset = 0.0

        self.ant = Ant(self.ant_arena.arena_center)

        self.other_ants = []
        rng = np.random.default_rng()
        for _ in range(20):
            x, y = rng.uniform(-self.ant_arena.arena_radius * 0.85, self.ant_arena.arena_radius * 0.85, size=2)
            self.other_ants.append((self.ant_arena.arena_center.x + x, self.ant_arena.arena_center.y + y))

        if self.other_ants is not None:
            obs = self.get_observations(self.other_ants)
        else:
            obs = self.get_observations()
        info = {
            'wall_distances': self.ant.wall_distances,
            'nearest_ant_distances': self.ant.nearest_ant_distances,
        }

        if self.render_mode == 'human':
            self._render_frame()

        return obs, info

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _get_action(self, dt):
        target_data = {
            'angle': [],
            'distance': [],
            'action': [],  # FORWARD, BACKWARD, TURN-LEFT, TURN-RIGHT, STOP, ...
        }

        action = [0, 0, 0, 0]
        info = {'distance': 0, 'target_angle': 0, 'rotation': 0}
        time = 0
        while self.t + time < len(self.target_trail) - 1 and time < dt:
            # Calculate current and next position
            current_pos = self.ant.pos
            next_pos = self.target_trail[self.t + int(time) + 1]

            # Calculate distance and angle to the next position
            dx = next_pos[0] - current_pos[0]
            dy = next_pos[1] - current_pos[1]
            distance = math.sqrt(dx ** 2 + dy ** 2)
            info['distance'] = distance
            angle_to_target = math.atan2(dy, dx)
            info['target_angle'] = angle_to_target

            # Determine angle difference from current orientation
            angle_diff = self._angle_difference(self.ant.theta, angle_to_target)
            info['rotation'] = angle_diff

            if distance > 0.05:
                action[0] = 1

            if angle_diff < 0:
                action[2] = 1
            elif angle_diff > 0:
                action[3] = 1

            time += 1

        return action, info

    def _get_stepped_action(self, dt):
        action = [0, 0, 0, 0]

        if self.target_data is None or len(self.target_data['angle']) == 0:
            self.target_data = {
                'angle': [],
                'distance': [],
                'action': []
            }
            if self.t + dt < len(self.target_trail) - 1:
                # Calculate current and next position
                current_pos = self.ant.pos
                next_pos = self.target_trail[self.t + int(dt)]

                # Calculate distance and angle to the next position
                dx = next_pos[0] - current_pos[0]
                dy = next_pos[1] - current_pos[1]
                distance = math.sqrt(dx ** 2 + dy ** 2)
                self.target_data['distance'] = [distance / dt for _ in range(dt)]
                angle_to_target = math.atan2(dy, dx)
                self.target_data['angle'] = [angle_to_target for _ in range(dt)]

                # Determine angle difference from current orientation
                angle_diff = self._angle_difference(self.ant.theta, angle_to_target)

                if distance > 0.05:
                    action[0] = 1

                if angle_diff < 0:
                    action[2] = 1
                elif angle_diff > 0:
                    action[3] = 1

            else:
                self.target_data['distance'].append(None)
                self.target_data['angle'].append(None)

        target_data = {
            'angle': self.target_data['angle'].pop(),
            'distance': self.target_data['distance'].pop()
        }

        return action, target_data

    def step(self, action):
        """
        Each step, take the given action and return observations, reward, done (bool)
        and any other additional information if necessary.
        """
        truncated = False
        self.t += 1

        # Pygame controls and resources if render_mode == 'human'
        if self.render_mode == "human":
            self._render_frame()

        # Implement target_data actions using auto_action controls
        auto_action = action # [FORWARD, BACKWARD, TURN_LEFT, TURN_RIGHT]

        # Update the ant position and orientation with the calculated distance and rotation
        # auto_action, movement_info = self._get_stepped_action(2)
        # print(movement_info['distance'], movement_info['angle'])

        self.ant.set_action(auto_action, AGENT_SPEED/15, 1)
        # self.ant.set_action(auto_action, movement_info['distance'], self.target_angles[self.t][1])
        self.ant.update()

        # Get observations of other ants in the arena
        if self.other_ants is not None:
            obs = self.get_observations(self.other_ants)
        else:
            obs = self.get_observations()
        self._track_trail(self.ant.pos)


        # Determine if the episode is done
        if self.t >= self.t_limit:
            truncated = True

        # Initialize reward value
        reward = 0
        reward = self._reward_function(auto_action)

        # Additional information (if necessary)
        info = {
            'wall_distances': self.ant.wall_distances,
            'nearest_ant_distances': self.ant.nearest_ant_distances,
        }

        return obs, reward, False, truncated, info

    def render(self, mode=None):
        if mode is not None:
            self.render_mode = mode
        # if self.render_mode == 'rgb_array':
        return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == 'human':
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption('WANNts')
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

        if DRAW_ANT_VISION:
            pygame.draw.circle(
                canvas,
                (225, 225, 230, 255),
                (int(self.ant.pos.x), int(self.ant.pos.y)),
                VISION_RANGE
            )
            for (start_angle, stop_angle), colour in vision_segments:
                # Calculate start and stop angles, normalized to 0 to 2*pi
                start_angle = (self.ant.theta + start_angle) % (2 * math.pi)
                stop_angle = (self.ant.theta + stop_angle) % (2 * math.pi)

                pygame.draw.line(
                    canvas,
                    colour,
                    (int(self.ant.pos.x), int(self.ant.pos.y)),
                    (
                        int(self.ant.pos.x + np.cos(start_angle) * VISION_RANGE),
                        int(self.ant.pos.y + np.sin(start_angle) * VISION_RANGE)
                    )
                )

        # Draw wall distance indicators
        # for segment, distance in self.ant.wall_distances.items():
        #     if distance is not None:
        #         # Calculate the central angle of the segment
        #         segment_idx = ['forward_l1', 'forward_l2', 'forward', 'forward_r2',
        #                     'forward_r1', 'backward_r', 'backward', 'backward_l'].index(segment)
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
        for segment, distance_info in self.ant.nearest_ant_distances.items():
            distance, angle = distance_info
            if distance is not None and distance != 0.0:
                # Calculate the central angle of the segment
                segment_idx = ['forward_l1', 'forward_l2', 'forward', 'forward_r2',
                               'forward_r1', 'backward_r', 'backward', 'backward_l'].index(segment)
                angle_range, _ = vision_segments[segment_idx]
                central_angle = (angle_range[0] + angle_range[1]) / 2
                central_angle = (self.ant.theta + central_angle) % (2 * math.pi)

                # Calculate end point based on nearest ant distance
                end_x = int(self.ant.pos.x + math.cos(angle) * (1 - distance) * VISION_RANGE)
                end_y = int(self.ant.pos.y + math.sin(angle) * (1 - distance) * VISION_RANGE)

                # Draw a line from the ant to the nearest ant
                pygame.draw.line(
                    canvas,
                    (0, 0, 0),  # black color for nearest ant distance
                    (int(self.ant.pos.x), int(self.ant.pos.y)),
                    (end_x, end_y),
                    1
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

        ### THEN DRAW ANTS AT THEIR CURRENT POSITIONS

        # Draw other ants' positions
        if self.other_ants is not None:
            try:
                for other_ant in self.other_ants:
                    pygame.draw.rect(
                        canvas,
                        (180, 180, 180),
                        (
                            other_ant[0] - ANT_DIM.x / 2.,
                            other_ant[1] - ANT_DIM.y / 2.,
                            ANT_DIM.x,
                            ANT_DIM.y
                        )
                    )

            except IndexError:
                print(other_ant)
                logger.error("End of time series reached for other ants.")
            except TypeError:
                print(other_ant)
                logger.error("Cannot draw ant with provided coordinates.")

        # Draw agent last; to ensure visibility.
        pygame.draw.rect(
            canvas,
            (0, 0, 180),
            (
                self.ant.pos.x - ANT_DIM.x / 2.,
                self.ant.pos.y - ANT_DIM.y / 2.,
                ANT_DIM.x,
                ANT_DIM.y
            )
        )
        pygame.draw.line(
            canvas,
            (0, 0, 180),
            (int(self.ant.pos.x), int(self.ant.pos.y)),
            (
                int(self.ant.pos.x + np.cos(self.ant.theta) * ANT_DIM.x * 3),
                int(self.ant.pos.y + np.sin(self.ant.theta) * ANT_DIM.x * 3)
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
    env = AntDynamicsEnv(render_mode='human')

    total_reward = 0
    obs, info = env.reset()

    manual_mode = True
    manual_action = [0, 0, 0, 0]

    done = False
    while not done:
        if manual_mode:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        done = True
                    if event.key == pygame.K_r:
                        env.reset()
                    if event.key == pygame.K_UP:    manual_action[0] = 1
                    if event.key == pygame.K_DOWN:  manual_action[1] = 1
                    if event.key == pygame.K_LEFT:  manual_action[2] = 1
                    if event.key == pygame.K_RIGHT: manual_action[3] = 1
                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_UP:    manual_action[0] = 0
                    if event.key == pygame.K_DOWN:  manual_action[1] = 0
                    if event.key == pygame.K_LEFT:  manual_action[2] = 0
                    if event.key == pygame.K_RIGHT: manual_action[3] = 0
            action = manual_action
            if done: break

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        if done: break

    env.close()
    print('Cumulative score:', total_reward)