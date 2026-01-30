import math
from collections import namedtuple
import configparser
import os

SIM_FPS = 60  # Simulation environment FPS

SCREEN_W = 1152
SCREEN_H = 1152
BOUNDARY_SCALE = 0.005

vec2d = namedtuple('vec2d', ['x', 'y'])
arena_sizes = namedtuple('arena_sizes', ['arena_center', 'arena_radius'])

# Global parameters for agent control
TIMESTEP = 1. / SIM_FPS  # Not sure if this will be necessary, given the fixed FPS?
TIME_LIMIT = SIM_FPS * 10  # 10 seconds

ANT_DIM = vec2d(6, 6)
AGENT_SPEED = 12.00  # Maximum ant speed observed in data (pixels)
AGENT_ACCELERATION = 4.0
TURN_RATE = math.pi / 2  # When turn_left = 1 and turn_right = -1 ori viceversa, allows maximum 180° turn per step
MAX_TRAVEL_PER_EPISODE = AGENT_SPEED * TIME_LIMIT

NEST_RADIUS = ANT_DIM[0] * 6
DRAW_NEST_SENSORS = True

SPEED_PENALTY_BASE = 1.5 ** (1 / AGENT_SPEED)

MAX_VISION_RANGE = ANT_DIM.x * 14
MIDDLE_VISION_RANGE = int(MAX_VISION_RANGE * (2 / 3))
MIN_VISION_RANGE = int(MAX_VISION_RANGE * (1 / 2))

MAX_FOOD_QUANTITY = 3
FOOD_SIZE_SCALING = ANT_DIM[0] * 2.5
ANT_CARRY_CAPACITY = 6
ANT_LIFE = 50
SHOULD_FOOD_DECREASE = True
SHOULD_FOOD_RESPAWN = False

DRAW_ANT_VISION = True
DRAW_FOOD_DENSITY = True
DRAW_PHEROMONES = True
DRAW_PHEROMONE_VISION = True
DRAW_FOOD_GRID = False
HIGHLIGHT_BEST_ANT = False

# 5 vision segments
# vision_segments = [
#     # Front arc: Directly in front of the agent
#     # Bright red, intense and strong
#     ((-math.pi / 2, -3 * math.pi / 10), (255, 0, 0), 'forward_l1'),
#     # Vivid orange, warm and energetic
#     ((-3 * math.pi / 10, -math.pi / 10), (255, 165, 0), 'forward_l2'),
#     # Bright yellow, cheerful and vibrant
#     ((-math.pi / 10, math.pi / 10), (255, 255, 0), 'forward'),
#     # Bright green, fresh and lively
#     ((math.pi / 10, 3 * math.pi / 10), (0, 255, 0), 'forward_r2'),
#     # Sky blue, cool and calm
#     ((3 * math.pi / 10, math.pi / 2), (0, 0, 255), 'forward_r1'),
# ]

# 3 vision segments
vision_segments = [
    # Bright red, intense and strong
    ((-math.pi / 2, -math.pi / 6), (255, 0, 0), 'forward_l1'),
    # Bright yellow, cheerful and vibrant
    ((-math.pi / 6, math.pi / 6), (255, 255, 0), 'forward'),
    # Sky blue, cool and calm
    ((math.pi / 6, math.pi / 2), (0, 0, 255), 'forward_r1'),
]

PHEROMONE_DETECTION_MAX_RANGE = MIN_VISION_RANGE
PHEROMONE_DETECTION_MIN_RANGE = PHEROMONE_DETECTION_MAX_RANGE / 1.5
PHEROMONE_TYPES = 2
PHEROMONE_COLORS = [(255, 0, 0), (0, 0, 255)]
PHEROMONE_DECAY = 0.99
PHEROMONE_DIFFUSION = 0.08
PHEROMONE_DEPOSIT = 1.0
MAX_PHEROMONE_CONCENTRATION_PER_ZONE = 6
MAX_DETECTABLE_PHEROMONE_CONCENTRATION = 2
MIN_DETECTABLE_PHEROMONE_CONCENTRATION = 0.01

view_ranges = [MAX_VISION_RANGE]
pheromone_ranges = [PHEROMONE_DETECTION_MIN_RANGE]

"""
Observation name and whether it should be used or not
"""
possible_observations = {
        "r_norm": False,  # Normalized radial position
        "theta_norm": False,  # Normalized angular position
        "speed": True,  # Normalized current speed
        "theta": True,  # Normalized current orientation
        "heartbeat": True,  # HEARTBEAT
        "travel_distance": True,  # normalized travel distance
        "current_energy": True,  # normalized health
        "carried_food": True,  # normalize food carry
        "ant_in_nest": True,  # 1 if ant is in nest, 0 otherwise
        "reward_sensor": True,  # how much reward is given for current step
        "penalty_sensor": True,  # how much penalty is incurred for current step
        "distance_to_nest": True,  # normalized to arena diameter, the maximum distance between ant and nest
        "angle_to_nest": True,  # normalized angle to nest
        "pheromones_sensor": False,  # inputs of pheromone detection areas
        "nearest_wall_distances": True,  # scaled nearest wall distances per vision segment
        "nearest_food_distances": True,  # nearest food distances per vision segment
        "nearest_food_angles": True,  # normalized angle to the nearest food per vision segment
        "food_density": False,  # a food density fraction per vision segment per view range
    }

def parse_vision_segments(config):
    """
    Parse vision segments from the configuration file.
    """
    segments = []
    if 'VisionSegments' in config:
        for key in config['VisionSegments']:
            try:
                segment_data = config['VisionSegments'][key].split(',')
                angles = (float(segment_data[0]), float(segment_data[1]))
                color = (int(segment_data[2]), int(segment_data[3]), int(segment_data[4]))
                label = segment_data[5].strip()
                segments.append((angles, color, label))
            except (ValueError, IndexError):
                print(f"Warning: Skipping invalid vision segment entry: {key}")
    return segments

def load_parameters(config_file):
    """
    Load parameters from a configuration file and update the global variables.
    Parameters not found in the file will retain their default values.
    """
    global SIM_FPS, SCREEN_W, SCREEN_H, BOUNDARY_SCALE, TIME_LIMIT, AGENT_SPEED, \
        AGENT_ACCELERATION, TURN_RATE, MAX_FOOD_QUANTITY, FOOD_SIZE_SCALING, \
        ANT_CARRY_CAPACITY, ANT_LIFE, SHOULD_FOOD_DECREASE, SHOULD_FOOD_RESPAWN, \
        DRAW_ANT_VISION, DRAW_FOOD_DENSITY, DRAW_PHEROMONES, DRAW_PHEROMONE_VISION, \
        DRAW_FOOD_GRID, HIGHLIGHT_BEST_ANT, PHEROMONE_DECAY, PHEROMONE_DEPOSIT, \
        MIN_DETECTABLE_PHEROMONE_CONCENTRATION, MAX_DETECTABLE_PHEROMONE_CONCENTRATION, \
        vision_segments, view_ranges, pheromone_ranges

    config = configparser.ConfigParser()
    config.read(config_file)

    if 'Arena' in config:
        SIM_FPS = config.getint('Arena', 'SIM_FPS', fallback=SIM_FPS)
        SCREEN_W = config.getint('Arena', 'SCREEN_W', fallback=SCREEN_W)
        SCREEN_H = config.getint('Arena', 'SCREEN_H', fallback=SCREEN_H)
        BOUNDARY_SCALE = config.getfloat('Arena', 'BOUNDARY_SCALE', fallback=BOUNDARY_SCALE)
        TIME_LIMIT = config.getint('Arena', 'TIME_LIMIT', fallback=TIME_LIMIT)
        AGENT_SPEED = config.getfloat('Arena', 'AGENT_SPEED', fallback=AGENT_SPEED)
        AGENT_ACCELERATION = config.getfloat('Arena', 'AGENT_ACCELERATION', fallback=AGENT_ACCELERATION)
        TURN_RATE = config.getfloat('Arena', 'TURN_RATE', fallback=TURN_RATE)
        MAX_FOOD_QUANTITY = config.getint('Arena', 'MAX_FOOD_QUANTITY', fallback=MAX_FOOD_QUANTITY)
        FOOD_SIZE_SCALING = config.getfloat('Arena', 'FOOD_SIZE_SCALING', fallback=FOOD_SIZE_SCALING)
        ANT_CARRY_CAPACITY = config.getint('Arena', 'ANT_CARRY_CAPACITY', fallback=ANT_CARRY_CAPACITY)
        ANT_LIFE = config.getint('Arena', 'ANT_LIFE', fallback=ANT_LIFE)
        SHOULD_FOOD_DECREASE = config.getboolean('Arena', 'SHOULD_FOOD_DECREASE', fallback=SHOULD_FOOD_DECREASE)
        SHOULD_FOOD_RESPAWN = config.getboolean('Arena', 'SHOULD_FOOD_RESPAWN', fallback=SHOULD_FOOD_RESPAWN)
        DRAW_ANT_VISION = config.getboolean('Arena', 'DRAW_ANT_VISION', fallback=DRAW_ANT_VISION)
        DRAW_FOOD_DENSITY = config.getboolean('Arena', 'DRAW_FOOD_DENSITY', fallback=DRAW_FOOD_DENSITY)
        DRAW_PHEROMONES = config.getboolean('Arena', 'DRAW_PHEROMONES', fallback=DRAW_PHEROMONES)
        DRAW_PHEROMONE_VISION = config.getboolean('Arena', 'DRAW_PHEROMONE_VISION', fallback=DRAW_PHEROMONE_VISION)
        DRAW_FOOD_GRID = config.getboolean('Arena', 'DRAW_FOOD_GRID', fallback=DRAW_FOOD_GRID)
        HIGHLIGHT_BEST_ANT = config.getboolean('Arena', 'HIGHLIGHT_BEST_ANT', fallback=HIGHLIGHT_BEST_ANT)
        PHEROMONE_DECAY = config.getfloat('Arena', 'PHEROMONE_DECAY', fallback=PHEROMONE_DECAY)
        PHEROMONE_DEPOSIT = config.getfloat('Arena', 'PHEROMONE_DEPOSIT', fallback=PHEROMONE_DEPOSIT)
        MIN_DETECTABLE_PHEROMONE_CONCENTRATION = config.getfloat('Arena', 'MIN_DETECTABLE_PHEROMONE_CONCENTRATION', fallback=MIN_DETECTABLE_PHEROMONE_CONCENTRATION)
        MAX_DETECTABLE_PHEROMONE_CONCENTRATION = config.getfloat('Arena', 'MAX_DETECTABLE_PHEROMONE_CONCENTRATION', fallback=MAX_DETECTABLE_PHEROMONE_CONCENTRATION)
        view_ranges = [MAX_VISION_RANGE]
        pheromone_ranges = [PHEROMONE_DETECTION_MIN_RANGE]

    # Update vision segments
    new_vision_segments = parse_vision_segments(config)
    if new_vision_segments:
        vision_segments = new_vision_segments


def save_parameters(dir_path):
    """
    Save the current parameters to a configuration file.
    """
    config = configparser.ConfigParser()

    # Save Arena parameters
    config['Arena'] = {
        'SIM_FPS': SIM_FPS,
        'SCREEN_W': SCREEN_W,
        'SCREEN_H': SCREEN_H,
        'BOUNDARY_SCALE': BOUNDARY_SCALE,
        'TIME_LIMIT': TIME_LIMIT,
        'AGENT_SPEED': AGENT_SPEED,
        'AGENT_ACCELERATION': AGENT_ACCELERATION,
        'TURN_RATE': TURN_RATE,
        'MAX_FOOD_QUANTITY': MAX_FOOD_QUANTITY,
        'FOOD_SIZE_SCALING': FOOD_SIZE_SCALING,
        'ANT_CARRY_CAPACITY': ANT_CARRY_CAPACITY,
        'ANT_LIFE': ANT_LIFE,
        'SHOULD_FOOD_DECREASE': SHOULD_FOOD_DECREASE,
        'SHOULD_FOOD_RESPAWN': SHOULD_FOOD_RESPAWN,
        'DRAW_ANT_VISION': DRAW_ANT_VISION,
        'DRAW_FOOD_DENSITY': DRAW_FOOD_DENSITY,
        'DRAW_PHEROMONES': DRAW_PHEROMONES,
        'DRAW_PHEROMONE_VISION': DRAW_PHEROMONE_VISION,
        'DRAW_FOOD_GRID': DRAW_FOOD_GRID,
        'HIGHLIGHT_BEST_ANT': HIGHLIGHT_BEST_ANT,
        'PHEROMONE_DECAY': PHEROMONE_DECAY,
        'PHEROMONE_DEPOSIT': PHEROMONE_DEPOSIT,
        'MIN_DETECTABLE_PHEROMONE_CONCENTRATION': MIN_DETECTABLE_PHEROMONE_CONCENTRATION,
        'MAX_DETECTABLE_PHEROMONE_CONCENTRATION': MAX_DETECTABLE_PHEROMONE_CONCENTRATION,
    }

    # Save VisionSegments
    config['VisionSegments'] = {}
    for i, segment in enumerate(vision_segments):
        angles, color, label = segment
        config['VisionSegments'][f'segment{i+1}'] = f"{angles[0]},{angles[1]},{color[0]},{color[1]},{color[2]},{label}"

    # Ensure the directory exists
    os.makedirs(os.path.dirname(dir_path), exist_ok=True)

    # Write to the file
    with open("arena_config.ini", 'w') as file:
        config.write(file)
