import math
from collections import namedtuple

SIM_FPS = 60  # Simulation environment FPS

SCREEN_W = 960
SCREEN_H = 960
BOUNDARY_SCALE = 0.01

vec2d = namedtuple('vec2d', ['x', 'y'])
arena_sizes = namedtuple('arena_sizes', ['arena_center', 'arena_radius'])

# Global parameters for agent control
TIMESTEP = 1. / SIM_FPS  # Not sure if this will be necessary, given the fixed FPS?
TIME_LIMIT = SIM_FPS * 10  # 12 seconds

ANT_DIM = vec2d(7, 7)
AGENT_SPEED = 13.00  # Maximum ant speed observed in data (pixels)
AGENT_ACCELERATION = 5.0
TURN_RATE = 2 * 0.3575  # Maximum turning rate observed in data (radians)
MAX_TRAVEL_PER_EPISODE = AGENT_SPEED * TIME_LIMIT

NEST_RADIUS = SCREEN_W / 20
DRAW_NEST_SENSORS = False

SPEED_PENALTY_BASE = 1.5 ** (1 / AGENT_SPEED)

MAX_VISION_RANGE = ANT_DIM.x * 16
MIDDLE_VISION_RANGE = int(MAX_VISION_RANGE * (2/3))
MIN_VISION_RANGE = int(MAX_VISION_RANGE * (1/3))

MAX_FOOD_QUANTITY = 150
FOOD_SIZE_SCALING = 0.3
ANT_CARRY_CAPACITY = 30
ANT_LIFE = 150
SHOULD_FOOD_DECREASE = True
SHOULD_FOOD_RESPAWN = True

DRAW_ANT_VISION = True
DRAW_FOOD_DENSITY = False
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
PHEROMONE_TYPES = 3
PHEROMONE_DECAY = 0.98
PHEROMONE_DEPOSIT = 1.0
MAX_DETECTABLE_PHEROMONE_CONCENTRATION = 2

view_ranges = [MIN_VISION_RANGE, MIDDLE_VISION_RANGE, MAX_VISION_RANGE]
pheromone_ranges = [PHEROMONE_DETECTION_MIN_RANGE ,PHEROMONE_DETECTION_MAX_RANGE]