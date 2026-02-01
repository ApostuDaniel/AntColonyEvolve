from .arena_config import vision_segments, view_ranges, pheromone_ranges, PHEROMONE_TYPES, nest_detection_segments

# S, M or L; Small, Medium or Large (logic implemented as "Not 'S' or 'M' then Large").
VERSION = "S"
VERSION_TEXT = "small" if VERSION == "S" else "medium" if VERSION == "M" else "large"


def get_input_coordinates():
    # coordinates that we try to allign parallel
    inputs = [
        # (0., 0.0), # r_norm
        # (0., -0.5), # theta_norm
        (-0.5, -0.2),  # speed
        (-0.5, -0.4),  # theta
        # (0.1, -1.0),  # heartbeat
        # (0.2, -1.0),  # travel_distance
        # (-0.2, -0.8),  # current_energy
        (+0.5, -0.2),  # carried_food
        (+0.5, -0.4),  # ant_in_nest
        (-0.3, -0.15),  # reward_sensor
        (+0.3, -0.15),  # penalty_sensor
        # (0.0, -0.3),  # distance_to_nest
        # (0.0, -0.5),  # angle_to_nest
    ]

    nest_segments_x = [-1. + i * 2. / (len(nest_detection_segments) - 1) for i in range(len(nest_detection_segments))]
    inputs.extend([(x, -0.9) for x in nest_segments_x])

    # depending on how many vision segments we use, compute their spacing in the substrate on x-axis
    vision_segments_x_coord = [-1. + i * 2. / (len(vision_segments) - 1) for i in range(len(vision_segments))]

    # provide density areas coordinates
    vision_ranges_y_coord = [-0.8 + 0.1 * i for i in range(len(view_ranges))]
    for y in vision_ranges_y_coord:
        inputs.extend([(x, y) for x in vision_segments_x_coord])

    # provide pheromones coordinates
    pheromone_ranges_y_modifiers = [0.1 * i for i in range(len(pheromone_ranges))]
    for y_modifier in pheromone_ranges_y_modifiers:
        for x in vision_segments_x_coord:
            inputs.extend([(x, -0.6 + (i * 0.1) + y_modifier) for i in range(PHEROMONE_TYPES)])  # pheromone coordinates for each vision segment

    inputs.extend([(x, -1.0) for x in vision_segments_x_coord])  # vision_segments_wall_distances
    # inputs.extend([(x, -0.9) for x in vision_segments_x_coord])  # vision_segments_food_distances
    # inputs.extend([(x, -0.8) for x in vision_segments_x_coord])  # vision_segments_food_angles

    return inputs


def get_output_coordinates():
    outputs = \
        [
            (0.0, 1.0),  # acceleration
            (-0.6, 0.85),  # turn left
            (+0.6, 0.85),  # turn_right
            (0.0, 0.8)  # pheromone type
        ]
    return outputs


# def params(version):
#     """
#     ES-HyperNEAT specific parameters.
#     """
#     return {
#         "initial_depth": 1 if version == "S" else 2 if version == "M" else 3,
#         "max_depth": 2 if version == "S" else 3 if version == "M" else 4,
#         "variance_threshold": 0.03,
#         "band_threshold": 0.30,
#         "iteration_level": 3,
#         "division_threshold": 0.50,
#         "max_weight": 5.0,
#         "activation": "tanh",
#     }
def params(version):
    """
    ES-HyperNEAT specific parameters.
    """
    if version == "S":  # Small/Simple - For debugging
        return {
            "initial_depth": 0,
            "max_depth": 1,
            "variance_threshold": 0.08,  # Higher = fewer hidden nodes
            "band_threshold": 0.4,  # Higher = more aggressive pruning
            "iteration_level": 1,  # Lower for simpler networks
            "division_threshold": 0.6,  # Higher = less subdivision
            "max_weight": 4.0,  # Lower to prevent extreme activations
            "activation": "sigmoid"  # Bounded output
        }

    elif version == "M":  # Medium - RECOMMENDED START HERE
        return {
            "initial_depth": 1,
            "max_depth": 2,
            "variance_threshold": 0.06,  # ⬆️ Increased from 0.04
            "band_threshold": 0.35,  # Standard
            "iteration_level": 1,  # ⬇️ Reduced from 2
            "division_threshold": 0.5,  # ⬇️ Reduced from 0.55
            "max_weight": 5.0,  # ⬇️ Reduced from 6.0
            "activation": "tanh"  # Good for motor control
        }

    else:  # "L" - Large/Complex - Use after M works
        return {
            "initial_depth": 1,
            "max_depth": 3,
            "variance_threshold": 0.04,  # Lower = more hidden nodes
            "band_threshold": 0.3,  # Standard from research
            "iteration_level": 2,  # Higher for finer resolution
            "division_threshold": 0.5,
            "max_weight": 5.0,
            "activation": "tanh"
        }
