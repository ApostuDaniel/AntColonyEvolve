from .arena_config import vision_segments, view_ranges, pheromone_ranges, PHEROMONE_TYPES

# S, M or L; Small, Medium or Large (logic implemented as "Not 'S' or 'M' then Large").
VERSION = "M"
VERSION_TEXT = "small" if VERSION == "S" else "medium" if VERSION == "M" else "large"


def get_input_coordinates():
    # coordinates that we try to allign parallel
    inputs = [
        # (0., 0.0), # r_norm
        # (0., -0.5), # theta_norm
        (-0.2, -1.0),  # speed
        (-0.1, -1.0),  # theta
        (0.1, -1.0),  # heartbeat
        (0.2, -1.0),  # travel_distance
        (-0.2, -0.8),  # current_energy
        (0.2, -0.8),  # carried_food
        (0., -0.8),  # ant_in_nest
        (-0.2, -0.6),  # reward_sensor
        (0.2, -0.6),  # penalty_sensor
        (0.2, -0.5),  # distance_to_nest
        (-0.2, -0.5),  # angle_to_nest
    ]

    # depending on how many vision segments we use, compute their spacing in the substrate on x-axis
    vision_segments_x_coord = [-1. + i * 2. / (len(vision_segments) - 1) for i in range(len(vision_segments))]

    # provide density areas coordinates
    vision_ranges_y_coord = [-0.4 - 0.1 * i for i in range(len(view_ranges))]
    for y in vision_ranges_y_coord:
        inputs.extend([(x, y) for x in vision_segments_x_coord])

    # provide pheromones coordinates
    pheromone_ranges_y_modifiers = [0.15 * i for i in range(len(pheromone_ranges))]
    for y_modifier in pheromone_ranges_y_modifiers:
        for x in vision_segments_x_coord:
            inputs.extend([(x, -0.2 - (i * 0.04) + y_modifier) for i in range(PHEROMONE_TYPES)])  # pheromone coordinates for each vision segment

    inputs.extend([(x, 0.05) for x in vision_segments_x_coord])  # vision_segments_wall_distances
    # inputs.extend([(x, 0.2) for x in vision_segments_x_coord])  # vision_segments_food_distances
    # inputs.extend([(x, 0.3) for x in vision_segments_x_coord])  # vision_segments_food_angles

    return inputs


def get_output_coordinates():
    outputs = \
        [
            (0., 0.7),  # acceleration
            (-1.0, 0.7),  # turn left
            (1.0, 0.7),  # turn_right
            (0., 0.5)  # pheromone type
        ]
    return outputs


def params(version):
    """
    ES-HyperNEAT specific parameters.
    """
    return {
        "initial_depth": 0 if version == "S" else 1 if version == "M" else 2,
        "max_depth": 1 if version == "S" else 2 if version == "M" else 3,
        "variance_threshold": 0.04,
        "band_threshold": 0.35,
        "iteration_level": 2,
        "division_threshold": 0.55,
        "max_weight": 6.0,
        "activation": "tanh"
    }
