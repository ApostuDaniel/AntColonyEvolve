import logging
import os
import pickle
from datetime import datetime

import numpy as np
from neat import StdOutReporter

import ant_colony_arena.ant_colony_arena as ant_arena
import neat
from utils.neat_plot_reporter import PlotReporter

from pureples.shared.visualize import draw_net
from pureples.shared.substrate import Substrate
from pureples.es_hyperneat.es_hyperneat import ESNetwork

# S, M or L; Small, Medium or Large (logic implemented as "Not 'S' or 'M' then Large").
VERSION = "S"
VERSION_TEXT = "small" if VERSION == "S" else "medium" if VERSION == "M" else "large"

# Original input coordinates based on diagram
# INPUT_COORDINATES = [
#     (0.0, 0.0), # r_norm
#     (0.0, -0.05), # theta_norm
#     (0.0, -0.1), # speed
#     (0.0, -0.2), # theta
#     (-0.95, -0.31), (-0.59, -0.81),(0, -1.0), (0.59, -0.81), (0.95, -0.31), #vision_segments_wall_distances
#     (-0.92, -0.3), (-0.57, -0.78), (0, -0.97), (0.57, -0.78), (0.92, -0.3),  # vision_segments_food_distances
#     (-0.31, -0.1), (-0.19, -0.26), (0, -0.33), (0.19, -0.26), (0.31, -0.1),  # vision_segments_food_angles
#     (-0.2, -0.06), (-0.12, -0.17), (0, -0.2), (0.12, -0.17), (0.2, -0.06),  # density_area_close
#     (-0.48, -0.15), (-0.3, -0.4), (0, -0.5), (0.3, -0.4), (0.48, -0.15),  # density_area_middle
#     (-0.8, -0.26), (-0.49, -0.68), (0, -0.82), (0.49, -0.68), (0.8, -0.26),  # density_area_far
# ]

# # Favourite
# INPUT_COORDINATES = [
#     (0.0, 0.0), # r_norm
#     (0.0, -0.1), # theta_norm
#     (0.0, -0.2), # speed
#     (0.0, -0.3), # theta
#     (-1, -0.3), (-0.6, -0.8),(0, -1.0), (0.6, -0.8), (1, -0.3), #vision_segments_wall_distances
#     (-0.9, -0.3), (-0.5, -0.8), (0, -0.9), (0.5, -0.8), (0.9, -0.3),  # vision_segments_food_distances
#     # (-0.30, -0.1), (-0.2, -0.3), (0, -0.3), (0.2, -0.3), (0.3, -0.1),  # vision_segments_food_angles
#     # (-0.2, -0.1), (-0.1, -0.15), (0, -0.2), (0.1, -0.15), (0.2, -0.1),  # density_area_close
#     # (-0.5, -0.15), (-0.3, -0.4), (0, -0.5), (0.3, -0.4), (0.5, -0.15),  # density_area_middle
#     # (-0.8, -0.25), (-0.5, -0.7), (0, -0.8), (0.5, -0.7), (0.8, -0.25),  # density_area_far
# ]

#coordinates that we try to allign parallel
INPUT_COORDINATES = [
    (0.0, 0.0), # r_norm
    (0.0, -0.1), # theta_norm
    (0.0, -0.2), # speed
    (0.0, -0.3), # theta
    (-1, -1.), (-0.5, -1.),(0, -1.), (0.5, -1.), (1, -1.), #vision_segments_wall_distances
    (-1, -0.8), (-0.5, -0.8),(0, -0.8), (0.5, -0.8), (1, -0.8),  # vision_segments_food_distances
    # (-1, -0.7), (-0.5, -0.7),(0, -0.7), (0.5, -0.7), (1, -0.7),  # vision_segments_food_angles
    # (-0.7, -0.4), (-0.35, -0.4), (0, -0.4), (0.35, -0.4), (0.7, -0.4),  # density_area_close
    # (-0.8, -0.5), (-0.4, -0.5), (0, -0.5), (0.4, -0.5), (0.8, -0.5),  # density_area_middle
    # (-0.9, -0.6), (-0.6, -0.6), (0, -0.6), (0.6, -0.6), (0.9, -0.6),  # density_area_far
]

OUTPUT_COORDINATES = \
    [
        (0, 0.5), # acceleration
        (-0.5, 0.5), # turn left
        (0.5, 0.5) #turn_right
    ]

SUBSTRATE = Substrate(INPUT_COORDINATES, OUTPUT_COORDINATES)

def params(version):
    """
    ES-HyperNEAT specific parameters.
    """
    return {
        "initial_depth": 0 if version == "S" else 1 if version == "M" else 2,
        "max_depth": 1 if version == "S" else 2 if version == "M" else 3,
        "variance_threshold": 0.03,
        "band_threshold": 0.3,
        "iteration_level": 1,
        "division_threshold": 0.03,
        "max_weight": 8.0,
        "activation": "tanh"
    }

# Config for CPPN.
CONFIG = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            './config_cppn_ant_evolution')

def ini_pop(state, stats, config, output, output_dir=None):
    """
    Initialize population attaching statistics reporter.
    """
    pop = neat.population.Population(config, state)
    if output:
        if output_dir:
            pop.add_reporter(PlotReporter(True, 20, output_dir))
        else:
            pop.add_reporter(StdOutReporter(True))
    pop.add_reporter(stats)
    return pop


def run_es(gens, env, max_steps, config, params, substrate, max_trials=100, output=True, output_dir=None):
    """
    Generic OpenAI Gym runner for ES-HyperNEAT.
    """
    trials = 3
    # pop_size = len(env.agents)
    def eval_fitness(genomes, config):

        # Modified neat/reproduction.py compute_spawn() function to limit spawn_amounts to pop size
        # for (_, g) in genomes: g.fitness = 0.
        # genomes = genomes[:pop_size]

        cppns = [neat.nn.FeedForwardNetwork.create(g, config) for _, g in genomes]
        networks = [ESNetwork(substrate, cppn, params) for cppn in cppns]
        nets = [network.create_phenotype_network() for network in networks]

        fitnesses = [[] for _ in genomes]
        for _ in range(trials):
            observations, infos = env.reset()
            terminations = [False]
            truncations = [False]

            for net in nets: net.reset()

            total_rewards = [0 for _ in genomes]

            while env.agents and not any(terminations) and not any(truncations):
                outputs = []
                for i, network in enumerate(networks):
                    for _ in range(network.activations):
                        output = nets[i].activate(np.array(observations[f"ant_{i}"]))
                    outputs.append(output)
                # outputs = [nets[i].activate(observations[i]) for i in range(len(networks)) for _ in range(networks[i].activations)]


                actions =  {f"ant_{i}": o for i, o in enumerate(outputs)}
                observations, rewards, terminations, truncations, infos = env.step(actions)
                total_rewards = [total + rewards[f"ant_{i}"] for i, total in enumerate(total_rewards)]
                terminations = [terminations[a] for a in terminations]
                truncations = [truncations[a] for a in truncations]

            for i in range(len(total_rewards)):
                fitnesses[i].append(total_rewards[i])

        for i, (_, g) in enumerate(genomes):
            g.fitness = np.array(fitnesses[i]).mean()

    # Create population and train the network. Return winner of network running 100 episodes.
    stats_one = neat.statistics.StatisticsReporter()
    pop = ini_pop(None, stats_one, config, output, output_dir)
    winner_one = pop.run(eval_fitness, gens)

    # stats_ten = neat.statistics.StatisticsReporter()
    # pop = ini_pop((pop.population, pop.species, 0), stats_ten, config, output)
    # trials = 10
    # winner_ten = pop.run(eval_fitness, gens)
    #
    # if max_trials == 0:
    #     return winner_ten, (stats_one, stats_ten)
    #
    # stats_hundred = neat.statistics.StatisticsReporter()
    # pop = ini_pop((pop.population, pop.species, 0),
    #               stats_hundred, config, output)
    # trials = max_trials
    # winner_hundred = pop.run(eval_fitness, gens)
    env.close()

    return winner_one, (stats_one,)

def run(gens, env, version, output_dir=None):
    """
    Run the ant evolution task using the Gym environment
    Returns the winning genome and the statistics of the run.
    """
    winner, stats = run_es(gens, env, 200, CONFIG, params(
        version), SUBSTRATE, max_trials=0, output_dir=output_dir)
    print(f"es_hyperneat_ant_evolution_{VERSION_TEXT} done")
    return winner, stats


if __name__ == "__main__":
    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.INFO)

    # env = ant_arena.AntColonyEnv(50, 50, render_mode='human')
    env = ant_arena.AntColonyEnv(50, 50)
    env.reset()

    run_specificity="es_hyperneat_parameters_fixed"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    current_run_directory = f"alg_runs/run_{timestamp}_{run_specificity}"

    # Run! Only relevant to look at the winner.
    WINNER = run(60, env, VERSION, os.path.join(current_run_directory, "plots"))[0]

    os.makedirs(current_run_directory, exist_ok=True)
    # Save CPPN if wished reused and draw it + winner to file.
    CPPN = neat.nn.FeedForwardNetwork.create(WINNER, CONFIG)
    NETWORK = ESNetwork(SUBSTRATE, CPPN, params(VERSION))
    NET = NETWORK.create_phenotype_network(
        filename=os.path.join(current_run_directory, f"es_hyperneat_ant_evolution_{VERSION_TEXT}_winner"))
    draw_net(CPPN, filename=os.path.join(current_run_directory, f"es_hyperneat_ant_evolution_{VERSION_TEXT}_cppn"))
    with open(os.path.join(current_run_directory, f'es_hyperneat_ant_evolution_{VERSION_TEXT}_cppn.pkl'), 'wb') as output:
        pickle.dump(CPPN, output, pickle.HIGHEST_PROTOCOL)

# Added before return in neat/reproduction.py compute_spawn() function
# # If there are more spawns than the inital population size, randomly remove based on roulete wheel selection
# total_spawn = sum(spawn_amounts)
# if total_spawn > pop_size:
#     roulete_intervals = list(adjusted_fitness)
#     for i in range(1, len(spawn_amounts)):
#         roulete_intervals[i] = roulete_intervals[i - 1] + spawn_amounts[i]
#
# while total_spawn > pop_size:
#     random_af_pick = random.random()
#     last_step = 0
#     for i, roulete_interval in enumerate(roulete_intervals):
#         if last_step <= random_af_pick < roulete_interval[i]:
#             if spawn_amounts[i] > min_species_size:
#                 spawn_amounts[i] -= 1
#                 total_spawn -= 1
#             break;
#         last_step = roulete_interval[i]