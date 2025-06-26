import copy
import logging
import os
import pickle
from datetime import datetime

import numpy as np
from neat import StdOutReporter

import ant_colony_arena.ant_colony_arena as ant_arena
import neat

from ant_colony_arena.ant_colony_arena import vision_segments, view_ranges, PHEROMONE_TYPES
from utils.neat_plot_reporter import PlotReporter

from pureples.shared.visualize import draw_net
from pureples.shared.substrate import Substrate
from pureples.es_hyperneat.es_hyperneat import ESNetwork

import concurrent.futures
import multiprocessing
from functools import partial
import itertools

# GLOBAL IN WORKER PROCESS
worker_env = None

# S, M or L; Small, Medium or Large (logic implemented as "Not 'S' or 'M' then Large").
VERSION = "M"
VERSION_TEXT = "small" if VERSION == "S" else "medium" if VERSION == "M" else "large"

#coordinates that we try to allign parallel
INPUT_COORDINATES = [
    # (0., 0.0), # r_norm
    # (0., -0.5), # theta_norm
    (-0.2, -1.0), # speed
    (-0.1, -1.0), # theta
    (0.1, -1.0),  # heartbeat
    (0.2, -1.0),  # travel_distance
    (-0.2, -0.8), # current_energy
    (0.2, -0.8), # carried_food
    (0., -0.8), # ant_in_nest
    (-0.2, -0.6),  # reward_sensor
    (0.2, -0.6),  # penalty_sensor
    # (0.2, -0.35), # distance_to_nest
    # (0.2, -0.35), # angle_to_nest
]

# depending on how many vision segments we use, compute their spacing in the substrate on x-axis
vision_segments_x_coord = [-1. + i * 2./(len(vision_segments) - 1) for i in range(len(vision_segments))]

# provide density areas coordinates
vision_ranges_y_coord = [-0.4 - 0.1 * i for i in range(len(view_ranges))]
# for y in vision_ranges_y_coord:
#     INPUT_COORDINATES.extend([(x, y) for x in vision_segments_x_coord])

# provide pheromones coordinates
for x in vision_segments_x_coord:
    INPUT_COORDINATES.extend([(x, -0.2 - (i * 0.05)) for i in range(PHEROMONE_TYPES)])  # pheromone coordinates for each vision segment

INPUT_COORDINATES.extend([(x, 0.0) for x in vision_segments_x_coord]) #vision_segments_wall_distances
INPUT_COORDINATES.extend([(x, 0.2) for x in vision_segments_x_coord]) #vision_segments_food_distances
INPUT_COORDINATES.extend([(x, 0.3) for x in vision_segments_x_coord]) #vision_segments_food_angles

OUTPUT_COORDINATES = \
    [
        (0., 0.7), # acceleration
        (-0.5, 0.7), # turn left
        (0.5, 0.7), #turn_right
        (0., 0.5) # pheromone type
    ]

SUBSTRATE = Substrate(INPUT_COORDINATES, OUTPUT_COORDINATES)

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

def worker_init(env_fun, agents_in_env):
    """Initialize worker-local environment once"""
    global worker_env
    worker_env = env_fun(agents_in_env)  # Create once per worker

def _run_one_trial(nets, networks, env):
    """
    Helper function to run a single trial using a pre-created environment and networks.

    Args:
        env: Pre-created environment instance
        nets: Pre-built phenotype networks
        networks: Pre-built ESNetwork configurations
    """
    # Reset network states
    for net in nets:
        net.reset()

    # We do this because sometimes len of networks is smaller than len of agents
    observations, infos = env.reset(options={"nr_of_agents": len(nets)})
    terminations = {"placeholder": False}
    truncations = {"placeholder": False}
    total_rewards = [0] * len(nets)
    # When agents are terminated during run, but the association needs to remain the same
    agent_nets = dict(zip(env.agents, list(range(len(nets)))))

    while env.agents and not (all(terminations.values()) or any(truncations.values())):

        outputs = []
        for agent in env.agents:
            net_id = agent_nets[agent]
            net, network = nets[net_id], networks[net_id]
            obs = observations[agent]
            # Multiple activations if configured
            for _ in range(network.activations):
                output = net.activate(np.array(obs))
            outputs.append((agent, output))

        actions = dict(outputs)
        observations, rewards, terminations, truncations, infos = env.step(actions)

        for agent in rewards:
            net_id = agent_nets[agent]
            total_rewards[net_id] += rewards[agent]

    return total_rewards

def _run_one_env(index, phenotype_net, nr_activations):
    """
    Helper function to run a single env with multiple copies of the same network.

    :param index: the index of the genome
    :param net: ANN to be multiplied
    :param network: needed for activation
    :return:
    """
    #Reuse worker-local environment for trials
    global worker_env

    # We do this because sometimes len of networks is smaller than len of agents
    observations, infos = worker_env.reset()
    nr_of_agents = len(worker_env.agents)
    terminations = {"placeholder": False}
    truncations = {"placeholder": False}
    reward = 0

    # Make as many nets as there are agents in the env
    nets = [copy.deepcopy(phenotype_net) for _ in range(nr_of_agents - 1)]
    nets.append(phenotype_net)
    # Reset network states
    for net in nets:
        net.reset()

    # When agents are terminated during run, but the association needs to remain the same
    agent_nets = dict(zip(worker_env.agents, list(range(len(nets)))))

    while worker_env.agents and not (all(terminations.values()) or any(truncations.values())):

        outputs = []
        for agent in worker_env.agents:
            net_id = agent_nets[agent]
            net = nets[net_id]
            obs = observations[agent]
            # Multiple activations if configured
            for _ in range(nr_activations):
                output = net.activate(np.array(obs))
            outputs.append((agent, output))

        actions = dict(outputs)
        observations, rewards, terminations, truncations, infos = worker_env.step(actions)

        reward += sum(rewards.values())

    return index, (reward / nr_of_agents)


def run_es(gens, env_fun, trials, max_steps, config, params, substrate, single_network_env=True, agents_in_env=1, parallel_trials=False, output=True, output_dir=None):
    """
    Generic OpenAI Gym runner for ES-HyperNEAT.
    """

    if single_network_env:
        # create an env for each genome
        envs = [env_fun(agents_in_env) for _ in range(config.pop_size * trials)]
    else:
        # Pre-create environments for all trials
        envs = [env_fun() for _ in range(trials)]

    def eval_fitness_parallel_envs(genomes, config):

        curr_envs = envs[:len(genomes) * trials]

        phenotype_nets = []
        networks = []
        for _, g in genomes:
            cppn = neat.nn.FeedForwardNetwork.create(g, config)
            es_net = ESNetwork(substrate, cppn, params)
            networks.append(es_net)
            net = es_net.create_phenotype_network()
            phenotype_nets.append(net)

        # Prepare trial runner
        env_runner = partial(
            _run_one_env,
        )

        # Run envs in parallel
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=16,
            initializer=worker_init,
            initargs=(env_fun, agents_in_env)
        ) as executor:
            # Map each environment to a net
            futures = []
            for j in range(trials):
                offset = j * len(genomes)
                for i, net in enumerate(phenotype_nets):
                    futures.append(executor.submit(env_runner, i, net, networks[i].activations))
                # futures.extend([executor.submit(env_runner, i, networks[offset + i], curr_envs[offset + i]) for i in range(len(genomes))])
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # sort and group results by genome index, should be 'trials' results with the same index
        #compute average for all trials
        results.sort(key=lambda x: x[0])
        mean_trials_results = [(key, sum(num for _, num in value) / trials) for key, value in itertools.groupby(results, lambda x: x[0])]

        for i, (_, g) in enumerate(genomes):
            g.fitness = mean_trials_results[i][1]

    def eval_fitness(genomes, config):

        # Modified neat/reproduction.py compute_spawn() function to limit spawn_amounts to pop size
        env = envs[0]
        cppns = [neat.nn.FeedForwardNetwork.create(g, config) for _, g in genomes]
        networks = [ESNetwork(substrate, cppn, params) for cppn in cppns]
        nets = [network.create_phenotype_network() for network in networks]

        fitnesses = [[] for _ in genomes]
        for _ in range(trials):
            total_rewards = _run_one_trial(nets, networks, env)
            for i in range(len(total_rewards)):
                fitnesses[i].append(total_rewards[i])

        for i, (_, g) in enumerate(genomes):
            g.fitness = np.array(fitnesses[i]).mean()

    def eval_fitness_parallel_trials(genomes, config):
        # Build all networks once per generation
        cppns = [neat.nn.FeedForwardNetwork.create(g, config) for _, g in genomes]
        networks = [ESNetwork(substrate, cppn, params) for cppn in cppns]
        nets = [network.create_phenotype_network() for network in networks]


        # Prepare trial runner
        trial_runner = partial(
            _run_one_trial,
            nets,
            networks
        )

        # Run trials in parallel
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Map each environment to a trial
            futures = [executor.submit(trial_runner, current_env) for current_env in envs]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # Calculate average fitness
        fitness_matrix = np.array(results)
        avg_fitness = fitness_matrix.mean(axis=0)

        for i, (_, g) in enumerate(genomes):
            g.fitness = avg_fitness[i]

    # Create population and train the network. Return network of winner.
    stats_one = neat.statistics.StatisticsReporter()
    pop = ini_pop(None, stats_one, config, output, output_dir)

    if single_network_env:
        winner_one = pop.run(eval_fitness_parallel_envs, gens)
    elif parallel_trials:
        winner_one = pop.run(eval_fitness_parallel_trials, gens)
    else:
        winner_one = pop.run(eval_fitness, gens)

    for used_env in envs:
        used_env.close()

    return winner_one, (stats_one,)

def run(gens, env_fun, trials, version, single_network_env=True, agents_in_env=1, parallel_trials=False, output_dir=None):
    """
    Run the ant evolution task using the Gym environment
    Returns the winning genome and the statistics of the run.
    """
    winner, stats = run_es(gens, env_fun, trials, 200, CONFIG, params(
        version), SUBSTRATE, single_network_env=single_network_env, agents_in_env=agents_in_env, parallel_trials=parallel_trials, output_dir=output_dir)
    print(f"es_hyperneat_ant_evolution_{VERSION_TEXT} done")
    return winner, stats

def make_ant_env(ants_in_env=1):
    # env = ant_arena.AntColonyEnv(ants_in_env, 13, render_mode='human')
    env = ant_arena.AntColonyEnv(ants_in_env, 1)
    env.reset()
    return env


if __name__ == "__main__":
    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.INFO)

    nr_gens = 40
    run_specificity=f"pheromone_test_1_food_13_ants_no_obstacles_gen{nr_gens}"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    current_run_directory = f"alg_runs/run_{timestamp}_{run_specificity}"

    # Run! Only relevant to look at the winner.
    WINNER = run(nr_gens, make_ant_env, 2, VERSION, True, 13, False, os.path.join(current_run_directory, "plots"))[0]

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