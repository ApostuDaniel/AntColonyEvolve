import copy
import logging
import os
import pickle
import time
from datetime import datetime

import numpy as np
from neat import StdOutReporter

import ant_colony_arena as ant_arena
import neat

from utils.neat_plot_reporter import PlotReporter

from pureples.shared.visualize import draw_net
from pureples.shared.substrate import Substrate
from pureples.es_hyperneat.es_hyperneat import ESNetwork

import concurrent.futures
from functools import partial
import itertools

from configs.evolution_config import *

# GLOBAL IN WORKER PROCESS
worker_env = None
worker_seed = 6

INPUT_COORDINATES = get_input_coordinates()
OUTPUT_COORDINATES = get_output_coordinates()
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
                            'configs/config_cppn_ant_evolution')


def init_pop(state, config, output_statistics, output_dir=None):
    """
    Initialize population attaching statistics reporter.
    """
    pop = neat.population.Population(config, state)
    if output_statistics:
        if output_dir:
            pop.add_reporter(PlotReporter(show_species_detail=True, plot_dir=output_dir))
        else:
            pop.add_reporter(StdOutReporter(True))
    return pop


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


def run_es(gens, env_fun, trials, config, parameters, substrate, agents_in_env=1,
           food_per_env=1, parallel_trials=False, output_statistics=True, output_dir=None, load_population_path=None):
    """
    Generic OpenAI Gym runner for ES-HyperNEAT.
    """
    envs = [env_fun(agents_in_env, food_per_env) for _ in range(trials)]

    def eval_fitness(genomes, config):

        # Modified neat/reproduction.py compute_spawn() function to limit spawn_amounts to pop size
        env = envs[0]
        env.render_mode = "human"
        env.reset()
        cppns = [neat.nn.FeedForwardNetwork.create(g, config) for _, g in genomes]
        networks = [ESNetwork(substrate, cppn, parameters) for cppn in cppns]
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
        networks = [ESNetwork(substrate, cppn, parameters) for cppn in cppns]
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
    state = None
    if load_population_path:
        with open(load_population_path, 'rb') as file:
            old_pop_state = pickle.load(file)
            old_pop_population, old_pop_species, _ = old_pop_state
            state = (old_pop_population, old_pop_species, 0)

    stats_reporter = neat.statistics.StatisticsReporter()

    pop = init_pop(state, config, output_statistics, output_dir)
    pop.add_reporter(stats_reporter)

    if parallel_trials:
        winner_one = pop.run(eval_fitness_parallel_trials, gens)
    else:
        winner_one = pop.run(eval_fitness, gens)

    if envs:
        for used_env in envs:
            used_env.close()

    for reporter in pop.reporters.reporters:
        if type(reporter) is PlotReporter:
            reporter.close()
            pop.reporters.remove(reporter)

    population_state = (pop.population, pop.species, pop.generation)

    return winner_one, stats_reporter, population_state


def make_ant_env(ants_in_env=1, food_in_env=1, seed=None, render_mode=None):
    env = ant_arena.AntColonyEnv(ants_in_env, food_in_env, seed=seed, render_mode=render_mode)
    return env


if __name__ == "__main__":
    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.INFO)

    nr_gens = 100
    trials = 2
    ants_per_env = CONFIG.pop_size
    food_per_env = 12
    parallel_trials = False

    # load_population_path = "./alg_runs/run_2025-06-30_18-14-26_no_pheromone_test_5_food_13_ants_no_obstacles_no_health_deacrease_gen1/es_hyperneat_ant_evolution_medium_population.pkl"
    load_population_path = None

    save_population = True
    run_specificity = f"rand_ants_no_pheromone_no_obstacles_no_health_ants-{ants_per_env}_food-{food_per_env}_gen-{nr_gens}"

    timestamp = datetime.now().strftime("%Y-%m-%d;%H-%M-%S")

    current_run_directory = f"alg_runs_single_colony/run_{timestamp}_{run_specificity}"

    # Run! Only relevant to look at the winner.
    winner, stats, population_state = run_es(
        gens=nr_gens,
        env_fun=make_ant_env,
        trials=trials,
        config=CONFIG,
        parameters=params(VERSION),
        substrate=SUBSTRATE,
        agents_in_env=ants_per_env,
        food_per_env=food_per_env,
        parallel_trials=parallel_trials,
        output_dir=os.path.join(current_run_directory, "plots"),
        load_population_path=load_population_path
    )

    print(f"es_hyperneat_ant_evolution_{VERSION_TEXT} done")

    os.makedirs(current_run_directory, exist_ok=True)
    # Save CPPN if wished to be reused and draw it + winner to file.
    CPPN = neat.nn.FeedForwardNetwork.create(winner, CONFIG)
    NETWORK = ESNetwork(SUBSTRATE, CPPN, params(VERSION))
    NET = NETWORK.create_phenotype_network(
        filename=os.path.join(current_run_directory, f"es_hyperneat_ant_evolution_{VERSION_TEXT}_winner")
    )

    draw_net(
        CPPN,
        filename=os.path.join(current_run_directory, f"es_hyperneat_ant_evolution_{VERSION_TEXT}_cppn")
    )

    with open(os.path.join(current_run_directory, f'es_hyperneat_ant_evolution_{VERSION_TEXT}_cppn.pkl'), 'wb') as output:
        pickle.dump(CPPN, output, pickle.HIGHEST_PROTOCOL)

    if save_population:
        with open(os.path.join(current_run_directory, f'es_hyperneat_ant_evolution_{VERSION_TEXT}_population.pkl'), 'wb') as output:
            pickle.dump(population_state, output, pickle.HIGHEST_PROTOCOL)
