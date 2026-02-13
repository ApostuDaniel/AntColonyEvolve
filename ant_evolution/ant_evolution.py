import copy
import logging
import os
import pickle
import time
import shutil
from datetime import datetime

import numpy as np
from neat import StdOutReporter
from neat.reporting import ReporterSet

import ant_colony_arena as ant_arena
import neat

from utils.neat_plot_reporter import PlotReporter
from utils.best_genome_reporter import BestGenomeReporter

from pureples.shared.visualize import draw_net
from pureples.shared.substrate import Substrate
from pureples.es_hyperneat.es_hyperneat import ESNetwork

import concurrent.futures
from functools import partial
import itertools

from configs.evolution_config import *
from ant_behaviour_display import display_learning_continuously

# GLOBAL IN WORKER PROCESS
worker_env = None
worker_seed = 6

INPUT_COORDINATES = get_input_coordinates()
OUTPUT_COORDINATES = get_output_coordinates()
SUBSTRATE = Substrate(INPUT_COORDINATES, OUTPUT_COORDINATES)


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


def worker_init(env_fun, agents_in_env, food_in_env, seed=6):
    """Initialize worker-local environment once"""
    global worker_env
    global worker_seed
    worker_seed = seed
    worker_env = env_fun(agents_in_env, food_in_env, seed=seed)  # Create once per worker


def _run_one_env(index, phenotype_net, nr_activations, trial=0):
    """
    Helper function to run a single env with multiple copies of the same network.

    :param index: the index of the genome
    :param net: ANN to be multiplied
    :param network: needed for activation
    :return:
    """
    # Reuse worker-local environment for trials
    global worker_env
    global worker_seed

    trial_seed = worker_seed + trial * 100

    # We do this because sometimes len of networks is smaller than len of agents
    observations, infos = worker_env.reset(seed=trial_seed)
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


def run_es(gens, env_fun, trials, config, parameters, substrate, agents_in_env=1,
           food_per_env=1, output_statistics=True, output_dir=None, winner_dir=None, winner_file_name=None,
           load_population_path=None):
    """
    Generic OpenAI Gym runner for ES-HyperNEAT.
    """
    def eval_fitness_parallel_environments(genomes, config):
        phenotype_nets = []
        networks = []

        for _, g in genomes:
            cppn = neat.nn.FeedForwardNetwork.create(g, config)
            es_net = ESNetwork(substrate, cppn, parameters)
            networks.append(es_net)
            net = es_net.create_phenotype_network()
            phenotype_nets.append(net)

        generation_seed = int(time.time())
        # generation_seed = 8

        cpu_count = os.cpu_count()

        # Prepare trial runner
        env_runner = partial(
            _run_one_env,
        )

        # Run envs in parallel
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=cpu_count,
                initializer=worker_init,
                initargs=(env_fun, agents_in_env, food_per_env, generation_seed)
        ) as executor:
            # Map each environment to a net
            futures = [
                executor.submit(env_runner, i, net, networks[i].activations, j)
                for j in range(trials)
                for i, net in enumerate(phenotype_nets)
            ]

            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # sort and group results by genome index, should be 'trials' results with the same index
        # compute average for all trials
        results.sort(key=lambda x: x[0])
        mean_trials_results = [(key, sum(num for _, num in value) / trials) for key, value in
                               itertools.groupby(results, lambda x: x[0])]
        #
        # min_trials_results = [(key, min(num for _, num in value)) for key, value in
        #                        itertools.groupby(results, lambda x: x[0])]

        for i, (_, g) in enumerate(genomes):
            g.fitness = mean_trials_results[i][1]

    # Create population and train the network. Return network of winner.
    state = None

    stats_reporter = neat.statistics.StatisticsReporter()
    best_genome_reporter = BestGenomeReporter(save_dir=winner_dir, save_file_name=winner_file_name, config=config,
                                              substrate=substrate, parameters=parameters)
    if load_population_path:
        with open(load_population_path, 'rb') as file:
            old_pop_state = pickle.load(file)
            old_pop_population, old_pop_species, last_generation = old_pop_state
            # reporter_set = ReporterSet()
            # reporter_set.add(stats_reporter)
            # reporter_set.add(best_genome_reporter)
            # old_pop_species.reporters = reporter_set
            state = (old_pop_population, old_pop_species, last_generation)

    pop = init_pop(state, config, output_statistics, output_dir)
    pop.add_reporter(stats_reporter)
    pop.add_reporter(best_genome_reporter)

    if load_population_path:
        pop.species.reporters = pop.reporters

    winner_one = pop.run(eval_fitness_parallel_environments, gens)

    for reporter in pop.reporters.reporters:
        if type(reporter) is PlotReporter:
            reporter.close()
            pop.reporters.remove(reporter)

    new_population_state = (pop.population, pop.species, pop.generation)

    return winner_one, stats_reporter, new_population_state


def make_ant_env(ants_in_env=1, food_in_env=1, seed=None, render_mode=None):
    env = ant_arena.AntColonyEnv(ants_in_env, food_in_env, seed=seed, render_mode=render_mode)
    return env


if __name__ == "__main__":
    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.INFO)

    nr_gens = 600
    trials = 3
    ants_per_env = 4
    food_per_env = 75

    # load_population_path = "./alg_runs/run_2026-02-12;01-38-04_tanh_angled_substrate_rand_nest_compass_ants-4_food-75_gen-600/es_hyperneat_ant_evolution_small_population.pkl"
    load_population_path = None

    save_population = True
    run_specificity = f"{params(VERSION)['activation']}_5vision_angled_substrate_nest_compass_no_pheromone_ants-{ants_per_env}_food-{food_per_env}_gen-{nr_gens}"

    timestamp = datetime.now().strftime("%Y-%m-%d;%H-%M-%S")

    current_run_directory = f"alg_runs/run_{timestamp}_{run_specificity}"
    winner_file_name = f"es_hyperneat_ant_evolution_{VERSION_TEXT}"

    os.makedirs(current_run_directory, exist_ok=True)
    shutil.copy("configs/config_cppn_ant_evolution", current_run_directory)

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
        output_dir=os.path.join(current_run_directory, "plots"),
        winner_dir=current_run_directory,
        winner_file_name=winner_file_name,
        load_population_path=load_population_path
    )

    print(f"es_hyperneat_ant_evolution_{VERSION_TEXT} done")

    if save_population:
        with open(os.path.join(current_run_directory, f'{winner_file_name}_population.pkl'),
                  'wb') as output:
            pickle.dump(population_state, output, pickle.HIGHEST_PROTOCOL)

    # Save CPPN if wished to be reused and draw it + winner to file.
    CPPN = neat.nn.FeedForwardNetwork.create(winner, CONFIG)

    display_seed = None
    # display_seed = 8
    # after learning has ended, display the best individuals in pygame
    display_env = make_ant_env(ants_per_env, food_per_env, seed=display_seed, render_mode="human")
    display_env.reset(seed=display_seed)

    display_learning_continuously(display_env, CPPN, seed=display_seed)
