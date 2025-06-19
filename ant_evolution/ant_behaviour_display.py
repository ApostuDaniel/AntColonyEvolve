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
from ant_evolution import VERSION, params, SUBSTRATE

# INPUT_COORDINATES = [
#     (0.0, 0.0), # r_norm
#     (0.0, -0.1), # theta_norm
#     (0.0, -0.2), # speed
#     (0.0, -0.3), # theta
#     (-1, -1.), (-0.5, -1.),(0, -1.), (0.5, -1.), (1, -1.), #vision_segments_wall_distances
#     (-1, -0.8), (-0.5, -0.8),(0, -0.8), (0.5, -0.8), (1, -0.8),  # vision_segments_food_distances
#     (-1, -0.7), (-0.5, -0.7),(0, -0.7), (0.5, -0.7), (1, -0.7),  # vision_segments_food_angles
#     # (-0.7, -0.4), (-0.35, -0.4), (0, -0.4), (0.35, -0.4), (0.7, -0.4),  # density_area_close
#     # (-0.8, -0.5), (-0.4, -0.5), (0, -0.5), (0.4, -0.5), (0.8, -0.5),  # density_area_middle
#     # (-0.9, -0.6), (-0.6, -0.6), (0, -0.6), (0.6, -0.6), (0.9, -0.6),  # density_area_far
# ]
#
# OUTPUT_COORDINATES = \
#     [
#         (0, 0.5), # acceleration
#         (-0.5, 0.5), # turn left
#         (0.5, 0.5) #turn_right
#     ]
#
# SUBSTRATE = Substrate(INPUT_COORDINATES, OUTPUT_COORDINATES)

if __name__ == "__main__":
    env = ant_arena.AntColonyEnv(1, 25, render_mode='human')
    env.reset()

    best_agent_file="./alg_runs/run_2025-06-19_16-48-43_1_net_1_env_single/es_hyperneat_ant_evolution_medium_cppn.pkl"
    CPPN = []
    with open(best_agent_file, 'rb') as file:
        CPPN = pickle.load(file)

    networks = [ESNetwork(SUBSTRATE, CPPN, params(VERSION)) for a in env.agents]
    nets = [network.create_phenotype_network() for network in networks]
    agent_nets = dict(zip(env.agents, list(range(len(nets)))))

    while True:
        observations, infos = env.reset()
        terminations = [False]
        truncations = [False]
        while env.agents and not (all(terminations) or any(truncations)):
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
            terminations = [terminations[a] for a in terminations]
            truncations = [truncations[a] for a in truncations]