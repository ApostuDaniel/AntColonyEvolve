import pickle

import numpy as np

from pureples.es_hyperneat.es_hyperneat import ESNetwork
from pureples.shared.substrate import Substrate

from configs.evolution_config import VERSION, VERSION_TEXT, get_input_coordinates, get_output_coordinates, params
from ant_colony_arena import *

INPUT_COORDINATES = get_input_coordinates()
OUTPUT_COORDINATES = get_output_coordinates()
SUBSTRATE = Substrate(INPUT_COORDINATES, OUTPUT_COORDINATES)


def display_learning_continuously(display_env: AntColonyEnv, evolved_cppn, seed=None):
    networks = [ESNetwork(SUBSTRATE, evolved_cppn, params(VERSION)) for a in display_env.agents]
    nets = [network.create_phenotype_network() for network in networks]
    agent_nets = dict(zip(display_env.agents, list(range(len(nets)))))

    while True:
        observations, infos = display_env.reset(seed=seed)
        terminations = [False]
        truncations = [False]
        total_reward = 0
        while display_env.agents and not (all(terminations) or any(truncations)):
            outputs = []
            for agent in display_env.agents:
                net_id = agent_nets[agent]
                net, network = nets[net_id], networks[net_id]
                obs = observations[agent]
                # Multiple activations if configured
                for _ in range(network.activations):
                    output = net.activate(np.array(obs))
                outputs.append((agent, output))

            actions = dict(outputs)
            observations, rewards, terminations, truncations, infos = display_env.step(actions)
            terminations = [terminations[a] for a in terminations]
            truncations = [truncations[a] for a in truncations]
            total_reward += sum(rewards.values())
        print(f"Average env fitness: {total_reward / len(agent_nets)}")


if __name__ == "__main__":
    env = AntColonyEnv(4, 75, render_mode='human', seed=None)
    env.reset()

    best_agent_file = "alg_runs/run_2026-02-13;12-00-54_tanh_5vision_angled_substrate_rand_no_nest_sensor_ants-4_food-75_gen-600/winner/es_hyperneat_ant_evolution_small_cppn.pkl"
    CPPN = []
    with open(best_agent_file, 'rb') as file:
        CPPN = pickle.load(file)

    display_learning_continuously(env, CPPN)
