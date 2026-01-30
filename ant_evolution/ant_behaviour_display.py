import pickle

import numpy as np

from pureples.es_hyperneat.es_hyperneat import ESNetwork
from pureples.shared.substrate import Substrate

from configs.evolution_config import VERSION, VERSION_TEXT, get_input_coordinates, get_output_coordinates, params
import ant_colony_arena as ant_arena


INPUT_COORDINATES = get_input_coordinates()
OUTPUT_COORDINATES = get_output_coordinates()
SUBSTRATE = Substrate(INPUT_COORDINATES, OUTPUT_COORDINATES)

if __name__ == "__main__":
    env = ant_arena.AntColonyEnv(5, 30, render_mode='human')
    env.reset()

    best_agent_file= "alg_runs/run_2026-01-30;16-52-45_rand_ants_no_pheromone_no_obstacles_no_health_ants-5_food-30_gen-200/es_hyperneat_ant_evolution_medium_cppn.pkl"
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
        total_reward = 0
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
            total_reward += sum(rewards.values())
        print(f"Average env fitness: {total_reward / len(agent_nets)}")