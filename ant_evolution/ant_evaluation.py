import pickle

import numpy as np

import os
import json
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict

from pureples.es_hyperneat.es_hyperneat import ESNetwork
from pureples.shared.substrate import Substrate

from configs.evolution_config import VERSION, VERSION_TEXT, get_input_coordinates, get_output_coordinates, params
from configs.arena_config import MAX_STEPS
from ant_colony_arena import *

INPUT_COORDINATES = get_input_coordinates()
OUTPUT_COORDINATES = get_output_coordinates()
SUBSTRATE = Substrate(INPUT_COORDINATES, OUTPUT_COORDINATES)


@dataclass
class EvaluationResult:
    """Stores reward data collected over timesteps across multiple evaluations"""
    algorithm_name: str
    rewards_per_timestep: list[float]  # average reward at each timestep across evaluations
    nr_of_evaluations: int
    max_timesteps: int

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary"""
        return cls(**data)


def evaluate_agent_with_timestep_tracking(
        display_env: AntColonyEnv,
        evolved_cppn,
        nr_of_evaluations: int = 10,
        algorithm_name: str = "Algorithm",
        results_dir: str = None
) -> EvaluationResult:
    """
    Evaluate agent over multiple runs, tracking cumulative rewards at each timestep.

    Args:
        display_env: The environment to evaluate in
        evolved_cppn: The evolved CPPN to create networks from
        nr_of_evaluations: Number of evaluation runs
        algorithm_name: Name of the algorithm for labeling results
        results_dir: Directory to save results (optional)

    Returns:
        EvaluationResult containing rewards averaged across evaluations at each timestep
    """
    networks = [ESNetwork(SUBSTRATE, evolved_cppn, params(VERSION)) for a in display_env.agents]
    nets = [network.create_phenotype_network() for network in networks]
    agent_nets = dict(zip(display_env.agents, list(range(len(nets)))))
    max_timesteps = MAX_STEPS

    # Store rewards at each timestep for each evaluation
    all_evaluations_rewards = []  # List of lists: [evaluation_1_rewards, evaluation_2_rewards, ...]

    for eval_num in range(nr_of_evaluations):
        observations, infos = display_env.reset(seed=eval_num)
        terminations = {agent: False for agent in display_env.agents}
        truncations = {agent: False for agent in display_env.agents}

        timestep_rewards = []  # Track cumulative reward at each timestep
        cumulative_reward = 0
        timestep = 0

        while display_env.agents and timestep < max_timesteps:
            # Check if all agents are done or truncated
            if all(terminations.values()) or any(truncations.values()):
                break

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

            # Accumulate rewards
            step_reward = sum(rewards.values()) / len(agent_nets)
            cumulative_reward += step_reward
            timestep_rewards.append(cumulative_reward)

            timestep += 1

            # Force truncation after max_timesteps
            if timestep >= max_timesteps:
                for agent in display_env.agents:
                    truncations[agent] = True

        all_evaluations_rewards.append(timestep_rewards)
        avg_cumulative_reward = cumulative_reward / len(agent_nets) if agent_nets else 0
        print(f"Evaluation {eval_num + 1}/{nr_of_evaluations} - "
              f"Final cumulative reward: {cumulative_reward:.4f}, "
              f"Timesteps: {timestep}")

    # Average rewards across evaluations at each timestep
    # Pad shorter runs with their final value to align all evaluations
    max_len = max(len(r) for r in all_evaluations_rewards) if all_evaluations_rewards else 0

    padded_rewards = []
    for rewards_list in all_evaluations_rewards:
        if len(rewards_list) < max_timesteps:
            # Pad with the final reward value
            final_reward = rewards_list[-1] if rewards_list else 0
            padded_rewards.append(rewards_list + [final_reward] * (max_timesteps - len(rewards_list)))
        else:
            padded_rewards.append(rewards_list)

    # Average across evaluations at each timestep
    avg_rewards_per_timestep = [
        np.mean([padded_rewards[eval_idx][t] for eval_idx in range(nr_of_evaluations)])
        for t in range(max_len)
    ]

    result = EvaluationResult(
        algorithm_name=algorithm_name,
        rewards_per_timestep=avg_rewards_per_timestep,
        nr_of_evaluations=nr_of_evaluations,
        max_timesteps=max_len
    )

    print(f"\nAverage reward over {nr_of_evaluations} evaluations at final timestep: "
          f"{avg_rewards_per_timestep[-1] if avg_rewards_per_timestep else 0:.4f}")

    # Save results if directory is provided
    if results_dir:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        # Create filename from algorithm name
        filename = results_dir / f"{algorithm_name}_rewards.json"

        with open(filename, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        print(f"Results saved to: {filename}")

    return result


def load_evaluation_results(file_path: str) -> EvaluationResult:
    """Load evaluation results from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return EvaluationResult.from_dict(data)


def plot_multiple_evaluations(
        results_list: list[tuple[str, EvaluationResult]],
        output_path: str = None
):
    """
    Plot multiple evaluation results together like the reference image.

    Args:
        results_list: List of tuples (algorithm_name, EvaluationResult)
        output_path: Path to save the plot (optional)
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 7))

    for algo_name, result in results_list:
        plt.plot(result.rewards_per_timestep, label=algo_name, linewidth=2)

    plt.xlabel("Simulation Timestep", fontsize=12)
    plt.ylabel("Environment fitness", fontsize=12)
    plt.title("Fitness Over Time in semi-clustered environment", fontsize=14)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")

    plt.show()


def evaluate_agent(display_env: AntColonyEnv, evolved_cppn, nr_of_evaluations=10):
    networks = [ESNetwork(SUBSTRATE, evolved_cppn, params(VERSION)) for a in display_env.agents]
    nets = [network.create_phenotype_network() for network in networks]
    agent_nets = dict(zip(display_env.agents, list(range(len(nets)))))

    all_evaluations = []
    for i in range(nr_of_evaluations):
        observations, infos = display_env.reset(seed=i)
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
        all_evaluations.append(total_reward / len(agent_nets))
        print(f"Average env fitness: {total_reward / len(agent_nets)}")
    print(f"Average fitness after {nr_of_evaluations} evaluations: {np.mean(all_evaluations)}")


if __name__ == "__main__":
    env = AntColonyEnv(4, 75, render_mode=None, seed=None)
    env.reset()

    agent_to_be_evaluated = "alg_runs/run_2026-02-12;16-06-12_tanh_5vision_angled_substrate_rand_nest_compass_ants-4_food-75_gen-600/winner/es_hyperneat_ant_evolution_small_cppn.pkl"
    evaluation_results_dir = "evaluation_results/full_sensor_ablation"
    nr_of_evaluations = 30
    CPPN = []
    with open(agent_to_be_evaluated, 'rb') as file:
        CPPN = pickle.load(file)

    # evaluate_agent(env, CPPN, nr_of_evaluations)

    # result = evaluate_agent_with_timestep_tracking(env, CPPN, nr_of_evaluations, "No food density", evaluation_results_dir)

    result_dict = {
        "Full sensor": "evaluation_results/full_sensor_ablation/Full-sensor_rewards.json",
        "No pheromone detection": "evaluation_results/full_sensor_ablation/No pheromone detection_rewards.json",
        "No nest sensor": "evaluation_results/full_sensor_ablation/No nest sensor_rewards.json",
        "No food density": "evaluation_results/full_sensor_ablation/No food density_rewards.json",
    }
    results = []

    for config_name, result_path in result_dict.items():
        try:
            result = load_evaluation_results(result_path)
            results.append((config_name, result))
        except FileNotFoundError:
            print(f"File not found for {config_name}")

    plot_multiple_evaluations(results, "evaluation_results/full_sensor_ablation/comparison_plot.png")

