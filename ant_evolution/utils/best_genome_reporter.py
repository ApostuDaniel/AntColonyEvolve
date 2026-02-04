import os
import pickle
from pathlib import Path

import neat
from neat.reporting import BaseReporter
from pureples.es_hyperneat.es_hyperneat import ESNetwork
from pureples.shared.visualize import draw_net
from pureples.shared.visualize import draw_net


class BestGenomeReporter(BaseReporter):

    def __init__(self, save_frequency=5, save_dir=None, save_file_name=None, config=None, substrate=None,
                 parameters=None):
        self.save_frequency = save_frequency
        self.save_dir = save_dir
        self.save_file_name = save_file_name
        self.generation = 0
        # Settings needed from ant_evolution to save the best solution
        self.config = config
        self.substrate = substrate
        self.evolution_parameters = parameters
        self.best_genome = None
        self.best_population = None
        self.best_generation = 0
        self.best_genome_changed = False

    def start_generation(self, generation):
        self.generation = generation

    def end_generation(self, config, population, species_set):
        pass

    def post_evaluate(self, config, population, species, best_genome):
        if self.best_genome is None or best_genome.fitness > self.best_genome.fitness:
            self.best_genome = best_genome
            self.best_population = population
            self.best_generation = self.generation
            self.best_genome_changed = True

        if (self.generation + 1) % self.save_frequency == 0 and self.best_genome_changed:
            self.save_best_genome()
            self.best_genome_changed = False

    def found_solution(self, config, generation, best):
        self.best_genome = best
        self.best_generation = self.generation
        self.save_best_genome()

    def save_best_genome(self):

        os.makedirs(self.save_dir, exist_ok=True)

        # Remove previous files
        for file in Path(self.save_dir).iterdir():
            if file.is_file():
                os.remove(file)

        # Save CPPN if wished to be reused and draw it + winner to file.
        CPPN = neat.nn.FeedForwardNetwork.create(self.best_genome, self.config)
        NETWORK = ESNetwork(self.substrate, CPPN, self.evolution_parameters)
        NET = NETWORK.create_phenotype_network(
            filename=os.path.join(self.save_dir, f"{self.save_file_name}_winner")
        )

        draw_net(
            CPPN,
            filename=os.path.join(self.save_dir, f"{self.save_file_name}_cppn")
        )

        with open(os.path.join(self.save_dir, f"{self.save_file_name}_cppn.pkl"),
                  'wb') as output:
            pickle.dump(CPPN, output, pickle.HIGHEST_PROTOCOL)

        # esnures function does not crush if called from found_solution, since last population is not returned
        if self.best_genome_changed and self.best_population is not None:
            with open(os.path.join(self.save_dir, f"{self.save_file_name}_population.pkl"),
                      'wb') as output:
                pickle.dump(self.best_population, output, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.save_dir, f"{self.save_file_name}_info.txt"),
                  'w') as output:
            output.write(f"Best generation: {self.best_generation}, best fitness: {self.best_genome.fitness}")
