import os
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import time
from neat.math_util import mean, stdev
from neat.six_util import itervalues, iterkeys
from neat.reporting import BaseReporter

matplotlib.use('TkAgg')

class PlotReporter(BaseReporter):
    """Uses `print` to output information and plots fitness statistics live."""

    def __init__(self, show_species_detail, save_interval=None, plot_dir="../plots", plot_filename_prefix="fitness_plot"):
        self.show_species_detail = show_species_detail
        self.save_interval = save_interval
        self.plot_filename_prefix = plot_filename_prefix
        self.plot_dir = plot_dir
        self.generation = None
        self.generation_start_time = None
        self.generation_times = []
        self.num_extinctions = 0
        # Data storage for plotting
        self.generation_numbers = []
        self.best_fitness = []
        self.avg_fitness = []
        self.std_fitness = []

        # Initialize plot
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_title('Fitness over Generations')
        self.ax.set_xlabel('Generation')
        self.ax.set_ylabel('Fitness')
        self.best_line, = self.ax.plot([], [], 'b-', label='Best Fitness')
        self.avg_line, = self.ax.plot([], [], 'g-', label='Average Fitness')
        self.fill = None  # Will hold the std deviation fill
        self.ax.grid(alpha=0.3)
        self.ax.legend(loc='best')
        plt.tight_layout()
        # plt.show(block=False)
        plt.pause(0.01)

    def start_generation(self, generation):
        self.generation = generation
        print('\n ****** Running generation {0} ****** \n'.format(generation))
        self.generation_start_time = time.time()
        # Reset data storage at the start of a new run
        if generation == 0:
            self.generation_numbers = []
            self.best_fitness = []
            self.avg_fitness = []
            self.std_fitness = []

    def end_generation(self, config, population, species_set):
        ng = len(population)
        ns = len(species_set.species)
        if self.show_species_detail:
            print('Population of {0:d} members in {1:d} species:'.format(ng, ns))
            sids = list(iterkeys(species_set.species))
            sids.sort()
            print("   ID   age  size  fitness  adj fit  stag")
            print("  ====  ===  ====  =======  =======  ====")
            for sid in sids:
                s = species_set.species[sid]
                a = self.generation - s.created
                n = len(s.members)
                f = "--" if s.fitness is None else "{:.1f}".format(s.fitness)
                af = "--" if s.adjusted_fitness is None else "{:.3f}".format(s.adjusted_fitness)
                st = self.generation - s.last_improved
                print(
                    "  {: >4}  {: >3}  {: >4}  {: >7}  {: >7}  {: >4}".format(sid, a, n, f, af, st))
        else:
            print('Population of {0:d} members in {1:d} species'.format(ng, ns))

        elapsed = time.time() - self.generation_start_time
        self.generation_times.append(elapsed)
        self.generation_times = self.generation_times[-10:]
        average = sum(self.generation_times) / len(self.generation_times)
        print('Total extinctions: {0:d}'.format(self.num_extinctions))
        if len(self.generation_times) > 1:
            print("Generation time: {0:.3f} sec ({1:.3f} average)".format(elapsed, average))
        else:
            print("Generation time: {0:.3f} sec".format(elapsed))

    def post_evaluate(self, config, population, species, best_genome):
        fitnesses = [c.fitness for c in itervalues(population)]
        fit_mean = mean(fitnesses)
        fit_std = stdev(fitnesses)
        best_species_id = species.get_species_id(best_genome.key)
        print('Population\'s average fitness: {0:3.5f} stdev: {1:3.5f}'.format(fit_mean, fit_std))
        print(
            'Best fitness: {0:3.5f} - size: {1!r} - species {2} - id {3}'.format(best_genome.fitness,
                                                                                 best_genome.size(),
                                                                                 best_species_id,
                                                                                 best_genome.key))

        # Store data for plotting
        self.generation_numbers.append(self.generation)
        self.best_fitness.append(best_genome.fitness)
        self.avg_fitness.append(fit_mean)
        self.std_fitness.append(fit_std)

        # Update plot
        self.update_plot()

    def update_plot(self):
        """Update the plot with new fitness data."""
        if not self.generation_numbers:
            return

        # Update best and average fitness lines
        self.best_line.set_data(self.generation_numbers, self.best_fitness)
        self.avg_line.set_data(self.generation_numbers, self.avg_fitness)

        # Update standard deviation fill
        if self.fill:
            self.fill.remove()
        y_lower = [a - s for a, s in zip(self.avg_fitness, self.std_fitness)]
        y_upper = [a + s for a, s in zip(self.avg_fitness, self.std_fitness)]
        self.fill = self.ax.fill_between(
            self.generation_numbers, y_lower, y_upper,
            color='g', alpha=0.1, label='±1 Std. Dev.'
        )

        # Adjust axes limits to fit all data
        self.ax.relim()
        self.ax.autoscale_view()

        # Save plot periodically if configured
        if (self.save_interval and
                self.generation > 0 and
                # generation starts at 0
                (self.generation + 1) % self.save_interval == 0):
            self.save_plot()

        # Redraw the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def save_plot(self):
        """Save the current plot to a file with timestamp."""
        # Create directory if needed
        os.makedirs(self.plot_dir, exist_ok=True)

        # remove all older plots, data is allways updated in the new plot
        last_files = [x for x in Path(self.plot_dir).iterdir()]
        for file in last_files:
            os.remove(file)

        # Generate filename with timestamp and generation number
        timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S")
        filename = os.path.join(self.plot_dir, f"{self.plot_filename_prefix}_{timestamp}_gen_{self.generation}.png")

        # Save as high-resolution PNG
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nSaved plot to {filename}")

    def complete_extinction(self):
        self.num_extinctions += 1
        print('All species extinct.')

    def found_solution(self, config, generation, best):
        print('\nBest individual in generation {0} meets fitness threshold - complexity: {1!r}'.format(
            self.generation, best.size()))

    def species_stagnant(self, sid, species):
        if self.show_species_detail:
            print("\nSpecies {0} with {1} members is stagnated: removing it".format(sid, len(species.members)))

    def info(self, msg):
        print(msg)