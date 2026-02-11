import os
import sys
from datetime import datetime
from pathlib import Path
import tkinter as tk
import matplotlib
import matplotlib.pyplot as plt
import time

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from neat.math_util import mean, stdev
from neat.six_util import itervalues, iterkeys
from neat.reporting import BaseReporter

matplotlib.use('TkAgg')


class PlotReporter(BaseReporter):
    """Uses `print` to output information and plots fitness statistics live. Saves best genome"""

    def __init__(self, show_species_detail, plot_save_frequency=10, plot_update_frequency=5, plot_dir="../plots",
                 plot_filename_prefix="fitness_plot"):
        self.log_file_path = os.path.join(plot_dir, "log.txt")
        os.makedirs(plot_dir, exist_ok=True)
        self.show_species_detail = show_species_detail
        self.plot_save_frequency = plot_save_frequency
        self.plot_update_frequency = plot_update_frequency
        self.data_updates_from_last_save = 0
        self.plot_filename_prefix = plot_filename_prefix
        self.plot_dir = plot_dir
        self.generation = None
        self.generation_start_time = None
        self.generation_times = []
        self.num_extinctions = 0

        # Data storage for population fitness plot
        self.generation_numbers = []
        self.best_fitness = []
        self.avg_fitness = []
        self.std_fitness = []

        # Data storage for species fitness plot
        self.species_best_fitness = {}  # species_id -> list of (generation, fitness)
        self.species_lines = {}  # species_id -> line object

        # Initialize population fitness plot
        plt.ion()

        # Create the main application window
        self.root = tk.Tk()
        self.root.title("ESHyperneat Plots")
        self.root.geometry("2000x850")

        # Create frames for each plot
        frame1 = tk.Frame(self.root)
        frame1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        frame2 = tk.Frame(self.root)
        frame2.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Create figures and embed them in Tkinter frames
        self.fig = plt.Figure(figsize=(8, 5), dpi=110)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title('Population Fitness over Generations')
        self.ax.set_xlabel('Generation')
        self.ax.set_ylabel('Fitness')
        self.best_line, = self.ax.plot([], [], 'b-', label='Best Fitness')
        self.avg_line, = self.ax.plot([], [], 'g-', label='Average Fitness')
        self.fill = None  # Will hold the std deviation fill
        self.ax.grid(alpha=0.3)
        self.ax.legend(loc='best')
        self.fig.tight_layout()

        self.fig2 = plt.Figure(figsize=(8, 5), dpi=110)
        self.ax2 = self.fig2.add_subplot(111)
        self.ax2.set_title('Species Best Fitness over Generations')
        self.ax2.set_xlabel('Generation')
        self.ax2.set_ylabel('Fitness')
        self.ax2.grid(alpha=0.3)
        self.fig2.tight_layout()

        # Embed plots in Tkinter frames
        self.canvas1 = FigureCanvasTkAgg(self.fig, master=frame1)
        self.canvas1.draw()
        self.canvas1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=frame2)
        self.canvas2.draw()
        self.canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Start the Tkinter event loop in a non-blocking way
        self.root.update()

    def _write_info(self, output):
        output_str = '\n'.join(output) + '\n'
        sys.stdout.write(output_str)
        with open(self.log_file_path, "a") as log:
            log.writelines(output_str)

    def start_generation(self, generation):
        self.generation = generation
        self._write_info(['\n ****** Running generation {0} ****** \n'.format(generation)])
        self.generation_start_time = time.time()
        # Reset data storage at the start of a new run
        if generation == 0:
            self.generation_numbers = []
            self.best_fitness = []
            self.avg_fitness = []
            self.std_fitness = []
            self.species_best_fitness = {}
            self.species_lines = {}

    def end_generation(self, config, population, species_set):
        ng = len(population)
        ns = len(species_set.species)
        output = []
        if self.show_species_detail:
            output.append('Population of {0:d} members in {1:d} species:'.format(ng, ns))
            sids = list(iterkeys(species_set.species))
            sids.sort()
            output.append("   ID   age  size  fitness  adj fit  stag")
            output.append("  ====  ===  ====  =======  =======  ====")
            for sid in sids:
                s = species_set.species[sid]
                a = self.generation - s.created
                n = len(s.members)
                f = "--" if s.fitness is None else "{:.1f}".format(s.fitness)
                af = "--" if s.adjusted_fitness is None else "{:.3f}".format(s.adjusted_fitness)
                st = self.generation - s.last_improved
                output.append("  {: >4}  {: >3}  {: >4}  {: >7}  {: >7}  {: >4}".format(sid, a, n, f, af, st))
        else:
            output.append('Population of {0:d} members in {1:d} species'.format(ng, ns))

        elapsed = time.time() - self.generation_start_time
        self.generation_times.append(elapsed)
        self.generation_times = self.generation_times[-10:]
        average = sum(self.generation_times) / len(self.generation_times)
        output.append('Total extinctions: {0:d}'.format(self.num_extinctions))
        if len(self.generation_times) > 1:
            output.append("Generation time: {0:.3f} sec ({1:.3f} average)".format(elapsed, average))
        else:
            output.append("Generation time: {0:.3f} sec".format(elapsed))
        self._write_info(output)


    def post_evaluate(self, config, population, species, best_genome):

        fitnesses = [c.fitness for c in itervalues(population)]
        fit_mean = mean(fitnesses)
        fit_std = stdev(fitnesses)
        best_species_id = species.get_species_id(best_genome.key)
        output = []
        output.append('Population\'s average fitness: {0:3.5f} stdev: {1:3.5f}'.format(fit_mean, fit_std))
        output.append('Best fitness: {0:3.5f} - size: {1!r} - species {2} - id {3}'.format(best_genome.fitness,
                                                                                 best_genome.size(),
                                                                                 best_species_id,
                                                                                 best_genome.key))

        self._write_info(output)
        # Store data for population fitness plot
        self.generation_numbers.append(self.generation)
        self.best_fitness.append(best_genome.fitness)
        self.avg_fitness.append(fit_mean)
        self.std_fitness.append(fit_std)

        # Store species fitness data using species.fitness attribute
        for sid, s in species.species.items():
            if s.fitness is not None:  # Only record if fitness is available
                if sid not in self.species_best_fitness:
                    self.species_best_fitness[sid] = []
                self.species_best_fitness[sid].append((self.generation, s.fitness))

        self.data_updates_from_last_save += 1

        if (self.generation + 1) % self.plot_update_frequency == 0:
            # Update plots
            self.update_plots()


    def update_plots(self):
        """Update both population and species fitness plots."""
        if not self.generation_numbers:
            return

        # Update population fitness plot
        self.best_line.set_data(self.generation_numbers, self.best_fitness)
        self.avg_line.set_data(self.generation_numbers, self.avg_fitness)

        if self.fill:
            self.fill.remove()
        y_lower = [a - s for a, s in zip(self.avg_fitness, self.std_fitness)]
        y_upper = [a + s for a, s in zip(self.avg_fitness, self.std_fitness)]
        self.fill = self.ax.fill_between(
            self.generation_numbers, y_lower, y_upper,
            color='g', alpha=0.1, label='±1 Std. Dev.'
        )
        self.ax.relim()
        self.ax.autoscale_view()

        # Update species fitness plot
        for sid, data in self.species_best_fitness.items():
            gens = [d[0] for d in data]
            fits = [d[1] for d in data]

            if sid in self.species_lines:
                self.species_lines[sid].set_data(gens, fits)
            else:
                # Create new line with random color
                line, = self.ax2.plot(gens, fits, label=f'Species {sid}')
                self.species_lines[sid] = line

        # Update legend for species plot
        handles, labels = self.ax2.get_legend_handles_labels()
        if handles:
            self.ax2.legend(handles, labels, loc='best')

        self.ax2.relim()
        self.ax2.autoscale_view()

        # Save plots periodically
        if self.data_updates_from_last_save >= self.plot_save_frequency:
            self.save_plots()
            self.data_updates_from_last_save = 0

        # Redraw both canvases
        self.canvas1.draw()
        self.canvas2.draw()
        self.root.update()

    def save_plots(self):
        """Save both population and species fitness plots."""
        os.makedirs(self.plot_dir, exist_ok=True)

        # Remove previous plots
        for file in Path(self.plot_dir).iterdir():
            if file.is_file() and file.suffix in ['.png', '.jpg', '.pdf']:
                os.remove(file)

        timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S")

        # Save population fitness plot
        pop_filename = os.path.join(
            self.plot_dir,
            f"{self.plot_filename_prefix}_{timestamp}_gen_{self.generation}.png"
        )
        self.fig.savefig(pop_filename, dpi=300, bbox_inches='tight')
        print(f"Saved population fitness plot to {pop_filename}")

        # Save species fitness plot
        species_filename = os.path.join(
            self.plot_dir,
            f"species_{self.plot_filename_prefix}_{timestamp}_gen_{self.generation}.png"
        )
        self.fig2.savefig(species_filename, dpi=300, bbox_inches='tight')
        print(f"Saved species fitness plot to {species_filename}")

    def complete_extinction(self):
        self.num_extinctions += 1
        self._write_info(['All species extinct.'])

    def found_solution(self, config, generation, best):
        self._write_info(['\nBest individual in generation {0} meets fitness threshold - complexity: {1!r}'.format(
            self.generation, best.size())])

    def species_stagnant(self, sid, species):
        if self.show_species_detail:
            self._write_info(["\nSpecies {0} with {1} members is stagnated: removing it".format(sid, len(species.members))])

    def info(self, msg):
        self._write_info([msg])

    def close(self):
        """Close all figures and release resources to enable pickling"""
        plt.close(self.fig)
        plt.close(self.fig2)

        # If using Tkinter directly
        if hasattr(self, 'root'):
            self.root.destroy()
            self.root.quit()

        # Ensure matplotlib releases resources
        plt.close('all')

        # Clear references to GUI objects
        if hasattr(self, 'fig'):
            del self.fig
        if hasattr(self, 'fig2'):
            del self.fig2
        if hasattr(self, 'root'):
            del self.root
        if hasattr(self, 'canvas1'):
            del self.canvas1
        if hasattr(self, 'canvas2'):
            del self.canvas2