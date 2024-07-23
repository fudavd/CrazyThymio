#!/usr/bin/env python3
import copy
import os
import sys

import scipy
from matplotlib import pyplot as plt

print('Python %s on %s' % (sys.version, sys.platform))
import numpy as np
from utils.Simulate_swarm_population import simulate_swarm_with_restart_population_split, EnvSettings
from utils.Individual import Individual, thymio_genotype


def search_file_list(rootname, file_name):
    file_list = []
    for root, dirs, files in os.walk(rootname):
        for file in files:
            if file_name in file:
                file_list.append(os.path.join(root, file))
    return file_list


def Ratios():
    n_input = 9
    n_output = 2
    arena = 30
    pop_size = 30
    arena_type = f"circle_{arena}x{arena}"
    simulator_settings = EnvSettings
    simulator_settings['arena_type'] = arena_type
    simulator_settings['objectives'] = ['gradient']
    genotype = thymio_genotype("NN", 9, 2)
    run = 4
    experiment_folder = f"./results/{arena}x{arena}_pop{pop_size}/{run}"
    reservoir_list = search_file_list(experiment_folder, "reservoir.npy")
    n_subs = len(reservoir_list)
    genotype['controller']["params"]['torch'] = False

    swarm_size = 20
    simulation_time = 600
    repetitions = 60

    ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
    for ratio in ratios:
        individuals = []
        for n_sub in range(n_subs):
            genotype['controller']["encoding"] = np.ones(n_output * n_input)
            genotype['morphology']['rgb'] = [2 * n_sub / n_subs, 1 - 2 * n_sub / n_subs, 0]
            # genotype['morphology']['rgb'] = [2 * n_sub / n_subs, 2 * n_sub / n_subs, 2 * n_sub / n_subs]
            sub_swarm = Individual(genotype, 0)
            learner_res_dir = reservoir_list[n_sub]
            sub_swarm.controller.load_geno(learner_res_dir[:-14])
            x = np.load(f"{experiment_folder}/x_best.npy")
            sub_swarm.geno2pheno(x[-1][n_sub * n_input * n_output:(1 + n_sub) * n_input * n_output])
            individuals += [sub_swarm] * int(swarm_size * np.abs(n_sub-ratio))

        print(f"STARTING retest best for subswarm ratio = {ratio} experiment: arena circle")
        fitnesses = simulate_swarm_with_restart_population_split(simulation_time, [individuals]*repetitions,
                                                     headless=True,
                                                     env_params=simulator_settings,
                                                     splits=1)
        print(f"\t Mean fitness: {round(fitnesses.mean(), 3)} \t +- {round(fitnesses.std(), 3)}")
        np.save(f"./results/Validation/ratios/r_1:{ratio}.npy", fitnesses)


def Aligment():
    if os.path.isfile("./results/Validation/alignment.npy"):
        n_input = 9
        n_output = 2
        arena = 30
        pop_size = 30
        arena_type = f"circle_{arena}x{arena}"
        simulator_settings = EnvSettings
        simulator_settings['arena_type'] = arena_type
        simulator_settings['objectives'] = ['alignment_sub', 'alignment', 'gradient', 'gradient_sub']

        genotype = thymio_genotype("NN", 9, 2)
        run = 4
        experiment_folder = f"./results/{arena}x{arena}_pop{pop_size}/{run}"
        reservoir_list = search_file_list(experiment_folder, "reservoir.npy")
        n_subs = len(reservoir_list)
        genotype['controller']["params"]['torch'] = False

        individuals = []
        swarm_size = 20
        for n_sub in range(n_subs):
            genotype['controller']["encoding"] = np.ones(n_output * n_input)
            genotype['morphology']['rgb'] = [2 * n_sub / n_subs, 1 - 2 * n_sub / n_subs, 0]
            # genotype['morphology']['rgb'] = [2 * n_sub / n_subs, 2 * n_sub / n_subs, 2 * n_sub / n_subs]
            sub_swarm = Individual(genotype, 0)
            learner_res_dir = reservoir_list[n_sub]
            sub_swarm.controller.load_geno(learner_res_dir[:-14])
            x = np.load(f"{experiment_folder}/x_best.npy")
            sub_swarm.geno2pheno(x[-1][n_sub * n_input * n_output:(1 + n_sub) * n_input * n_output])
            individuals += [sub_swarm] * int((1-n_sub)*swarm_size)
        simulation_time = 600
        _ = simulate_swarm_with_restart_population_split(simulation_time,
                                                         [individuals],
                                                         headless=True,
                                                         env_params=simulator_settings,
                                                         splits=1)
    else:
        alignment = np.load("./results/Validation/alignment.npy")
        time_stamps = [0, 45, 130, 300, 360, 600]
        time = np.arange(0, len(alignment))*0.1
        # plt.rcParams['text.usetex'] = True
        fig, ax = plt.subplots(2)
        ax[0].plot(time, alignment[:, 3].squeeze(), 'k', label='Swarm', zorder=4)
        ax[0].plot(time, alignment[:, 4].squeeze(), label='reservoir 1', color='g')
        ax[0].plot(time, alignment[:, 5].squeeze(), label='reservoir 2', color='r')
        ax[0].vlines(time_stamps, 0, 0.6, colors=['k']*len(time_stamps), linestyles='dotted')#, [':k']*len(time_stamps) )
        ax[0].set_title('Retest best run: Fitness')
        ax[0].set_ylabel('Performance')
        ax[0].set_ylim(0.1, 0.4)
        ax[0].legend(loc='lower right')
        ax[1].set_title('Retest best run: Alignment')
        ax[1].plot(time, alignment[:, 2].squeeze(), '-k', label='Swarm', zorder=4)
        ax[1].plot(time, alignment[:, 0].squeeze(), label='reservoir 1', color='g')
        ax[1].plot(time, alignment[:, 1].squeeze(), label='reservoir 2', color='r')
        ax[1].vlines(time_stamps, 0, 0.6, colors=['k']*len(time_stamps), linestyles='dotted')#, [':k']*len(time_stamps))
        ax[1].set_xlabel('Time (minutes)')
        ax[1].set_ylabel('Alignment:' + r' $\Phi$')
        ax[1].set_ylim(0, 0.6)
        ax[1].legend(loc='upper right')
        fig.tight_layout()
        plt.show()
        fig.savefig('./results/Validation/align/Retest_best.pdf')
        print(alignment[-1, 4])


def Scalability():
    n_input = 9
    n_output = 2
    arena = 30
    pop_size = 30
    arena_type = f"circle_{arena}x{arena}"
    simulator_settings = EnvSettings
    simulator_settings['arena_type'] = arena_type
    simulator_settings['objectives'] = ['gradient']

    swarm_sizes = [10, 20, 50]
    simulation_time = 600
    repetitions = 60

    best_baseline_run = 1
    experiment_folder = f"./results/{arena}x{arena}_pop{pop_size}/Baseline/{best_baseline_run}"
    for swarm_size in swarm_sizes:
        if os.path.isfile(f"./results/Validation/scalability/baseline_swarm_size_{swarm_size}.npy"):
            print(f"completed baseline experiment swarm size {swarm_size}")
            continue


        genotype = thymio_genotype("NN", 9, 2)
        genotype['controller']["params"]['torch'] = False
        genotype['controller']["encoding"] = np.ones(n_output * n_input)
        swarm = Individual(genotype, 0)
        swarm.controller.load_geno(experiment_folder)
        x = np.load(f"{experiment_folder}/x_best.npy")
        swarm.geno2pheno(x[-1])
        individuals = [swarm] * swarm_size

        print(f"STARTING SCALABILITY baseline for swarm size = {swarm_size} experiment: arena circle")
        fitnesses = simulate_swarm_with_restart_population_split(simulation_time, [individuals]*repetitions,
                                                                 headless=True,
                                                                 env_params=simulator_settings,
                                                                 splits=4,)
        print(f"\t Mean fitness: {round(fitnesses.mean(), 3)} \t +- {round(fitnesses.std(), 3)}")
        np.save(f"./results/Validation/scalability/baseline_swarm_size_{swarm_size}.npy", fitnesses)
        print(f"completed experiment swarm size {swarm_size}")


    best_hebbian_run = 0
    experiment_folder = f"./results/Hebbian/{best_hebbian_run}"
    print("Loading experiment: ", experiment_folder)
    for swarm_size in swarm_sizes:
        individuals = []
        if os.path.isfile(f"./results/Validation/scalability/hebbian_swarm_size_{swarm_size}.npy"):
            print(f"completed heterogeneous experiment {arena}")
            continue
        genotype = thymio_genotype("hNN", n_input, n_output)
        genotype['controller']["params"]['torch'] = False
        genotype['morphology']['rgb'] = [1, 0.5, 0]
        swarm = Individual(genotype, 0)
        swarm.controller.load_geno(experiment_folder)
        x = np.load(experiment_folder + 'x_best.npy')
        swarm.geno2pheno(x[-1])

        for _ in range(swarm_size):
            individuals += [copy.deepcopy(swarm)]

        print(f"STARTING SCALABILITY hebbian for swarm size = {swarm_size} experiment: arena circle")
        fitnesses = simulate_swarm_with_restart_population_split(simulation_time, individuals,
                                                                 headless=True,
                                                                 env_params=simulator_settings,
                                                                 splits=4)
        print(f"\t Mean fitness: {round(fitnesses.mean(), 3)} \t +- {round(fitnesses.std(), 3)}")
        np.save(f"./results/Validation/scalability/hebbian_swarm_size_{swarm_size}.npy", fitnesses)
        print(f"completed experiment swarm size {swarm_size}")

    c_types = ['baseline','hebbian']
    for swarm_size in swarm_sizes:
        comp = []
        for control in c_types:
            data = f"./results/Validation/scalability/{control}_swarm_size_{swarm_size}.npy"
            comp.append(np.load(data))
        print(f'Experiment: {swarm_size}\n\tBest\tvs.\tAdaptive: {scipy.stats.ttest_ind(comp[0], comp[1])}')
        print(np.mean(comp[0]).round(5), np.std(comp[0]).round(5), '\t', np.mean(comp[1]).round(5),
              np.std(comp[1]).round(5))


def Robustness():
    n_input = 9
    n_output = 2
    arena_size = 30
    pop_size = 30
    arenas = [f"bimodal_{arena_size}x{arena_size}",
              f"linear_{arena_size}x{arena_size}",
              f"banana_{arena_size}x{arena_size}",]
    simulator_settings = EnvSettings
    simulator_settings['objectives'] = ['gradient']

    swarm_size = 20
    simulation_time = 600
    repetitions = 60

    best_baseline_run = 1
    experiment_folder = f"./results/Baseline/{best_baseline_run}"
    for arena in arenas:  # Retest best controller
        simulator_settings['arena_type'] = arena
        if os.path.isfile(f"./results/Validation/robustness/baseline_arena_type_{arena}.npy"):
            print(f"completed baseline experiment {arena}")
            continue

        genotype = thymio_genotype("NN", 9, 2)
        genotype['controller']["params"]['torch'] = False
        genotype['controller']["encoding"] = np.ones(n_output * n_input)
        swarm = Individual(genotype, 0)
        swarm.controller.load_geno(experiment_folder)
        x = np.load(f"{experiment_folder}/x_best.npy")
        swarm.geno2pheno(x[-1])
        individuals = [swarm] * swarm_size

        print(f"STARTING ROBUSTNESS baseline for arena: {arena}")
        fitnesses = simulate_swarm_with_restart_population_split(simulation_time, [individuals]*repetitions,
                                                                 headless=True,
                                                                 env_params=simulator_settings,
                                                                 splits=4)
        print(f"\t Mean fitness: {round(fitnesses.mean(), 3)} \t +- {round(fitnesses.std(), 3)}")
        print(f"completed experiment {arena}")
        np.save(f"./results/Validation/robustness/baseline_arena_type_{arena}.npy", fitnesses)

    best_hebbian_run = 0
    experiment_folder = f"./results/Hebbian/{best_hebbian_run}"

    for arena in arenas:  # Retest best controller
        simulator_settings['arena_type'] = arena
        individuals = []
        if os.path.isfile(f"./results/Validation/robustness/hebbian_arena_type_{arena}.npy"):
            print(f"completed heterogeneous experiment {arena}")
            continue
        genotype = thymio_genotype("hNN", n_input, n_output)
        genotype['controller']["params"]['torch'] = False
        genotype['morphology']['rgb'] = [1, 0.5, 0]
        swarm = Individual(genotype, 0)
        swarm.controller.load_geno(experiment_folder)
        x = np.load(experiment_folder + 'x_best.npy')
        swarm.geno2pheno(x[-1])

        for _ in range(swarm_size):
            individuals += [copy.deepcopy(swarm)]

        print(f"STARTING ROBUSTNESS hebbian for arena: {arena}")
        fitnesses = simulate_swarm_with_restart_population_split(simulation_time, individuals,
                                                                 headless=True,
                                                                 env_params=simulator_settings,
                                                                 splits=4)
        print(f"\t Mean fitness: {round(fitnesses.mean(), 3)} \t +- {round(fitnesses.std(), 3)}")
        print(f"completed experiment {arena}")
        np.save(f"./results/Validation/robustness/hebbian_arena_type_{arena}.npy", fitnesses)

    c_types = ['baseline', 'hebbian']
    for arena in arenas:
        comp = []
        for control in c_types:
            data = f"./results/Validation/robustness/{control}_arena_type_{arena}.npy"
            comp.append(np.load(data))
        print(f'Experiment: {arena}\n\tBest\tvs.\tAdaptive: {scipy.stats.ttest_ind(comp[0], comp[1])}')
        print(np.mean(comp[0]).round(5), np.std(comp[0]).round(5), '\t', np.mean(comp[1]).round(5),
              np.std(comp[1]).round(5))


if __name__ == '__main__':
    print("STARTING retesting best controller experiments")
    c_types = ['baseline', 'hebbian']
    arenas = [f"bimodal_30x30",
              f"linear_30x30",
              f"banana_30x30", ]
    swarm_sizes = [10, 20, 50]

    for experiment_name in c_types:
        results_dir = os.path.join("./results", experiment_name)
        filenames_fit = search_file_list(results_dir, 'fitnesses.npy')
        best_fitness = -np.inf
        best_genome = None
        best_folder = None
        best_runs = []
        for filename in filenames_fit:
            fitnesses = np.load(filename)
            if fitnesses.max() > best_fitness:
                best_fitness = fitnesses.max()
                best_folder = filename.replace('fitnesses.npy', '')
                best_genome = np.load(best_folder + 'x_best.npy')[-1]

                best_run = int(best_folder.split('/')[-1])
                print(best_folder, best_fitness)
        if best_genome is None:
            print("No best genome found")
            break

    ## Scalability [swarm size] and Robustness [arenas] with best and adaptive ratio
    Scalability()
    Robustness()

    data_best = []
    data_adapt = []
    for swarm_size in swarm_sizes:
        comp = []
        for control in c_types:
            data = f"./results/Validation/scalability/{control}_swarm_size_{swarm_size}.npy"
            comp.append(np.load(data).squeeze().tolist())
        data_best += comp[0]
        data_adapt += comp[1]

    for arena in arenas:
        comp = []
        for control in c_types:
            data = f"./results/Validation/robustness/{control}_arena_type_{arena}.npy"
            comp.append(np.load(data).squeeze().tolist())
        data_best += comp[0]
        data_adapt += comp[1]
    print(f'Experiment: Aggregate\n\tBest\tvs.\tAdaptive: {scipy.stats.ttest_ind(data_best, data_adapt)}')
    print(np.mean(data_best).round(5), np.std(data_best).round(5), '\t', np.mean(data_adapt).round(5),
          np.std(data_adapt).round(5))
    print("FINISHED")
