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
from utils.Fitnesses import Calculate_fitness_size


def search_file_list(rootname, file_name):
    file_list = []
    for root, dirs, files in os.walk(rootname):
        for file in files:
            if file_name in file:
                file_list.append(os.path.join(root, file))
    return file_list


def RetestBest(experiment_folder='./results'):
    n_input = 9
    n_output = 2
    arena = 10
    arena_type = f"circle_{arena}x{arena}"
    simulator_settings = EnvSettings
    simulator_settings['arena_type'] = arena_type
    simulator_settings['record_video'] = True
    # simulator_settings['objectives'] = ['alignment_sub', 'alignment', 'gradient', 'gradient_sub']

    genotype = thymio_genotype("NN", 9, 2)
    genotype['controller']["params"]['torch'] = False

    swarm_size = 10
    reservoir_list = search_file_list(experiment_folder, "reservoir.npy")
    n_subs = len(reservoir_list)
    genotype['controller']["params"]['torch'] = False

    swarm_members = []
    for n_sub in range(n_subs):
        genotype['controller']["encoding"] = np.ones(n_output * n_input)
        genotype['morphology']['rgb'] = [2 * n_sub / n_subs, 1 - 2 * n_sub / n_subs, 0]
        # genotype['morphology']['rgb'] = [2 * n_sub / n_subs, 2 * n_sub / n_subs, 2 * n_sub / n_subs]
        sub_swarm = Individual(genotype, 0)
        learner_res_dir = reservoir_list[n_sub]
        sub_swarm.controller.load_geno(learner_res_dir[:-14])
        x = np.load(f"{experiment_folder}/x_best.npy")
        sub_swarm.geno2pheno(x[-1][n_sub * n_input * n_output:(1 + n_sub) * n_input * n_output])
        swarm_members += [copy.deepcopy(sub_swarm)] * int(swarm_size/n_subs)
    simulation_time = 600
    simulator_settings['fitness_size'] = Calculate_fitness_size(swarm_members, simulator_settings)
    _ = simulate_swarm_with_restart_population_split(simulation_time,
                                                     [swarm_members],
                                                     headless=False,
                                                     env_params=simulator_settings,
                                                     splits=1)

def Ratios(experiment_folder = f"./results/"):
    n_input = 9
    n_output = 2
    arena = 10
    pop_size = 10
    arena_type = f"circle_{arena}x{arena}"
    simulator_settings = EnvSettings
    simulator_settings['arena_type'] = arena_type
    simulator_settings['objectives'] = ['gradient']
    genotype = thymio_genotype("NN", 9, 2)

    reservoir_list = search_file_list(experiment_folder, "reservoir.npy")
    n_subs = len(reservoir_list)
    genotype['controller']["params"]['torch'] = False

    swarm_size = 10
    simulation_time = 600
    repetitions = 60

    ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
    distances = [0.5,]
    for dist in distances:
        simulator_settings['init_distance'] = dist
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
            np.save(f"./results/Validation/ratios/r_{dist}:{ratio}.npy", fitnesses)

if __name__ == '__main__':
    print("STARTING retesting best controller experiments")
    ## Video
    ## Re-test best controller same environment
    Ratios("./Heterogeneous/0")
    RetestBest('./Heterogeneous/0')
    RetestBest('./Baseline/0')

    print("FINISHED")
