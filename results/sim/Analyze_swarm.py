import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
mpl.rc('image', cmap='viridis')
colors = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

def search_file_list(rootname, file_name):
    file_list = []
    for root, dirs, files in os.walk(rootname):
        if file_name in files:
            file_list.append(os.path.join(root, file_name))
    return file_list


if __name__ == '__main__':
    experiment_folder = f"./"
    conditions = ["Heterogeneous", "Baseline"]

    DATA = []
    best_run = []

    fig, ax = plt.subplots(1, 1, figsize=(6,3))
    for ind, condition in enumerate(conditions):
        file_path = search_file_list(os.path.join(experiment_folder, condition), 'fitnesses.npy')[0]
        mean_fitnesses = []
        max_fitnesses = []
        genomes_std = []
        best_ind = 0
        best_fit = -np.inf

        fit = np.load(file_path)[:100]
        gens = np.load(file_path.replace('fitnesses.npy', 'genomes.npy'))[:100]
        f_max = np.max(fit, axis=1)


        generations = np.arange(0, 100)
        SE95 = np.std(fit, axis=1)/np.sqrt(30)*1.96
        f_mean = np.mean(fit, axis=1)
        ax.scatter(generations, f_max,
                      color=colors[ind], label=f'{condition} Max', zorder=len(conditions)-ind)
        ax.fill_between(generations, f_mean - SE95, f_mean + SE95,
                           color=colors[ind], alpha=.25, zorder=len(conditions)-ind)
        ax.plot(generations, f_mean,
                   color=colors[ind], label=f'{condition} Mean', zorder=len(conditions)-ind)
        print(f_max.max())


    # ax.set_title('Average fitness')
    ax.set_ylabel('Fitness', fontsize=16)
    ax.set_xlabel('Generation', fontsize=16)
    ax.legend(loc='lower left')

    fig.tight_layout()
    plt.show()
    fig.savefig('fitness_curves.pdf')
