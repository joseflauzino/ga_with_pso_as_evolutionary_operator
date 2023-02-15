import sys
import numpy as np
import matplotlib.pyplot as plt

from ga import GA
from util import *

#########################################################
# Parâmetros GA:
# generations     # Number of generations
# pop_size        # Population size
# cx_prob         # Probability of two parents procreate
# fun             # Function to optimize
# bounds          # Problem boundaries
# mt_prob         # Probability that a bit is flipped over
# pso_mutation    # Toggle PSO-based mutation
# with_inertia

# Parametros PSO:
# global num_dimensions = 3   #
# ga_population               # genetic algorithm population sorted in ascending order based on fitness
# costFunc                    # the function to be optimized (fitness)# the function to be optimized (fitness)
# bounds                      #
# maxiter = 1                 #
# topology = global           #
# inertia = 0.5               #
# constriction = False        #
# with_inertia = True         #

# Parametros das particulas
# position_i           # particle position (x_i)
# velocity_i           # particle velocity (v_i)
# fitness_i            # fitness individual
# best_pos_i           # best position individual (p_i)
# best_fitness_i       # best fitness individual (f_p_i)
# neighbors            # list of other particles ordered by proximity
# best_pos_g           # best position social/group (p_g)
# best_fitness_g       # best fitness social/group (p_g)
# inertia              # particle inertia value
# constriction         # 1 if using constriction factor
# with_inertia

# Update Velocity
# c1 = 2.1
# c2 = 2.1

#########################################################


num_gen = 100
num_exec = 30
x = np.arange(1, num_gen + 1, 1)  # iteration number array
x_zoom = np.arange(20, num_gen, 1)  # iteration number array

ms = 5
lw = 2
capsize = 3
elw = 0.5


def population_size(function_name, function, bounds):
    print('Optimizing the population_size', function_name, 'function...')
    ga_results = []
    ga_results_mean = []
    ga_particle_number = []
    ga_results_best_fitness = []

    ga_pso_results = []
    ga_pso_results_mean = []
    ga_pso_particle_number = []
    ga_pso_results_best_fitness = []
    distance_exec = []

    # varying the number of chromosomes
    for n in range(30, 101, 10):
        best_fitness_t = np.inf
        ga_pso = GA(function, bounds, pop_size=n, mt_prob=0.25,
                    pso_mutation=True, with_inertia=True)
        partial_results = []
        # executing the algorithm 30 times
        for i in range(num_exec):
            execution, best_fitness, chromosome_list = ga_pso.run()
            partial_results.append(execution)
            best_fitness.sort()
            if best_fitness_t > best_fitness[0]:
                best_fitness_t = best_fitness[0]
            if n == 50:
                distance_exec.append(
                    calculate_population_distance(chromosome_list))

        ga_pso_particle_number.append(n)
        ga_pso_results.append(partial_results)
        ga_pso_results_mean.append(np.mean(partial_results))
        ga_pso_results_best_fitness.append(best_fitness_t)

    for n in range(30, 101, 10):
        best_fitness_t = np.inf
        ga = GA(function, bounds, pop_size=n, mt_prob=0.25)
        partial_results = []
        # executing the algorithm 30 times
        for i in range(num_exec):
            execution, best_fitness, chromosome_list = ga.run()
            partial_results.append(execution)
            best_fitness.sort()
            if best_fitness_t > best_fitness[0]:
                best_fitness_t = best_fitness[0]

        ga_particle_number.append(n)
        ga_results.append(partial_results)
        ga_results_mean.append(np.mean(partial_results))
        ga_results_best_fitness.append(best_fitness_t)

    # ploting
    print('Plotting Fitness vs Population Size...')
    plt.figure(figsize=(10, 7))
    plt.plot(ga_pso_results_mean, 'ro')
    plt.plot(ga_pso_results_best_fitness, 'b+')
    plt.plot(ga_results_mean, 'go')
    plt.plot(ga_results_best_fitness, 'm+')
    plt.xticks(range(1, len(ga_pso_results) + 1), ga_pso_particle_number)
    max_y = max(ga_pso_results_mean)
    step = (max_y/10)
    plt.yticks(np.arange(0, max_y+step, step=step))
    plt.title(function_name)
    plt.ylabel("Fitness")
    plt.xlabel("Population Size")
    plt.legend(['Fitness Média GA-PSO', 'Melhor Fitness GA-PSO',
               'Fitness Média GA', 'Melhor Fitness GA'], loc=0)
    plt.savefig("imgs/fitness_vs_population_size(" + function_name + ").png")

    print('Plotting Average Distance over Generations GA-PSO...')
    result_distance = []
    for gen in range(100):
        aux_result_distance = []
        for list_avg in distance_exec:
            aux_result_distance.append(list_avg[gen])
        # calcula a media da geração aqui
        result_distance.append(mean(aux_result_distance))
    plt.figure(figsize=(10, 7))
    plt.plot(result_distance, 'ro')
    plt.title(function_name)
    plt.ylabel("Average Distance")
    plt.xlabel("Generation")
    plt.yticks(np.arange(0, 110, step=10))
    plt.savefig(
        "imgs/average_distance_over_generations(" + function_name + ").png")
    print('')


def mutation(function_name, function, bounds):
    print('Optimizing the mutation', function_name, 'function...')

    best_fitnesses_n = []
    gen_results_n = [[] for i in range(num_gen)]
    gen_mean_n = []
    gen_std_n = []

    best_fitnesses_m = []
    gen_results_m = [[] for i in range(num_gen)]
    gen_mean_m = []
    gen_std_m = []

    best_fitnesses_pso_m = []
    gen_results_pso_m = [[] for i in range(num_gen)]
    gen_mean_pso_m = []
    gen_std_pso_m = []

    ga_no_mutation = GA(function, bounds, generations=num_gen)
    ga_mutation = GA(function, bounds, generations=num_gen, mt_prob=0.25)
    ga_pso_mutation = GA(function, bounds, generations=num_gen, mt_prob=0.25,
                         pso_mutation=True, with_inertia=True)

    for i in range(num_exec):
        _, execution_best_fitnesses_n, _ = ga_no_mutation.run()
        best_fitnesses_n.append(execution_best_fitnesses_n)

    for i in range(num_exec):
        _, execution_best_fitnesses_m, _ = ga_mutation.run()
        best_fitnesses_m.append(execution_best_fitnesses_m)

    for i in range(num_exec):
        _, execution_best_fitnesses_pso_m, _ = ga_pso_mutation.run()
        best_fitnesses_pso_m.append(execution_best_fitnesses_pso_m)

    # grouping results by iteration number
    for i in range(len(best_fitnesses_m)):
        for j in range(len(best_fitnesses_m[i])):
            gen_results_m[j].append(best_fitnesses_m[i][j])
            gen_results_n[j].append(best_fitnesses_n[i][j])
            gen_results_pso_m[j].append(best_fitnesses_pso_m[i][j])

    # calculating mean and std values by iteration number
    for i in range(len(gen_results_m)):
        gen_mean_m.append(np.mean(gen_results_m[i]))
        gen_std_m.append(np.std(gen_results_m[i]))

        gen_mean_n.append(np.mean(gen_results_n[i]))
        gen_std_n.append(np.std(gen_results_n[i]))

        gen_mean_pso_m.append(np.mean(gen_results_pso_m[i]))
        gen_std_pso_m.append(np.std(gen_results_pso_m[i]))

    upper_limit = max(gen_mean_m[0], gen_mean_n[0], gen_mean_pso_m[0]) + \
        max(gen_std_m[0], gen_std_n[0], gen_std_pso_m[0]) + 0.2

    plt.figure(figsize=(16, 9))
    plt.ylim(0, upper_limit)
    plt.errorbar(x, np.array(gen_mean_n), np.array(gen_std_n), ms=ms, lw=lw,
                 marker="o", capsize=capsize, elinewidth=elw, label="No Mutation")
    plt.errorbar(x, np.array(gen_mean_m), np.array(gen_std_m), ms=ms, ls="--", lw=lw, marker="o",
                 capsize=capsize, ecolor="red", color="r", elinewidth=elw, label="Mutation", alpha=0.8)
    plt.errorbar(x, np.array(gen_mean_pso_m), np.array(gen_std_pso_m), ms=ms, ls=":", lw=lw, marker="o",
                 capsize=capsize, ecolor="green", color="g", elinewidth=elw, label=" PSO-Mutation", alpha=0.8)
    plt.title(f'{function_name} - No Mutation X Mutation')
    plt.ylabel("Fitness")
    plt.xlabel("Generation")
    plt.legend(loc='upper right', prop={'size': 16})
    plt.savefig(
        "imgs/mutation(" + function_name + ").png")


def topology(function_name, function, bounds):
    print('Optimizing the topology', function_name, 'function...')

    best_fitnesses_g = []
    best_fitnesses_l = []
    gen_results_g = [[] for i in range(num_gen)]
    gen_results_l = [[] for i in range(num_gen)]
    gen_mean_g = []
    gen_mean_l = []
    gen_std_g = []
    gen_std_l = []

    # executing global topology algorithm 30 times
    ga_pso_g = GA(function, bounds, generations=num_gen, mt_prob=0.25,
                  pso_mutation=True, with_inertia=True)

    for i in range(num_exec):
        _, execution_best_fitnesses_g, _ = ga_pso_g.run()
        best_fitnesses_g.append(execution_best_fitnesses_g)

    # executing local topology algorithm 30 times
    ga_pso__l = GA(function, bounds, generations=num_gen, mt_prob=0.25,
                   pso_mutation=True, with_inertia=True, topology='ring')

    for i in range(num_exec):
        _, execution_best_fitnesses_l, _ = ga_pso__l.run()
        best_fitnesses_l.append(execution_best_fitnesses_l)

    # grouping results by iteration number
    for i in range(len(best_fitnesses_g)):
        for j in range(len(best_fitnesses_g[i])):
            gen_results_g[j].append(best_fitnesses_g[i][j])
            gen_results_l[j].append(best_fitnesses_l[i][j])

    # calculating mean and std values by iteration number
    for i in range(len(gen_results_g)):
        gen_mean_g.append(np.mean(gen_results_g[i]))
        gen_std_g.append(np.std(gen_results_g[i]))

        gen_mean_l.append(np.mean(gen_results_l[i]))
        gen_std_l.append(np.std(gen_results_l[i]))

    upper_limit = max(gen_mean_g[0], gen_mean_l[0]) + \
        max(gen_std_g[0], gen_std_l[0]) + 0.2

    plt.figure(figsize=(16, 9))
    plt.ylim(-0.2, upper_limit)
    plt.errorbar(x, np.array(gen_mean_g), np.array(gen_std_g), ms=ms, lw=lw, marker="o",
                 capsize=capsize, ecolor="blue", elinewidth=elw, label='Global')
    plt.errorbar(x, np.array(gen_mean_l), np.array(gen_std_l), ms=ms, lw=lw, ls='--',
                 marker="s", capsize=capsize, color="r", ecolor="r", elinewidth=elw, label='Local', alpha=0.7)
    plt.title(f'{function_name} - Global x Local')
    plt.ylabel("Fitness")
    plt.xlabel("Generation")
    plt.legend(loc='upper right', prop={'size': 16})

    plt.savefig(
        "imgs/topology(" + function_name + ").png")
    

def social(function_name, function, bounds):
    print('Optimizing the social', function_name, 'function...')

    best_fitnesses_inertia_false = []
    best_fitnesses_inertia_true = []
    best_fitnesses_ga = []
    gen_results_inertia_false = [[] for i in range(num_gen)]
    gen_results_inertia_true = [[] for i in range(num_gen)]
    gen_results_ga= [[] for i in range(num_gen)]
    gen_mean_inertia_false = []
    gen_mean_inertia_true = []
    gen_mean_ga = []
    gen_std_inertia_false = []
    gen_std_inertia_true = []
    gen_std_ga = []

    # executing global topology algorithm 30 times
    ga_pso_inertia_false = GA(function, bounds, generations=num_gen, mt_prob=0.25,
                  pso_mutation=True, with_inertia=False, max_iter=1)

    for i in range(num_exec):
        _, execution_best_fitnesses_inertia_false, _ = ga_pso_inertia_false.run()
        best_fitnesses_inertia_false.append(execution_best_fitnesses_inertia_false)

    # executing local topology algorithm 30 times
    ga_pso_inertia_true = GA(function, bounds, generations=num_gen, mt_prob=0.25,
                   pso_mutation=True, with_inertia=True, max_iter=10)

    for i in range(num_exec):
        _, execution_best_fitnesses_inertia_true, _ = ga_pso_inertia_true.run()
        best_fitnesses_inertia_true.append(execution_best_fitnesses_inertia_true)

    # executing just genetic algorithm
    ga = GA(function, bounds, generations=num_gen, mt_prob=0.25)

    for i in range(num_exec):
        _, execution_best_fitnesses_ga, _ = ga.run()
        best_fitnesses_ga.append(execution_best_fitnesses_ga)

    # grouping results by iteration number
    for i in range(len(best_fitnesses_inertia_false)):
        for j in range(len(best_fitnesses_inertia_false[i])):
            gen_results_inertia_false[j].append(best_fitnesses_inertia_false[i][j])
            gen_results_inertia_true[j].append(best_fitnesses_inertia_true[i][j])
            gen_results_ga[j].append(best_fitnesses_ga[i][j])

    # calculating mean and std values by iteration number
    for i in range(len(gen_results_inertia_false)):
        gen_mean_inertia_false.append(np.mean(gen_results_inertia_false[i]))
        gen_std_inertia_false.append(np.std(gen_results_inertia_false[i]))

        gen_mean_inertia_true.append(np.mean(gen_results_inertia_true[i]))
        gen_std_inertia_true.append(np.std(gen_results_inertia_true[i]))

        gen_mean_ga.append(np.mean(gen_results_ga[i]))
        gen_std_ga.append(np.std(gen_results_ga[i]))

    upper_limit = max(gen_mean_inertia_false[0], gen_mean_inertia_true[0], gen_mean_ga[0]) + \
        max(gen_std_inertia_false[0], gen_std_inertia_true[0], gen_std_ga[0]) + 0.2

    plt.figure(figsize=(16, 9))
    plt.ylim(-0.2, upper_limit)
    plt.errorbar(x, np.array(gen_mean_inertia_false), np.array(gen_std_inertia_false), ms=ms, lw=lw, marker="o",
                 capsize=capsize, ecolor="blue", elinewidth=elw, label='Ignore Social')
    plt.errorbar(x, np.array(gen_mean_inertia_true), np.array(gen_std_inertia_true), ms=ms, lw=lw, ls='--',
                 marker="s", capsize=capsize, color="r", ecolor="r", elinewidth=elw, label='With Social', alpha=0.7)
    plt.errorbar(x, np.array(gen_mean_ga), np.array(gen_std_ga), ms=ms, lw=lw, ls='--',
                 marker="*", capsize=capsize, color="g", ecolor="g", elinewidth=elw, label='Genetic Algorithm', alpha=0.7)
    plt.title(f'{function_name} - Global x Local')
    plt.ylabel("Fitness")
    plt.xlabel("Generation")
    plt.legend(loc='upper right', prop={'size': 16})

    plt.savefig(
        "imgs/social-complete(" + function_name + ").png")
    
    #zoom
    N = 20
    gen_mean_inertia_false_small = gen_mean_inertia_false[N:]
    gen_mean_inertia_true_small = gen_mean_inertia_true[N:]
    gen_mean_ga_small = gen_mean_ga[N:]

    gen_std_inertia_false_small = gen_std_inertia_false[N:]
    gen_std_inertia_true_small = gen_std_inertia_true[N:]
    gen_std_ga_small = gen_std_ga[N:]

    upper_limit = max(gen_mean_inertia_false[N], gen_mean_inertia_true[N], gen_mean_ga[N]) + \
        max(gen_std_inertia_false[N], gen_std_inertia_true[N], gen_std_ga[N])
    
    plt.figure(figsize=(16, 9))
    plt.ylim(0, upper_limit)
    plt.errorbar(x_zoom, np.array(gen_mean_inertia_false_small), np.array(gen_std_inertia_false_small), ms=ms, lw=lw, marker="o",
                 capsize=capsize, ecolor="blue", elinewidth=elw, label='Ignore Social')
    plt.errorbar(x_zoom, np.array(gen_mean_inertia_true_small), np.array(gen_std_inertia_true_small), ms=ms, lw=lw, ls='--',
                 marker="s", capsize=capsize, color="r", ecolor="r", elinewidth=elw, label='With Social', alpha=0.7)
    plt.errorbar(x_zoom, np.array(gen_mean_ga_small), np.array(gen_std_ga_small), ms=ms, lw=lw, ls='--',
                 marker="*", capsize=capsize, color="g", ecolor="g", elinewidth=elw, label='Genetic Algorithm', alpha=0.7)
    plt.title(f'{function_name} - Global x Local')
    plt.ylabel("Fitness")
    plt.xlabel("Generation")
    plt.legend(loc='upper right', prop={'size': 16})

    plt.savefig(
        "imgs/social-zoom(" + function_name + ").png")


def main(function_name, function, bounds):
    #population_size(function_name, function, bounds)
    #mutation(function_name, function, bounds)
    #topology(function_name, function, bounds)
    social(function_name, function, bounds)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python main.py <function>")
    function_name, function, bounds = get_function_and_bounds(sys.argv[1])
    main(function_name, function, bounds)
