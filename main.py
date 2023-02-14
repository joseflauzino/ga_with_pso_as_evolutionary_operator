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

def main(function_name, function, bounds):

    results = []
    results_mean = []
    particle_number = []
    results_best_fitness = []
    distance_exec = []

    print('Optimizing the', function_name, 'function...')
    # varying the number of chromosomes
    for n in range(30, 101, 10):
        best_fitness_t = np.inf
        ga = GA(function, bounds, pop_size=n, mt_prob=0.5,
                pso_mutation=True, with_inertia=True)
        partial_results = []
        # executing the algorithm 30 times
        for i in range(1):
            execution, best_fitness, chromosome_list = ga.run()
            partial_results.append(execution)
            best_fitness.sort()
            if best_fitness_t > best_fitness[0]:
                best_fitness_t = best_fitness[0]
            if n == 50:
                distance_exec.append(
                    calculate_population_distance(chromosome_list))

        particle_number.append(n)
        results.append(partial_results)
        results_mean.append(np.mean(partial_results))
        results_best_fitness.append(best_fitness_t)

    # ploting
    result_distance = []
    for gen in range(100):
        aux_result_distance = []
        for list_avg in distance_exec:
            aux_result_distance.append(list_avg[gen])
        # calcula a media da geração aqui
        result_distance.append(mean(aux_result_distance))

    print('Plotting Average Distance over Generations...')
    plt.figure(figsize=(10, 7))
    plt.plot(result_distance, 'ro')
    plt.title(function_name)
    plt.ylabel("Average Distance")
    plt.xlabel("Generation")
    plt.yticks(np.arange(0, 110, step=10))
    plt.savefig(
        "imgs/average_distance_over_generations(" + function_name + ").png")
    print('')

    print('Plotting Fitness vs Population Size...')
    plt.figure(figsize=(10, 7))
    plt.plot(results_mean, 'ro')
    plt.plot(results_best_fitness, 'b+')
    plt.xticks(range(1, len(results) + 1), particle_number)
    max_y = max(results_mean)
    step = (max_y/10)
    plt.yticks(np.arange(0, max_y+step, step=step))
    plt.title(function_name)
    plt.ylabel("Fitness")
    plt.xlabel("Population Size")
    plt.legend(['Fitness Média', 'Melhor Fitness'], loc=0)
    plt.savefig("imgs/fitness_vs_population_size(" + function_name + ").png")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python main.py <function>")
    function_name, function, bounds = get_function_and_bounds(sys.argv[1])
    main(function_name, function, bounds)
