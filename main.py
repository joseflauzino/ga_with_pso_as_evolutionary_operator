import sys
import numpy as np
import matplotlib.pyplot as plt

from ga import GA
from util import *


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
        ga = GA(function, bounds, pop_size=n, mt_prob=0.2, pso_mutation=True, with_inertia=True)
        partial_results = []
        # executing the algorithm 30 times
        for i in range(30):
            execution, best_fitness, chromosome_list = ga.run()
            partial_results.append(execution)
            best_fitness.sort()
            if best_fitness_t > best_fitness[0]:
                best_fitness_t = best_fitness[0]
            if n == 50:
                distance_exec.append(calculate_population_distance(chromosome_list))

        particle_number.append(n)
        results.append(partial_results)
        results_mean.append(np.mean(partial_results))
        results_best_fitness.append(best_fitness_t)


    # ploting
    #TODO: calcular as médias por execução aqui
    result_distance = []
    for gen in range(100):
        aux_result_distance = []
        for list_avg in distance_exec:
            aux_result_distance.append(list_avg[gen])
        #calcula a media da geração aqui
        result_distance.append(mean(aux_result_distance))

    print('Plotting Average Distance over Generations...')
    plt.figure(figsize=(10, 7))
    plt.plot(result_distance, 'ro')
    plt.title(function_name)
    plt.ylabel("Average Distance")
    plt.xlabel("Generation")
    plt.yticks(np.arange(0, 110, step=10))
    plt.savefig(
        "imgs/ga_average_distance_over_generations(" + function_name + ").png")
    print('')

    
    print('Plotting Fitness vs Population Size...')
    plt.figure(figsize=(10, 7))
    plt.plot(results_mean, 'ro')
    plt.plot(results_best_fitness, 'b^')
    plt.xticks(range(1, len(results) + 1), particle_number)
    plt.yticks(np.arange(0, 0.1, step=0.05))
    plt.title(function_name)
    plt.ylabel("Fitness")
    plt.xlabel("Population Size")
    plt.legend(['Fitness Média', 'Melhor Fitness'], loc=0)
    plt.savefig("imgs/ga_fitness_vs_population_size(" + function_name + ").png")





if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python main.py <function>")
    function_name, function, bounds = get_function_and_bounds(sys.argv[1])
    main(function_name, function, bounds)
