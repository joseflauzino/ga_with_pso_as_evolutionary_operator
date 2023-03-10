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

    print('Optimizing the', function_name, 'function...')
    # varying the number of chromosomes
    for n in range(30, 101, 10):
        ga = GA(function, bounds, pop_size=n)
        partial_results = []
        best_fitness_t = np.inf
        # executing the algorithm 30 times
        for i in range(30):
            execution, best_fitness, chromosome_list = ga.run()
            partial_results.append(execution)
            best_fitness.sort()
            if best_fitness_t > best_fitness[0]:
                best_fitness_t = best_fitness[0]
        if n == 50:
            result_distance = calculate_polulation_distance(chromosome_list)

        particle_number.append(n)
        results.append(partial_results)
        results_mean.append(np.mean(partial_results))
        results_best_fitness.append(best_fitness_t)

    # ploting
    print('Plotting Fitness vs Population Size...')
    plt.figure(figsize=(10, 7))
    plt.plot(results_mean, 'ro')
    plt.plot(results_best_fitness, 'b^')
    plt.xticks(range(1, len(results) + 1), particle_number)
    plt.yticks(np.arange(0, 1, step=0.5))
    plt.title(function_name)
    plt.ylabel("Fitness")
    plt.xlabel("Population Size")
    plt.legend(['Fitness Média', 'Melhor Fitness'], loc=0)
    plt.savefig("imgs/ga_fitness_vs_population_size(" + function_name + ").png")


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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python main.py <function>")
    function_name, function, bounds = get_function_and_bounds(sys.argv[1])
    main(function_name, function, bounds)
