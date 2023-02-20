from copy import copy
import sys
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt

from ga import GA
from util import *
from plot import *

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
num_exec = 60
x = np.arange(1, num_gen + 1, 1)  # iteration number array
x_zoom = np.arange(20, num_gen, 1)  # iteration number array

ms = 5
lw = 2
capsize = 3
elw = 0.5

mt_prob=0.25
topology = "ring"
topology=topology

def print_winner(result_list):
    ordered_list = sorted(result_list, key=itemgetter('result')) 
    if ordered_list[0]['result'] == ordered_list[1]['result']:
        print("There was a tie!")
    else:
        print(ordered_list[0]['name'],'won!')

def population_size(function_name, function, bounds, global_minimum):
    print('Optimizing the', function_name, 'function by varying the population size...')

    def run(algorithm_instance):
        results = {'average_fitness': [], 'best_fitness': [], 'average_distance': []}
        for n in range(30, 101, 10):
            algorithm_instance.pop_size = n
            best_fitness_t = np.inf
            partial_results = []
            for i in range(num_exec):
                execution, best_fitness, chromosome_list = algorithm_instance.run()
                partial_results.append(execution)
                best_fitness.sort()
                if best_fitness_t > best_fitness[0]:
                    best_fitness_t = best_fitness[0]
                if n == 50:
                    results['average_distance'].append(calculate_population_distance(chromosome_list))
            results['average_fitness'].append(np.mean(partial_results))
            results['best_fitness'].append(copy(best_fitness_t))
        return results
    
    ga_pso_results = run(GA(function, bounds, mt_prob=mt_prob, pso_mutation=True, with_inertia=True, topology=topology))
    ga_pso_best_fitness = min(ga_pso_results['best_fitness'])
    print("Best Fitness ga_pso population_size:", ga_pso_best_fitness)

    ga_results = run(GA(function, bounds, mt_prob=mt_prob))
    ga_best_fitness = min(ga_results['best_fitness'])
    print("Best Fitness GA population_size:", ga_best_fitness)

    print_winner([
        {'name': 'GA-PSO', 'result': ga_pso_best_fitness},
        {'name': 'GA', 'result': ga_best_fitness}])

    plot_population_size(function_name, global_minimum, ga_results, ga_pso_results)


def run_generic(algorithm_instance):
    best_fitness_list = []
    best_fitnesses = np.inf
    for i in range(num_exec):
        _, exec_best_fitnesses, _ = algorithm_instance.run()
        best_fitness_list.append(exec_best_fitnesses)
        if best_fitnesses > min(exec_best_fitnesses):
            best_fitnesses = copy(min(exec_best_fitnesses))
    return best_fitness_list, best_fitnesses

def group(best_fitness_list):
    # grouping results by iteration number
    gen_results = [[] for i in range(num_gen)]
    for i in range(len(best_fitness_list)):
        for j in range(len(best_fitness_list[i])):
            gen_results[j].append(best_fitness_list[i][j])
    return gen_results

def calc_average(gen_results_m):
    # calculating mean values by iteration number
    average_list = []
    for i in range(len(gen_results_m)):
        average_list.append(np.mean(gen_results_m[i]))
    return average_list

def mutation(function_name, function, bounds, global_minimum):
    print('Optimizing the mutation', function_name, 'function by varying the mutation strategy...')

    best_fitness_list_no_mutation, best_fitness_no_mutation = run_generic(
        GA(function, bounds, generations=num_gen))
    best_fitness_list_with_mutation, best_fitness_with_mutation = run_generic(
        GA(function, bounds, generations=num_gen, mt_prob=mt_prob))
    best_fitness_list_pso_mutation, best_fitness_pso_mutation = run_generic(
        GA(function, bounds, generations=num_gen, mt_prob=mt_prob, 
           pso_mutation=True, with_inertia=True, topology=topology))

    results_no_mutation = group(best_fitness_list_no_mutation)
    results_with_mutation = group(best_fitness_list_with_mutation)
    results_pso_mutation = group(best_fitness_list_pso_mutation)
    
    average_fitness = {
        'no_mutation': calc_average(results_no_mutation),
        'mutation': calc_average(results_with_mutation),
        'pso_mutation': calc_average(results_pso_mutation)
    }
    
    print("Best Fitness - GA with no mutation:", best_fitness_no_mutation)
    print("Best Fitness - GA with mutation:", best_fitness_with_mutation)
    print("Best Fitness - GA with PSO mutation:", best_fitness_pso_mutation)

    print_winner([
        {'name': 'GA with no mutation', 'result': best_fitness_no_mutation},
        {'name': 'GA with mutation', 'result': best_fitness_with_mutation},
        {'name': 'GA with PSO mutation', 'result': best_fitness_pso_mutation}])
    
    plot_mutation(function_name, average_fitness, global_minimum)


def topology_func(function_name, function, bounds, global_minimum):
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
    ga_pso_g = GA(function, bounds, generations=num_gen, mt_prob=mt_prob,
                  pso_mutation=True, with_inertia=True)

    gen_best_fitnesses_g = np.inf
    for i in range(num_exec):
        _, execution_best_fitnesses_g, _ = ga_pso_g.run()
        best_fitnesses_g.append(execution_best_fitnesses_g)
        if gen_best_fitnesses_g > min(execution_best_fitnesses_g):
            gen_best_fitnesses_g = copy(min(execution_best_fitnesses_g))

    # executing local topology algorithm 30 times
    ga_pso__l = GA(function, bounds, generations=num_gen, mt_prob=mt_prob,
                   pso_mutation=True, with_inertia=True, topology='ring')

    gen_best_fitnesses_l = np.inf
    for i in range(num_exec):
        _, execution_best_fitnesses_l, _ = ga_pso__l.run()
        best_fitnesses_l.append(execution_best_fitnesses_l)
        if gen_best_fitnesses_l > min(execution_best_fitnesses_l):
            gen_best_fitnesses_l = copy(min(execution_best_fitnesses_l))

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

    print("Best Fitness topology - ga_pso global:", gen_best_fitnesses_g)
    print("Best Fitness topology - ga_pso local:", gen_best_fitnesses_l)


def social(function_name, function, bounds, global_minimum):
    print('Optimizing the social', function_name, 'function...')

    best_fitnesses_inertia_false = []
    best_fitnesses_inertia_true = []
    best_fitnesses_ga = []
    gen_results_inertia_false = [[] for i in range(num_gen)]
    gen_results_inertia_true = [[] for i in range(num_gen)]
    gen_results_ga = [[] for i in range(num_gen)]
    gen_mean_inertia_false = []
    gen_mean_inertia_true = []
    gen_mean_ga = []
    gen_std_inertia_false = []
    gen_std_inertia_true = []
    gen_std_ga = []

    # executing global topology algorithm 30 times
    ga_pso_inertia_false = GA(function, bounds, generations=num_gen, mt_prob=mt_prob,
                              pso_mutation=True, with_inertia=False, max_iter=1, topology=topology)

    gen_best_fitnesses_inertia_false = np.inf
    for i in range(num_exec):
        _, execution_best_fitnesses_inertia_false, _ = ga_pso_inertia_false.run()
        best_fitnesses_inertia_false.append(execution_best_fitnesses_inertia_false)
        if gen_best_fitnesses_inertia_false > min(execution_best_fitnesses_inertia_false):
            gen_best_fitnesses_inertia_false = copy(min(execution_best_fitnesses_inertia_false))

    # executing local topology algorithm 30 times
    ga_pso_inertia_true = GA(function, bounds, generations=num_gen, mt_prob=mt_prob,
                             pso_mutation=True, with_inertia=True, max_iter=10, topology=topology)

    gen_best_fitnesses_inertia_true = np.inf
    for i in range(num_exec):
        _, execution_best_fitnesses_inertia_true, _ = ga_pso_inertia_true.run()
        best_fitnesses_inertia_true.append(execution_best_fitnesses_inertia_true)
        if gen_best_fitnesses_inertia_true > min(execution_best_fitnesses_inertia_true):
            gen_best_fitnesses_inertia_true = copy(min(execution_best_fitnesses_inertia_true))

    # executing just genetic algorithm
    ga = GA(function, bounds, generations=num_gen, mt_prob=mt_prob)

    gen_best_fitnesses_ga = np.inf
    for i in range(num_exec):
        _, execution_best_fitnesses_ga, _ = ga.run()
        best_fitnesses_ga.append(execution_best_fitnesses_ga)
        if gen_best_fitnesses_ga > min(execution_best_fitnesses_ga):
            gen_best_fitnesses_ga = copy(min(execution_best_fitnesses_ga))

    # grouping results by iteration number
    for i in range(len(best_fitnesses_inertia_false)):
        for j in range(len(best_fitnesses_inertia_false[i])):
            gen_results_inertia_false[j].append(
                best_fitnesses_inertia_false[i][j])
            gen_results_inertia_true[j].append(
                best_fitnesses_inertia_true[i][j])
            gen_results_ga[j].append(best_fitnesses_ga[i][j])

    # calculating mean and std values by iteration number
    for i in range(len(gen_results_inertia_false)):
        gen_mean_inertia_false.append(np.mean(gen_results_inertia_false[i]))
        gen_std_inertia_false.append(np.std(gen_results_inertia_false[i]))

        gen_mean_inertia_true.append(np.mean(gen_results_inertia_true[i]))
        gen_std_inertia_true.append(np.std(gen_results_inertia_true[i]))

        gen_mean_ga.append(np.mean(gen_results_ga[i]))
        gen_std_ga.append(np.std(gen_results_ga[i]))


def main(function_name, function, bounds, global_minimum):
    # population_size(function_name, function, bounds, global_minimum)
    mutation(function_name, function, bounds, global_minimum)
    # topology_func(function_name, function, bounds, global_minimum)
    #social(function_name, function, bounds, global_minimum)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python main.py <function>")
    function_name, function, bounds, global_minimum = get_function_and_bounds(
        sys.argv[1])
    main(function_name, function, bounds, global_minimum)
