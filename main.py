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
    print('Optimizing the', function_name, 'function by varying the mutation strategy...')

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
    print('Optimizing the', function_name, 'function by varying the topology...')

    best_fitness_list_global, best_fitness_global = run_generic(GA(function, bounds, generations=num_gen, mt_prob=mt_prob,
                  pso_mutation=True, with_inertia=True))
    best_fitness_list_local, best_fitness_local = run_generic(GA(function, bounds, generations=num_gen, mt_prob=mt_prob,
                   pso_mutation=True, with_inertia=True, topology='ring'))

    results_global = group(best_fitness_list_global)
    results_local = group(best_fitness_list_local)
    
    average_fitness = {
        'global': calc_average(results_global),
        'local': calc_average(results_local)
    }

    print("Best Fitness - GA-PSO with Global Topology:", best_fitness_global)
    print("Best Fitness - GA-PSO with Local Topology:", best_fitness_local)

    print_winner([
        {'name': 'GA-PSO with Global Topology', 'result': best_fitness_global},
        {'name': 'GA-PSO with Local Topology', 'result': best_fitness_local}])
    
    plot_topology(function_name, average_fitness, global_minimum)

def inertia_and_cognitive(function_name, function, bounds, global_minimum):
    print('Optimizing the', function_name, 'function by varying the number of PSO iterations (add/remove inertia and cognitive components)...')

    best_fitness_list_cognitive_false, best_fitness_cognitive_false = run_generic(
        GA(function, bounds, generations=num_gen, mt_prob=mt_prob, 
           pso_mutation=True, with_inertia=False, max_iter=1, topology=topology))
    best_fitness_list_cognitive_true, best_fitness_cognitive_true = run_generic(
        GA(function, bounds, generations=num_gen, mt_prob=mt_prob, 
           pso_mutation=True, with_inertia=True, max_iter=10, topology=topology))
    best_fitness_list_ga, best_fitness_ga = run_generic(
        GA(function, bounds, generations=num_gen, mt_prob=mt_prob))

    results_cognitive_false = group(best_fitness_list_cognitive_false)
    results_cognitive_true = group(best_fitness_list_cognitive_true)
    results_ga = group(best_fitness_list_ga)
    
    average_fitness = {
        'cognitive_false': calc_average(results_cognitive_false),
        'cognitive_true': calc_average(results_cognitive_true),
        'ga': calc_average(results_ga)
    }
    
    print("Best Fitness - GA-PSO 1 iteration:", best_fitness_cognitive_false)
    print("Best Fitness - GA-PSO 10 iterations:", best_fitness_cognitive_true)
    print("Best Fitness - GA:", best_fitness_ga)

    print_winner([
        {'name': 'GA-PSO 1 iteration', 'result': best_fitness_cognitive_false},
        {'name': 'GA-PSO 10 iterations', 'result': best_fitness_cognitive_true},
        {'name': 'GA', 'result': best_fitness_ga}])
    
    plot_inertia_and_cognitive(function_name, average_fitness, global_minimum)

def main(function_name, function, bounds, global_minimum):
    # population_size(function_name, function, bounds, global_minimum)
    # mutation(function_name, function, bounds, global_minimum)
    # topology_func(function_name, function, bounds, global_minimum)
    inertia_and_cognitive(function_name, function, bounds, global_minimum)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python main.py <function>")
    function_name, function, bounds, global_minimum = get_function_and_bounds(
        sys.argv[1])
    main(function_name, function, bounds, global_minimum)
