from copy import copy
import numpy as np
import matplotlib.pyplot as plt

def plot_population(function_name, global_minimum, population):
    print('Plotting the Population Evolution...')
    plt.rcParams['font.size'] = '16'
    plt.figure(figsize=(12, 9))

    plt.savefig("imgs/" + function_name +
                "/population(" + function_name + ")-GA.pdf", bbox_inches='tight')
    
    plt.savefig("imgs/" + function_name +
                "/population(" + function_name + ")-GA-PSO.pdf", bbox_inches='tight')
    
def plot_population_size(function_name, global_minimum, ga_results, ga_pso_results):
    print('Plotting Fitness vs Population Size...')
    plt.rcParams['font.size'] = '16'
    plt.figure(figsize=(12, 9))
    plt.xticks(list(range(30, 101, 10)))  # Set text labels.
    data_range_fit_vs_pop = list(range(30, 101, 10))
    plt.plot(data_range_fit_vs_pop, ga_pso_results['average_fitness'], 'rs', label="Fitness Média GA-PSO")
    plt.plot(data_range_fit_vs_pop, ga_pso_results['best_fitness'], 'b*', label="Melhor Fitness GA-PSO")
    plt.plot(data_range_fit_vs_pop, ga_results['average_fitness'], 'g+', label="Fitness Média GA")
    plt.plot(data_range_fit_vs_pop, ga_results['best_fitness'], 'mx', label="Melhor Fitness GA")
    max_y = max(ga_pso_results['average_fitness'])
    step = abs(max_y/10)
    plt.yticks(np.arange(global_minimum, max_y+step, step=step))
    plt.title(function_name, fontsize=18)
    plt.ylabel("Fitness Média", fontsize=18)
    plt.xlabel("Tamanho da População", fontsize=18)
    plt.legend(loc=0, prop={'size': 16})
    plt.savefig("imgs/" + function_name +
                "/fitness_vs_population_size(" + function_name + ").pdf", bbox_inches='tight')

    print('Plotting Average Distance over Generations...')
    result_distance_ga_pso = []
    result_distance_ga = []
    for gen in range(100):
        aux_result_distance_ga_pso = []
        aux_result_distance_ga = []
        for list_avg in ga_pso_results['average_distance']:
            aux_result_distance_ga_pso.append(copy(list_avg[gen]))
        # calcula a media da geração aqui
        result_distance_ga_pso.append(np.mean(aux_result_distance_ga_pso))

        for list_avg in ga_results['average_distance']:
            aux_result_distance_ga.append(copy(list_avg[gen]))
        # calcula a media da geração aqui
        result_distance_ga.append(np.mean(aux_result_distance_ga))
    plt.figure(figsize=(12, 9))
    plt.xticks([1, 20, 40, 60, 80, 100])  # Set text labels.
    data_range = np.arange(1, 101)
    plt.plot(data_range, result_distance_ga_pso, 'ro', label="GA-PSO")
    plt.plot(data_range, result_distance_ga, 'bo', label="GA")
    plt.title(function_name, fontsize=18)
    plt.ylabel("Distância Média", fontsize=18)
    plt.xlabel("Geração", fontsize=18)
    plt.yticks(np.arange(0, 110, step=10))
    plt.legend(loc=0, prop={'size': 16})
    plt.savefig(
        "imgs/" + function_name + "/average_distance_over_generations(" + function_name + ").pdf", bbox_inches='tight')
    print('')

def plot_mutation(function_name, average_fitness, global_minimum):
    upper_limit = max(average_fitness['mutation'][0], average_fitness['no_mutation'][0], average_fitness['pso_mutation'][0])
    plt.rcParams['font.size'] = '16'
    plt.figure(figsize=(12, 9))
    plt.ylim(global_minimum, upper_limit)
    plt.plot(average_fitness['no_mutation'], 'bo', label="SEM Mutação")
    plt.plot(average_fitness['mutation'], 'rx', label="COM Mutação")
    plt.plot(average_fitness['pso_mutation'], 'g+', label="COM Mutação - PSO")
    plt.title(function_name, fontsize=18)
    plt.ylabel("Fitness Média", fontsize=18)
    plt.xlabel("Geração", fontsize=18)
    plt.legend(loc=0, prop={'size': 16})
    plt.savefig(
        "imgs/" + function_name + "/mutation(" + function_name + ").pdf", bbox_inches='tight')
    print('')
    
def plot_topology(function_name, average_fitness, global_minimum):
    upper_limit = max(average_fitness['global'][0], average_fitness['local'][0])
    plt.rcParams['font.size'] = '16'
    plt.figure(figsize=(12, 9))
    plt.ylim(global_minimum, upper_limit)
    plt.xticks([1, 20, 40, 60, 80, 100])  # Set text labels.
    data_range_complete = np.arange(1, 101)
    plt.plot(data_range_complete, average_fitness['global'], 'bo', label="Topologia Global")
    plt.plot(data_range_complete, average_fitness['local'], 'rx', label="Topologia Local")
    plt.title(function_name, fontsize=18)
    plt.ylabel("Fitness Média", fontsize=18)
    plt.xlabel("Geração", fontsize=18)
    plt.legend(loc=0, prop={'size': 16})

    plt.savefig(
        "imgs/" + function_name + "/topology(" + function_name + ").pdf", bbox_inches='tight')
    print('')
    
def plot_inertia_and_cognitive(function_name, average_fitness, global_minimum):
    upper_limit = max(
        average_fitness['cognitive_false'][0], average_fitness['cognitive_true'][0], average_fitness['ga'][0])
    plt.rcParams['font.size'] = '16'
    plt.figure(figsize=(12, 9))
    plt.ylim(-0.2, upper_limit)
    plt.ylim(global_minimum, upper_limit)
    plt.xticks([1, 20, 40, 60, 80, 100])  # Set text labels.
    data_range_complete = np.arange(1, 101)
    plt.plot(data_range_complete, average_fitness['cognitive_false'], 'ro', label="GA-PSO (1 iteração do PSO)")
    plt.plot(data_range_complete, average_fitness['cognitive_true'], 'b+', label="GA-PSO (10 iterações do PSO)")
    plt.plot(data_range_complete, average_fitness['ga'], 'gx', label="GA")
    plt.title(function_name, fontsize=18)
    plt.ylabel("Fitness Média", fontsize=18)
    plt.xlabel("Geração", fontsize=18)
    plt.legend(loc=0, prop={'size': 16})

    plt.savefig("imgs/" + function_name + "/cognitive-complete(" + function_name + ").pdf", bbox_inches='tight')
    print('')

    # zoom
    N = 20
    upper_limit = max(max(average_fitness['cognitive_false'][N:41]), max(average_fitness['cognitive_true'][N:41]), max(average_fitness['ga'][N:41]))
    plt.rcParams['font.size'] = '16'
    plt.figure(figsize=(12, 9))
    plt.ylim(global_minimum, upper_limit)
    plt.xticks(np.arange(20, 41, step=10))  # Set text labels.
    data_range = np.arange(20, 41)
    plt.plot(data_range, average_fitness['cognitive_false'][N:41], 'ro', label="GA-PSO (1 iteração do PSO)")
    plt.plot(data_range, average_fitness['cognitive_true'][N:41], 'b+', label="GA-PSO (10 iterações do PSO)")
    plt.plot(data_range, average_fitness['ga'][N:41], 'gx', label='GA')
    plt.title(function_name, fontsize=18)
    plt.ylabel("Fitness Média", fontsize=18)
    plt.xlabel("Geração", fontsize=18)
    plt.legend(loc=0, prop={'size': 16})

    plt.savefig("imgs/" + function_name + "/cognitive-zoom(" + function_name + ").pdf", bbox_inches='tight')