import sys
import numpy as np
import matplotlib.pyplot as plt

from ga import GA
from util import *
from functions import sphere, rastringin, ackley

bounds = [(-5, 5), (-5, 5), (-5, 5)]

# usage: python main.py <function>
if len(sys.argv) < 2:
  print('Invalid usage!')
  sys.exit()

function_arg = sys.argv[1]
print('Function:', function_arg)
function = sphere #TODO: atribuir o valor da variavel function de acordo com o arg passado
function_name = function.__name__.title()
#TODO: atribuir o valor da variavel bounds de acordo com o valor da variavel function (ou seja, de acordo com a funcao)

results = []
particle_number = []

print('Optimizing the', function_name, 'function...')
# varying the number of chromosomes
for n in range(30, 101, 10):
  ga = GA(function, bounds, pop_size=n)
  partial_results = []
  # executing the algorithm 30 times
  for i in range(30):
    execution, _, chromosome_list = ga.run()
    partial_results.append(execution)
  if n == 50:
    result_distance = calculate_polulation_distance(chromosome_list)

  particle_number.append(n)
  results.append(partial_results)

# ploting
print('Plotting Fitness vs Population Size...')
plt.figure(figsize=(10, 7))
plt.boxplot(results)
plt.xticks(range(1, len(results) + 1), particle_number)
plt.title(function_name)
plt.ylabel("Fitness")
plt.xlabel("Population Size")
plt.savefig("ga_fitness_vs_population_size("+ function_name +").png")

print('Plotting Average Distance over Generations...')
plt.figure(figsize=(10, 7))
plt.plot(result_distance, 'ro')
plt.title(function_name)
plt.ylabel("Average Distance")
plt.xlabel("Generation")
plt.yticks(np.arange(0, 110, step=10))
plt.savefig("ga_average_distance_over_generations("+ function_name +").png")

print('')