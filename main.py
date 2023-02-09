import sys
import numpy as np
import matplotlib.pyplot as plt

from ga import GA
from util import *

# usage: python main.py <function>
if len(sys.argv) < 2:
  print('Invalid usage!')
  sys.exit()

function_arg = sys.argv[1]
function, bounds = get_function_and_bounds(function_arg)
function_name = function_arg.title()

results = []
particle_number = []

print('Optimizing the', function_name, 'function...')
# varying the number of chromosomes
for n in range(30, 101, 10):
  ga = GA(function, bounds, pop_size=n, mt_prob=0.4, pso_mutation=True)
  partial_results = []
  # executing the algorithm 30 times
  for i in range(1):
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
plt.yticks(np.arange(0, 11, step=1))
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