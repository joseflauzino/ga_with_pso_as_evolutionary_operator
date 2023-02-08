from statistics import mean
from functions import sphere, rastringin, ackley

# Hamming distance
def hammingDistance(str1, str2):
    i = 0
    count = 0
    while(i < len(str1)):
        if(str1[i] != str2[i]):
          count += 1
        i += 1
    return count

def calculate_polulation_distance(generations):
  distances_per_generation = []
  for population in generations:
    distances_per_population = []
    for i in range(len(population) - 1):
      x = hammingDistance(population[i], population[i+1])
      distances_per_population.append(x)
    distances_per_generation.append(distances_per_population)

  avg_distances_per_generation = []
  for p in distances_per_generation:
    avg_distances_per_generation.append(mean(p))

  return avg_distances_per_generation

def get_function_and_bounds(function_name: str):
  if function_name.lower() == 'sphere':
     return sphere, [(-5, 5), (-5, 5), (-5, 5)]
  if function_name.lower() == 'rastringin':
     return rastringin, [(-5.12, 5.12), (-5.12, 5.12), (-5.12, 5.12)]
  if function_name.lower() == 'ackley':
     return ackley, [(-5, 5), (-5, 5), (-5, 5)]
  else:
     raise "Invalid function name"