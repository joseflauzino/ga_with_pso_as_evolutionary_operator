from statistics import mean
from functions import *
import struct


def hammingDistance(str1, str2):
    i = 0
    count = 0
    while (i < len(str1)):
        if (str1[i] != str2[i]):
            count += 1
        i += 1
    return count


def calculate_population_distance(generations):
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
    """ 
        Args: The function name.
        Returns: Function Name, Function Object, Bounds, Global Minimum
    """
    if function_name.lower() == 'sphere':
        return 'Sphere', sphere, [(-5.12, 5.12)], 0
    if function_name.lower() == 'rastrigin':
        return 'Rastrigin', rastrigin, [(-5.12, 5.12)], 0
    if function_name.lower() == 'ackley':
        return 'Ackley', ackley, [(-32.768, 32.768)], 0
    if function_name.lower() == 'eggholder':
        return 'Eggholder', eggholder, [(-512, 512)], -959.6407
    if function_name.lower() == 'drop_wave':
        return 'Drop Wave', drop_wave, [(-5.12, 5.12)], -1
    else:
        raise "Invalid function name"

  
def float_to_bin(num):
    return format(struct.unpack('!I', struct.pack('!f', num))[0], '032b')


def bin_to_float(binary):
    return struct.unpack('!f', struct.pack('!I', int(binary, 2)))[0]
    
