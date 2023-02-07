import sys
import numpy as np
import matplotlib.pyplot as plt

from ga import GA
from functions import sphere, rastringin, ackley



bounds = [(-5, 5), (-5, 5), (-5, 5)]

sphere_results = []
sphere_particle_number = []

# varying the number of chromosomes
for n in range(30, 101, 10):
    ga = GA(sphere, bounds, pop_size=n)
    results = []
    # executing the algorithm 30 times
    for i in range(30):
        execution, _ = ga.run()
        results.append(execution)
        
    sphere_particle_number.append(n)
    sphere_results.append(results)