import sys
import numpy as np
import matplotlib.pyplot as plt

from pso import PSO
from functions import sphere, rastringin, ackley

bounds = [(-5, 5), (-5, 5), (-5, 5)]

sphere_results = []
sphere_particle_number = []

# varying the number of particles
for n in range(30, 101, 10):
    pso = PSO(sphere, bounds, maxiter=200, num_particles=n)
    results = []
    # executing the algorithm 30 times
    for i in range(30):
        execution, _ = pso.run()
        results.append(execution)
        
    sphere_particle_number.append(n)
    sphere_results.append(results)

plt.figure(figsize=(20, 5))
plt.boxplot(sphere_results)
plt.xticks(range(1, len(sphere_results) + 1), sphere_particle_number)
plt.title('Sphere')
plt.ylabel("Fitness")
plt.xlabel("Number of Particles")

plt.savefig("pso_result.png")