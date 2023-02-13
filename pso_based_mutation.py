import random
import textwrap
from typing import List
import numpy as np
from operator import itemgetter
from util import *

# TODO: verificar se quando rod o PSO ele mantem individuos iguais


class Particle:
    def __init__(self, x0, fitness, global_best_solution, global_best_fitness, inertia, constriction, with_inertia):
        self.position_i = []              # particle position (x_i)
        self.velocity_i = []              # particle velocity (v_i)
        self.fitness_i = fitness          # fitness individual
        self.best_pos_i = []              # best position individual (p_i)
        self.best_fitness_i = np.inf      # best fitness individual
        self.neighbors = []               # list of other particles ordered by proximity
        # best position social/group (p_g)
        self.best_pos_g = global_best_solution
        # best position social/group (p_g)
        self.best_fitness_g = global_best_fitness
        self.inertia = inertia            # particle inertia value
        self.constriction = constriction  # 1 if using constriction factor
        self.with_inertia = with_inertia

        for i in range(0, num_dimensions):
            if self.with_inertia != True:
                self.velocity_i.append(0)
            else:
                self.velocity_i.append(random.uniform(-1, 1))
            self.position_i.append(x0[i])

    # evaluate current fitness
    def evaluate(self, costFunc):
        self.fitness_i = costFunc(self.position_i)

        # check to see if the current position is an individual best
        if self.fitness_i < self.best_fitness_i:
            self.best_pos_i = self.position_i
            self.best_fitness_i = self.fitness_i

    # update new particle velocity
    def update_velocity(self):
        # constant inertia weight (how much to weigh the previous velocity)
        w = self.inertia
        c1 = 2.1        # cognitive constant
        c2 = 2.1        # social constant
        phi = c1+c2
        k = 2/(np.absolute(2-phi-np.sqrt(phi**2 - 4*phi)))

        for i in range(0, num_dimensions):
            r1 = random.random()
            r2 = random.random()
            vel_cognitive = c1 * r1 * (self.best_pos_i[i] - self.position_i[i])
            vel_social = c2 * r2 * \
                (self.best_pos_g[i] - self.position_i[i])
            if self.constriction:
                self.velocity_i[i] = k * \
                    (self.velocity_i[i] + vel_cognitive + vel_social)
            else:
                self.velocity_i[i] = w * self.velocity_i[i] + \
                    vel_cognitive + vel_social

    # update the particle position based off new velocity updates
    def update_position(self, bounds):
        for i in range(0, num_dimensions):
            self.position_i[i] = self.position_i[i] + self.velocity_i[i]

            # adjust maximum position if necessary
            if self.position_i[i] > bounds[i][1]:
                self.position_i[i] = bounds[i][1]

            # adjust minimum position if necessary
            if self.position_i[i] < bounds[i][0]:
                self.position_i[i] = bounds[i][0]

    # calculate euclidian distance between 2 particles
    def euclidian_distance(self, other_particle):
        coord_self = np.array(self.position_i)
        coord_other = np.array(other_particle.position_i)
        distance = np.linalg.norm(coord_self - coord_other)
        return distance

    # find best position locally, using neighbors (local topology only)
    def find_best_neighbors(self, swarm, topology):
        if topology == "ring":
            for p_n in self.neighbors:
                if swarm[p_n].fitness_i < self.best_fitness_g:
                    self.best_pos_g = swarm[p_n].position_i
                    self.best_fitness_g = swarm[p_n].fitness_i
        else:
            for p_n in swarm:
                if p_n.fitness_i < self.best_fitness_g:
                    self.best_pos_g = p_n.position_i
                    self.best_fitness_g = p_n.fitness_i


class PSO_Mutation():
    def __init__(self, ga_population, costFunc, bounds, topology='', maxiter=1, inertia=0.5, constriction=False, with_inertia=True):
        global num_dimensions
        num_dimensions = len(bounds)
        # genetic algorithm population sorted in ascending order based on fitness
        self.ga_population = ga_population
        # the function to be optimized (fitness)
        self.costFunc = costFunc
        self.bounds = bounds
        self.maxiter = maxiter
        self.topology = str(topology).lower()
        self.inertia = inertia
        self.constriction = constriction
        self.with_inertia = with_inertia

    def to_swarm(self, population):
        """ Converts individuals of a population into particles of a swarm """

        def chromosome_to_postition(individual):
            """ Converts the chromosome of an individual to the position of a particle """
            # Breaks the chromosome string into strings of 32 bits (a float binary)
            binary_value = textwrap.wrap(individual['chromosome'], 32)
            # Converts the binaries to a float value
            return [bin_to_float(b) for b in binary_value]

        # the list of particles (i.e., the position of each particle)
        swarm = []
        # the global best particle
        global_best_solution = chromosome_to_postition(
            population[0])   
        global_best_fitness = population[0]['fitness']

        for individual in population:
            i_position = chromosome_to_postition(individual)
            swarm.append(Particle(
                i_position, individual['fitness'], global_best_solution, global_best_fitness, self.inertia, self.constriction, self.with_inertia))
        return swarm

    def get_neighbors(self, swarm):
        N = len(swarm)
        if self.topology == "ring":
            list_neighbors = (
                np.array([np.arange(3) for _ in range(N)]) + np.arange(N)[:, None] - 1) % N
            for index, particle in enumerate(swarm):
                particle.neighbors = list_neighbors[index]
        else:
            list_neighbors = list(range(0, N))
            for index, particle in enumerate(swarm):
                particle.neighbors = list_neighbors.copy()

    def to_individuals(self, swarm):
        """ Converts particles of a swarm into individuals of a population """

        def position_to_chromosome(particle: Particle):
            """ Converts the position of a particle to the chromosome of an individual """
            chromosome = ''
            for value in particle.position_i:
                # converts to a binary value (a string)
                chromosome += float_to_bin(value)
            return chromosome

        for i, particle in enumerate(swarm):
            self.ga_population[i]['chromosome'] = position_to_chromosome(
                particle)
            self.ga_population[i]['fitness'] = particle.fitness_i
        return self.ga_population

    def run(self):

        # establishes the swarm
        swarm: List[Particle] = self.to_swarm(self.ga_population)

        # establishes the neighborhood of the particle
        self.get_neighbors(swarm)

        # begin optimization loop
        for i in range(self.maxiter):

            # cycle through particles in swarm and evaluate fitness
            for x_i in range(len(swarm)):
                swarm[x_i].evaluate(self.costFunc)

                swarm[x_i].find_best_neighbors(swarm, self.topology)
                swarm[x_i].update_velocity()
                swarm[x_i].update_position(self.bounds)

        # return the individuals representing the childs (i.e., the 'mutated' individuals)
        return self.to_individuals(swarm)


if __name__ == "__PSO_Mutation__":
    main()
