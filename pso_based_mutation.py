from __future__ import division
import random
import textwrap
import numpy as np
from operator import itemgetter
from util import *

#TODO: verificar se quando rod o PSO ele mantem individuos iguais
#TODO: deixar como parametro a opção de usar ou não no calculo da velocidade a inercia e o componente cognitivo

class Particle:
    def __init__(self, x0, fitness, inertia, constriction):
        self.position_i = []              # particle position (x_i)
        self.velocity_i = []              # particle velocity (v_i)
        self.fitness_i = fitness          # fitness individual
        self.best_pos_i = []              # best position individual (p_i)
        self.best_fitness_i = np.inf      # best fitness individual
        self.neighbors = []               # list of other particles ordered by proximity
        self.best_pos_g = []              # best position social/group (p_g)
        self.inertia = inertia            # particle inertia value
        self.constriction = constriction  # 1 if using constriction factor

        for i in range(0, num_dimensions):
            #TODO: colocar um if aqui pra anular a inercia 
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
    def update_velocity(self, pos_best_g, num_neighbors):
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
            if num_neighbors >= 0:
                vel_social = c2 * r2 * \
                    (self.best_pos_g[i] - self.position_i[i]) #TODO
            else:
                vel_social = c2 * r2 * (pos_best_g[i] - self.position_i[i])
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
    def find_best_local(self, num_neighbors):
        fitness_best_l = self.fitness_i
        self.best_pos_g = self.position_i
        for i in range(0, num_neighbors):
            if self.neighbors[i]['particle'].fitness_i < fitness_best_l:
                self.best_pos_g = self.neighbors[i]['particle'].position_i


class PSO_Mutation():
    def __init__(self, ga_population, costFunc, bounds, maxiter=1, num_neighbors=-1, inertia=0.5, constriction=False):
        global num_dimensions
        num_dimensions = len(bounds)
        #TODO: corrigir a forma que o swarm é feito, tornar uma lista de Particulas(objeto)
        self.ga_population = ga_population # genetic algorithm population sorted in ascending order based on fitness
        self.costFunc = costFunc           # the function to be optimized (fitness)
        self.bounds = bounds
        self.maxiter = maxiter
        self.num_neighbors = num_neighbors
        self.inertia = inertia
        self.constriction = constriction

    def to_swarm(self, population):
        """ Converts individuals of a population into particles of a swarm """

        def chromosome_to_postition(individual):
            """ Converts the chromosome of an individual to the position of a particle """
            # Breaks the chromosome string into strings of 32 bits (a float binary)
            binary_value = textwrap.wrap(individual['chromosome'], 32)
            # Converts the binaries to a float value
            return [bin_to_float(b) for b in binary_value]
        
        swarm = [] # the list of particles (i.e., the position of each particle)
        global_best_solution = chromosome_to_postition(population[0])   # the global best position
        global_best_fitness = population[0]['fitness']  # the fitness of the global best position/solution
        for individual in population:
            i_position = chromosome_to_postition(individual)
            swarm.append(Particle(i_position, individual['fitness'], self.inertia, self.constriction))
        return swarm, global_best_solution, global_best_fitness

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
            self.ga_population[i]['chromosome'] = position_to_chromosome(particle)
            self.ga_population[i]['fitness'] = particle.fitness_i
        return self.ga_population

    def run(self):

        # array of best fitness of each iteration
        iter_best_fitness = []            

        # establishes the swarm
        swarm, global_best_solution, global_best_fitness = self.to_swarm(self.ga_population)

        # begin optimization loop
        for i in range(self.maxiter):

            # cycle through particles in swarm and evaluate fitness
            for j in range(0, len(self.ga_population)):
                swarm[j].evaluate(self.costFunc)

                # determine if current particle is the best (globally)
                #TODO: cada particular tem que armazenar o seu melhor individual e o melhor global (que ela conhece)
                if swarm[j].fitness_i < global_best_fitness:
                    global_best_solution = list(swarm[j].position_i)
                    global_best_fitness = float(swarm[j].fitness_i)

                # find ordered list of neighbors (by distance from closest to farthest
                #Tá considerando todas as particluas como um único enxame, ou seja, não tem separação de global e local
                #Calcula a distancia pra todas as particulas, não somente dos vizinhos
                if self.num_neighbors >= 0:
                    for k in range(0, len(self.ga_population)):
                        if swarm[j] is not swarm[k]:
                            distance = swarm[j].euclidian_distance(swarm[k])
                            swarm[j].neighbors.append(
                                {'particle': swarm[k], 'distance': distance})

                    swarm[j].neighbors.sort(key=itemgetter('distance'))
                    swarm[j].find_best_local(self.num_neighbors)

            # save best fitnesses by iteration
            iter_best_fitness.append(global_best_fitness)

            # cycle through swarm and update velocities and position
            for j in range(0, len(self.ga_population)):
                swarm[j].update_velocity(global_best_solution, self.num_neighbors) #quando for utilizar vizinho, tem que ser a melhor solução baseada na quantidade de vizinhos
                swarm[j].update_position(self.bounds)
        
        # return fitness_best_g, iter_best_fitness
        
        return self.to_individuals(swarm) # return the individuals representing the childs (i.e., the 'mutated' individuals)


if __name__ == "__PSO_Mutation__":
    main()