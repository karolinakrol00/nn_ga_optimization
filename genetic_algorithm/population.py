import numpy as np
from .individual import Individual

class Population:

    def __init__(self, n_inividulas_var, genotype_blueprint):
        self.n_individuals_var = n_inividulas_var
        self.genotype_blueprint = genotype_blueprint
        self.individuals = [Individual(genotype_blueprint).assign_generated_genotype() for _ in range(self.n_individuals_var)]

    def individual_fitness_sort_key(self, individual: Individual):
        return individual.fitness

    def sort_by_fitness(self):
        return sorted(self.individuals, key = self.individual_fitness_sort_key, reverse = True)
        
    def get_the_fittest(self, n: int):
        return self.sort_by_fitness()[:n]
    
    def crossover(self, crossover_function, crossover_percentage, *args):
        mating_pool = self.get_the_fittest(round(self.n_individuals_var * 0.5))
        children_genotype = crossover_function(mating_pool, *args)
        new_population = [Individual(self.genotype_blueprint, genotype) for genotype in children_genotype]
        self.individuals = [*new_population, *self.get_the_fittest(round(self.n_individuals_var * (1 - crossover_percentage)))]

    def mutation(self, mutation_function, *args):
        for ind in self.individuals: mutation_function(ind, *args)