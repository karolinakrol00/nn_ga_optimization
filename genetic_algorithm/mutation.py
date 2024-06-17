import numpy as np

def random_mutation(individual, mutation_prob, gene_number, round = False):

    mutated_gene = individual.generate_genotype()[gene_number]

    prob = np.random.uniform()

    if prob < mutation_prob: 
        individual.genotype[gene_number] = mutated_gene
        if round: 
            round(individual.genotype[gene_number])

    individual.fitness = None

def remove_topology_element(individual, mutation_prob, gene_number):

    gene = individual.genotype[gene_number]
    n_genes = len(gene)

    if np.sum(gene) != 1:
         for i in range(n_genes):
            prob = np.random.uniform()

            if prob < mutation_prob:
                gene[i] = 0
    
    individual.fitness = None

def add_topology_element(individual, mutation_prob, gene_number):

    gene = individual.genotype[gene_number]
    n_genes = len(gene)

    for i in range(n_genes):
        prob = np.random.uniform()

        if prob < mutation_prob and gene[i] == 0:
            gene[i] = 1

    individual.fitness = None

def add_neurons(individual, mutation_prob, gene_number):

    gene = individual.genotype[gene_number]
    n_genes = len(gene)

    for i in range(n_genes):
        prob = np.random.uniform()

        if prob < mutation_prob and gene[i] > 0:
            low = gene[i] + 1
            high = low * 2
            gene[i] = round(np.random.uniform(low, high))

    individual.fitness = None

def remove_neurons(individual, mutation_prob, gene_number):

    gene = individual.genotype[gene_number]
    n_genes = len(gene)

    for i in range(n_genes):
        prob = np.random.uniform()

        if prob < mutation_prob and gene[i] > 1:
            low = 1
            high = gene[i] * 0.5
            gene[i] -= round(np.random.uniform(low, high))

    individual.fitness = None

def random_weight_mutation(individual, mutation_prob):

    n_genes = len(individual.genotype)
    prob = np.random.uniform()
    
    if prob < mutation_prob:
        pos = np.random.randint(0, n_genes, 1)[0]
        individual.genotype[pos] = np.random.normal()
        individual.fitness = None

def swap_weight_mutation(individual, mutation_prob):

    n_genes = len(individual.genotype)
    prob = np.random.uniform()
    
    if prob < mutation_prob:
        pos_1, pos_2 = np.random.randint(0, n_genes, 2)
        swap_1 = individual.genotype[pos_1]
        individual.genotype[pos_1] = individual.genotype[pos_2]
        individual.genotype[pos_2] = swap_1
        individual.fitness = None

def scramble_weight_mutation(individual, mutation_prob, mutation_size):

    n_genes = len(individual.genotype)
    prob = np.random.uniform()
    mutation_start = np.random.randint(0, n_genes - mutation_size)
    mutation_end = mutation_start + mutation_size

    if prob < mutation_prob: 
        np.random.shuffle(individual.genotype[mutation_start:mutation_end])
        individual.fitness = None

def inversion_weight_mutation(individual, mutation_prob, mutation_size):

    n_genes = len(individual.genotype)
    prob = np.random.uniform()
    mutation_start = np.random.randint(0, n_genes - mutation_size)
    mutation_end = mutation_start + mutation_size

    if prob < mutation_prob: 
        individual.genotype[mutation_start:mutation_end] = individual.genotype[mutation_start:mutation_end][::-1]
        individual.fitness = None