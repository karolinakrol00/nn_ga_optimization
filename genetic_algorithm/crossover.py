import numpy as np

def k_point_crossover(mating_pool, crossover_percentage = 0.8, k = 1):

    n_genes = len(mating_pool[0].genotype)
    mating_pool_size = len(mating_pool)
    mating_number = round(mating_pool_size * crossover_percentage)
        
    crossover_points = np.sort(np.random.randint(0, n_genes, k))

    children = []

    for _ in range(mating_number): 

        parents_ind = np.random.choice(mating_pool_size, 2, replace = False)
        parent_1 = mating_pool[parents_ind[0]]
        parent_2 = mating_pool[parents_ind[1]]

        for cp in crossover_points:

            child_1 = [*parent_1.genotype[:cp], *parent_2.genotype[cp:]]
            child_2 = [*parent_2.genotype[:cp], *parent_1.genotype[cp:]]

        children.extend([child_1, child_2])
        
    return children

def mean_crossover(mating_pool, crossover_percentage = 0.8):

    mating_pool_size = len(mating_pool)
    mating_number = 2 * round(mating_pool_size * crossover_percentage)

    children = []

    for _ in range(mating_number): 

        parents_ind = np.random.choice(mating_pool_size, 2, replace = False)
        parent_1 = mating_pool[parents_ind[0]]
        parent_2 = mating_pool[parents_ind[1]]

        child = (parent_1.genotype + parent_2.genotype)/2

        children.append(child)
        
    return children

def choose_gene(genotype_1, genotype_2):
        
        prob = np.random.uniform(size = genotype_1.shape) < 0.5
        prob_to_int = prob.astype(int)

        child_genotype = genotype_1 * prob_to_int + genotype_2 * np.logical_not(prob_to_int)
        
        return child_genotype

def uniform_crossover(mating_pool, crossover_percentage = 0.8): 

    mating_pool_size = len(mating_pool)
    mating_number = 2 * round(mating_pool_size * crossover_percentage)

    children = []

    for _ in range(mating_number):

        parents_ind = np.random.choice(mating_pool_size, 2, replace = False)
        parent_1 = mating_pool[parents_ind[0]]
        parent_2 = mating_pool[parents_ind[1]]

        child = choose_gene(parent_1.genotype, parent_2.genotype)
        children.append(child)
    
    return children
