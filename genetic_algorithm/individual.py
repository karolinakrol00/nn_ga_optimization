import numpy as np

class Individual:

    def __init__(self, genotype_blueprint, genotype = None):
        self.genotype = genotype
        self.genotype_blueprint = genotype_blueprint
        self.fitness = 0

    def generate_genotype(self):
        genotype = []
        gene_rules = list(self.genotype_blueprint.values())
        for gene_rule in gene_rules:
            gene_type = gene_rule[0]
            gene_param = gene_rule[1]
            
            if gene_type == 'exp':
                gene_value = np.random.exponential(gene_param)
            elif gene_type == 'unif':
                gene_value = np.random.uniform(*gene_param)
            elif gene_type == 'str':
                gene_value = np.random.choice(gene_param)
            elif gene_type == 'max_topology_size':
                gene_value = np.array([1, *[0] * (gene_param - 1)])
            elif gene_type == 'weights':
                gene_value = np.random.normal(0, 1, gene_param)

            if gene_type == 'weights':
                genotype = gene_value
            else:
                genotype.append(gene_value)

        return genotype

    def assign_generated_genotype(self):
        self.genotype = self.generate_genotype()
        return self