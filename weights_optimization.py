import numpy as np
from neural_network import Network, FCLayer
from genetic_algorithm import Population, k_point_crossover, swap_weight_mutation

def fit_ga(network, n_individuals, x_train, y_train, n_generations, mutation_probability):

    """Optimize neural network weights and biases using genetic algorithm."""

    def assign_weights_and_biases(nn_individual, genotype):

            """Assign weights and biases to the neural network based on the genotype."""

            vector_start = 0

            for layer in nn_individual.layers:
                if isinstance(layer, FCLayer):
                    weights_end = vector_start + layer.input_size * layer.output_size
                    weights_vector = genotype[vector_start:weights_end]
                    biases_end = weights_end + layer.output_size
                    biases_vector = genotype[weights_end:biases_end]
                    layer.weights = np.array(weights_vector).reshape(layer.input_size, layer.output_size)
                    layer.bias = np.array(biases_vector)
                    vector_start = biases_end

    def calculate_fitness(individual, x_train, y_train, network):

        """Calculate the fitness of a neural network individual based on training data."""

        # Create a copy of the network to avoid modifying the original network
        nn_individual = Network()
        nn_individual.__dict__.update(network.__dict__)

        # Assign weights and biases to the neural network based on the individual's genotype
        assign_weights_and_biases(nn_individual, individual.genotype)

        error = 0

        # Forward propagate the input through the network
        for j in range(len(x_train)):
            output = x_train[j]
            for layer in nn_individual.layers:
                output = layer.forward_propagation(output)

            # Network loss is the error 
            error -= nn_individual.loss(y_train[j], output)

        return error

    n_params = sum(layer.input_size * (layer.output_size + 1) for layer in network.layers if isinstance(layer, FCLayer))

    # Create population
    population = Population(n_individuals, dict(weights = ['weights', n_params]))
    errors_over_epochs = []

    for epoch in range(n_generations):
        for individual in population.individuals:
            individual.fitness = calculate_fitness(individual, x_train, y_train, network)
        
        min_error = max([individual.fitness for individual in population.individuals])
        print(f'Generation number: {epoch + 1}. Error: {-1*min_error}')
        errors_over_epochs.append(min_error)

        # Perform crossover and mutation
        population.crossover(k_point_crossover, 0.6, 1)
        population.mutation(swap_weight_mutation, mutation_probability)

    for individual in population.individuals:
            individual.fitness = calculate_fitness(individual, x_train, y_train, network)

    assign_weights_and_biases(network, population.get_the_fittest(1)[0].genotype) 

    return errors_over_epochs