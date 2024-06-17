from sklearn.neural_network import MLPClassifier
import numpy as np
from typing import Any, Dict
import warnings
from sklearn.exceptions import ConvergenceWarning
from genetic_algorithm import Population, k_point_crossover, random_mutation, add_topology_element, add_neurons, remove_neurons, remove_topology_element

warnings.filterwarnings(action = 'ignore', category = ConvergenceWarning)

def optimize_hyperparameters_by_ga(X_train: np.ndarray[Any, np.dtype[np.float32]], X_test: np.ndarray[Any, np.dtype[np.float32]],
                                   y_train: np.ndarray[Any, np.dtype[np.float32]], y_test: np.ndarray[Any, np.dtype[np.float32]],
                                   hyperparameters_to_optimize: Dict[str, int], 
                                   population_size: int,
                                   n_generations: int) -> Dict[str, int]:

    """
    Optimizes hyperparameters of sklearn MLPClassifier using a genetic algorithm.

    Parameters:
    - X_train, y_train: Training data and labels
    - X_test, y_test: Testing data and labels
    - hyperparameters_to_optimize: Dictionary of hyperparameters to optimize
    - population_size: Size of the population for the genetic algorithm
    - n_generations: Number of generations for the genetic algorithm
    Returns:
    - Dictionary of optimized hyperparameters
    """
    acc_over_generations = []
    # Initialize population with random individuals
    population = Population(population_size, hyperparameters_to_optimize)

    # Extract parameter names and types from the hyperparameters dictionary
    params_names = list(hyperparameters_to_optimize.keys())

    def prepare_params(params_dict):

        """
        Prepares parameters to the form accepted by MLPClassifier.
        """

        if 'batch_size' in params_dict: params_dict['batch_size'] = round(params_dict['batch_size'])
        if 'max_iter' in params_dict: params_dict['max_iter'] = round(params_dict['max_iter'])
        if 'hidden_layer_sizes' in params_dict: params_dict['hidden_layer_sizes'] = tuple(filter(lambda a: a != 0, params_dict['hidden_layer_sizes']))
        return params_dict


    def build_nn_and_evaluate():

        """
        Evaluates the fitness of each individual in the population by training an MLPClassifier and scoring it on the test set.
        """

        for individual in population.individuals:
            if individual.fitness is None:

                # Prepare parameters for the MLPClassifier
                params_dict = dict(zip(params_names, individual.genotype))
                prepare_params(params_dict)

                # Train MLPClassifier and evaluate its performance
                clf = MLPClassifier(**params_dict, random_state=78).fit(X_train, y_train)
                individual.fitness = clf.score(X_test, y_test)

    max_fitness = 0
    generation = 1

    # Main loop of the genetic algorithm
    while generation < n_generations + 1:

        # Perform crossover and evaluate individuals
        population.crossover(k_point_crossover, 2)
        build_nn_and_evaluate()

        i = 0
        # Apply mutation based on parameter type and evaluate individuals
        for param_type in params_names:
            if param_type == 'hidden_layer_sizes':
                population.mutation(add_topology_element, 0.1, i)
                population.mutation(add_neurons, 0.1, i)
                population.mutation(remove_neurons, 0.05, i)
                population.mutation(remove_topology_element, 0.02, i)
            else:
                population.mutation(random_mutation, 0.1, i)

            i += 1

        build_nn_and_evaluate()
        # Update max_fitness and print progress
        max_fitness = population.get_the_fittest(1)[0].fitness
        acc_over_generations.append(max_fitness)
        print(f'Generation number: {generation}. Accuracy: {max_fitness}')
        generation += 1
        fittest_genotype = population.get_the_fittest(1)[0].genotype

    params_dict = dict(zip(params_names, fittest_genotype))
    prepare_params(params_dict)
    clf = MLPClassifier(**params_dict, random_state=78).fit(X_train, y_train)
    predicted = clf.predict(X_test)
    predicted_proba = clf.predict_proba(X_test)

    return {'predicted': [predicted], 'pred_proba': [predicted_proba], 'accuracy': [acc_over_generations], 'best_architecture': [fittest_genotype], 'final_acc': acc_over_generations[-1]}
