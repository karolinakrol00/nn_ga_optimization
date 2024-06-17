from sklearn import datasets
from hyperparameters_optimization import optimize_hyperparameters_by_ga
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from metrics_and_transformations import normalize
import os

rs = np.random.randint(1)

# prepare dataset
bc = datasets.load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(bc.data, bc.target, test_size = 0.2, random_state = rs)
X_train = normalize(X_train)
X_test = normalize(X_test)

# prepare hyperparameters to optimize as dictionary
params = dict(learning_rate_init = ['exp', 0.1], 
              batch_size = ['unif', [1, 50]], 
              activation = ['str', ['logistic', 'tanh', 'relu', 'identity']],
              solver = ['str', ['lbfgs', 'adam']],
              hidden_layer_sizes = ['max_topology_size', 4],
              alpha = ['exp', 1/1000])

n_generations = 5
n_individuals = 30
results = pd.DataFrame()

for i in range(20):

    # Optimization of hyperparametrs by genetic algorithm
    result_new = optimize_hyperparameters_by_ga(X_train, X_test, y_train, y_test, params, n_individuals, n_generations)
    results = pd.concat([results, pd.DataFrame(result_new)], ignore_index=True)

relative_path = os.path.dirname(__file__)
output_path = relative_path + '\hyperparateters_diabetes_results.csv'
results.to_csv(output_path)

