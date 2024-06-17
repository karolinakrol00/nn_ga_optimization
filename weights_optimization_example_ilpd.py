from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from neural_network import Network, FCLayer, ActivationLayer, tanh, sigmoid, binary_cross_entropy
from weights_optimization import fit_ga
from ucimlrepo import fetch_ucirepo
from metrics_and_transformations import normalize, one_hot_encoder, binary_accuracy, specificity, sensitivity
import os
from imblearn.over_sampling import RandomOverSampler

# prepare data
ilpd = fetch_ucirepo(id = 225)  
X = ilpd.data.features 
y = ilpd.data.targets
X['target'] = y
X = X.dropna(how = 'any')
y = X[['target']]
X = X.drop('target', axis = 1)
le = LabelEncoder()
X['Gender'] = le.fit_transform(X['Gender'])
oversampler = RandomOverSampler(random_state = np.random.randint(1))
X_resampled, y_resampled = oversampler.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size = 0.2, random_state = np.random.randint(1))
X_train_normalized = normalize(np.array(X_train))
X_test_normalized = normalize(np.array(X_test))
y_train_encoded = one_hot_encoder(np.array(np.abs(y_train['target'] - 2)))
y_test_encoded = np.array(np.abs(y_test['target'] - 2))

n_attributes = X_train.shape[1]
n_classes = y_train_encoded.shape[1]

errors_list = []
accuracy_list = []
specificity_list = []
sensitivity_list = []
predictions_list = []

n_generations = 50
n_individuals = 100
mutation_probability = 0.7

for i in range(20):
    
    # neural netork architecture
    net = Network()
    net.add(FCLayer(n_attributes, 5))
    net.add(ActivationLayer(sigmoid))
    net.add(FCLayer(5, 10))
    net.add(ActivationLayer(tanh))
    net.add(FCLayer(10, n_classes))
    net.add(ActivationLayer(sigmoid))
    net.use(binary_cross_entropy)

    # neural network optimzation using genetic algorithm
    errors = fit_ga(net, X_train_normalized.reshape((X_train.shape[0], 1, n_attributes)), y_train_encoded, n_individuals, n_generations, mutation_probability)
    errors_list.append(errors)

    # calculate metrics
    predicted = net.predict(X_test_normalized.reshape((X_test.shape[0], 1, n_attributes)))
    est = [np.argmax(p) for p in np.concatenate(predicted).ravel().reshape(y_test_encoded.shape[0], 2)]
    print(predicted)
    print(est)
    predictions_list.append(est)
    acc = binary_accuracy(est, y_test_encoded)
    accuracy_list.append(acc)
    spec = specificity(y_test_encoded, est)
    specificity_list.append(spec)
    sens = sensitivity(y_test_encoded, est)
    sensitivity_list.append(sens)
    print(f'Accuracyr: {acc}. Specificity: {spec}. Sensitivity: {sens}')

# load results to csv file
results_df = pd.DataFrame({'errors': errors_list, 'accuracy': accuracy_list, 'predictions': predictions_list, 'specificity': specificity_list, 'sensitivity': sensitivity_list})
relative_path = os.path.dirname(__file__)
output_path = relative_path + '\weights_ilpd_results.csv'
results_df.to_csv(output_path)