import numpy as np
from network import create_network, feed_forward, train_network, predict, calculate_mse, calculate_weights_changes, adjust_weigths, backpropagate
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')


def hot_encoding(y):
    unique_classes, encoded = np.unique(y, return_inverse=True)
    print(unique_classes)
    print(encoded)
    print(np.eye(len(unique_classes))[encoded])
    return np.eye(len(unique_classes))[encoded]


def describe_data(data): # describe -> returns a String
    return f"number of neurons is: {data.shape[0]} and number of datapoints is {data.shape[1]}"


def describe_layer(layer):
    return f"there is {layer.shape[1]} neurons is and {layer.shape[0]} features in inputs"



dataset_path = "gender1.csv"
dataset = pd.read_csv(dataset_path)
no_features = dataset.shape
y = dataset.iloc[:, -1].values
X = dataset.iloc[:, :-1]
print(y)
print(X)
Y = hot_encoding(y)
Y = Y.T
print(Y)
X = pd.get_dummies(dataset.iloc[:, :-1])
X = X.T
network = create_network(X.shape[0], Y.shape[0], [7,6,8])
Y_predicted = predict(X, network)
print(f"MSE = {calculate_mse(Y_predicted, Y)}")
responses = feed_forward(X, network)
for idx, response in enumerate(responses):
    print(f"for response of {idx} layer {describe_data(response)}")
print(np.argmax(Y_predicted, axis = 0))

gradients = backpropagate(network, responses, Y)
for idx, gradient in enumerate(gradients):
    print(f"For gradient of {idx} layer {describe_data(gradient)}")
changes = calculate_weights_changes(network, X, responses,gradients, 0.3)
for idx, change in enumerate(changes):
    print(f"for change of {idx} layer {describe_layer(change)}")
network = adjust_weigths(network, changes)
for idx, layer in enumerate(network):
    print(f"In layer {idx},{describe_layer(layer)}")
Y_predicted = predict(X, network)
print(f"MSE = {calculate_mse(Y_predicted, Y)}")

network = create_network(X.shape[0], Y.shape[0], [7,6])
network, mse_history = train_network(network, X, Y, 0.01, 20)
plt.plot(mse_history)
plt.show()

feature_columns = X.T.columns.tolist()

def get_user_input():
    user_data = {}
    print("Please answer the following:")
    for col in feature_columns:
        val = input(f"{col} (0 or 1): ")
        user_data[col] = int(val)
    return pd.DataFrame(user_data, index=[0])

user_df = get_user_input()
user_df = user_df[feature_columns]
user_input = user_df.to_numpy().T

user_prediction = predict(user_input, network)
predicted_class_index = np.argmax(user_prediction, axis=0)[0]

# Map back to label
unique_labels = np.unique(dataset.iloc[:, -1].values)
print(f"\nPredicted class: {unique_labels[predicted_class_index]}")
