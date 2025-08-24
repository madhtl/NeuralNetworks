import numpy as np

def extend_input_with_bias(network_input):
    bias = np.ones(network_input.shape[1]).reshape(1, -1) # 1D array -> 2D row vector; -1 = network_input.shape[1]
    network_input = np.vstack([bias, network_input])
    return network_input


def create_network(input_size, output_size, hidden_sizes):
    network = []  # list of hidden layers
    layer_size = hidden_sizes
    layer_size.append(output_size)
    for neuron_count in layer_size:
        layer = np.random.rand(input_size +1, neuron_count)*2-1 # it only returns values from 0 to 1, thats why multiplying- and allowing negative values appearing by substracting 1 so [-1,1] range
        input_size = neuron_count
        network.append(layer)
    return network



def unipolar_activation(u):
    return 1 / (1 + np.exp(-u))


def unipolar_derivative(d):
    d = unipolar_activation(d)
    return d*(1-d)


def feed_forward(network_input, network):
    layer_input = network_input
    responses = []
    for weights in network:
        layer_input = extend_input_with_bias(layer_input)
        response = unipolar_activation(weights.T @ layer_input)
        layer_input = response
        responses.append(response)
    return responses


def predict(network_input, network):
    return feed_forward(network_input, network)[-1]


def calculate_mse(predicted, expected):
    return np.sum((predicted - expected) ** 2) / len(predicted)


def backpropagate(network, responses, expected_output_layer_response):
    gradients = []
    error = responses[-1] - expected_output_layer_response
    for weights, response in zip(reversed(network), reversed(responses)):
        gradient = error*unipolar_derivative(response)
        gradients.append(gradient)
        error = weights @ gradient
        error = error[1:,:]
    return list(reversed(gradients))


def calculate_weights_changes(network, network_input, network_respones, gradients, learning_factor):
    layer_inputs = [network_input] + network_respones[:-1]
    weights_changes =[]
    for weights, layer_input, gradient in zip(network, layer_inputs, gradients):
        layer_input = extend_input_with_bias(layer_input)
        print(layer_input)
        change = layer_input.dot(gradient.T)*learning_factor
        weights_changes.append(change)
    return weights_changes


def adjust_weigths(network, changes):
    new_network = []
    for weights, changes in zip(network, changes):
        new_weights = weights - changes
        new_network.append(new_weights)
    return new_network


def train_network(network, network_input, expected_output, learning_factor, epochs):
    mse_history = []
    for _ in range(epochs):
        responses = feed_forward(network_input, network)
        mse_history.append(calculate_mse(responses[-1], expected_output))
        gradients = backpropagate(network, responses, expected_output)
        changes = calculate_weights_changes(network, network_input, responses, gradients, learning_factor)
        network = adjust_weigths(network, changes)
    mse_history.append(calculate_mse(responses[-1], expected_output))
    return network, np.asarray(mse_history)

