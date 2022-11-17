import numpy as np


def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1/(1+np.exp(-x))


def sigmoid_prime(x):
    """
    # Derivative of the sigmoid function
    """
    return sigmoid(x) * (1 - sigmoid(x))


learnrate = 0.5
x = np.array([1, 2, 3, 4])
y = np.array(0.5)

# Initial weights
w = np.array([0.5, -0.5, 0.3, 0.1])

# Calculate one gradient descent step for each weight

# Calculate the node's linear combination of inputs and weights
h = sum([(x[idx] * weight) for idx, weight in enumerate(w)])

# Calculate output of neural network (y-hat)
nn_output = sigmoid(x=h)

# Calculate error of neural network (y - y-hat)
error = y - nn_output

# Calculate output gradient (f'(h))
output_grad = sigmoid_prime(h)

# Calculate the error term (lowercase delta)
error_term = error * output_grad

# Calculate change in weights
del_w = [(learnrate * error_term * input) for input in x]

print('Neural Network output:')
print(nn_output)
print('Amount of Error:')
print(error)
print('Change in Weights:')
print(del_w)
