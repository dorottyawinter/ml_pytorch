import numpy as np
from data_prep import features, targets, features_test, targets_test


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Use to same seed to make debugging easier
np.random.seed(42)

n_records, n_features = features.shape
last_loss = None

# Initialize weights
weights = np.random.normal(scale=1 / n_features**.5, size=n_features)

# Neural Network hyperparameters
epochs = 1000
learnrate = 0.5

for e in range(epochs):
    del_w = np.zeros(weights.shape)
    for x, y in zip(features.values, targets):
        # Loop through all records, x is the input, y is the target

        # Calculate the output (y-hat), where the dot product is h
        output = sigmoid(x=np.dot(weights, x))

        # Calculate the error (y - y-hat)
        error = y - output

        # Calculate the error term (lowecase delta)
        # The error term
        #   Notice we calulate f'(h) here instead of defining a separate
        #   sigmoid_prime function. This just makes it faster because we
        #   can re-use the result of the sigmoid function stored in
        #   the output variable
        error_term = error * output * (1 - output)

        # Calculate the change in weights for this sample
        # and add it to the total weight change
        del_w += error_term * x

    # Update weights using the learning rate and the average change in weights
    weights += learnrate * (del_w / len(del_w))

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        out = sigmoid(np.dot(features, weights))
        loss = np.mean((out - targets) ** 2)
        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss


# Calculate accuracy on test data
tes_out = sigmoid(np.dot(features_test, weights))
predictions = tes_out > 0.5
accuracy = np.mean(predictions == targets_test)
print(f'Prediction accuracy: {accuracy:.3f}')
