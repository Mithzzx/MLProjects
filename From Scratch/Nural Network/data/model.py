import h5py
import os
import numpy as np

file_path = 'train_catvnoncat.h5'

# Initialize variables to store the data
x_train = None
y_train = None

# Check if the file exists
if os.path.exists(file_path):
    with h5py.File(file_path, 'r') as file:
        # Extract the datasets
        x_train = file['train_set_x'][:]
        y_train = file['train_set_y'][:]
else:
    print(f"File not found: {file_path}")

# Print the shapes of the loaded data
print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")

# Reshape x_train to a single-dimensional vector for each image
x_train_flattened = x_train.reshape(x_train.shape[0], -1)

# Print the new shape of x_train
print(f"x_train_flattened shape: {x_train_flattened.shape}")

# Normalize the pixel values
x_train_normalized = x_train_flattened / 255

# Reshape y_train to be a 2-dimensional array
y_train = y_train.reshape(1, -1)

# first layer of the neural network
n_x = x_train_normalized.shape[1]
n_y = 1

# Initialize the weights and biases
W1 = np.random.randn(4, n_x) * 0.01
b1 = np.zeros((4, 1))

W2 = np.random.randn(2, 4) * 0.01
b2 = np.zeros((2, 1))

W3 = np.random.randn(1, 2) * 0.01
b3 = np.zeros((1, 1))

# define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# define the forward propagation
def forward_prop(X, W, b):
    Z = np.dot(W, X) + b
    A = sigmoid(Z)
    return A

# define the cost function
def compute_cost(A, Y):
    m = Y.shape[1]
    cost = -1/m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    return cost

# define the backward propagation
def backward_prop(X, Y, A):
    m = Y.shape[1]
    dZ = A - Y
    dW = 1/m * np.dot(dZ, X.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    return dW, db

# define the update parameters function
def update_parameters(W, b, dW, db, learning_rate):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return W, b

# define the model
def model(X, Y, W1, b1, W2, b2, W3, b3, learning_rate, num_iterations):
    for i in range(num_iterations):
        # Forward propagation
        A1 = forward_prop(X.T, W1, b1)
        A2 = forward_prop(A1, W2, b2)
        A3 = forward_prop(A2, W3, b3)

        # Compute cost
        cost = compute_cost(A3, Y)

        # Backward propagation
        dW3, db3 = backward_prop(A2, Y, A3)
        dW2, db2 = backward_prop(A1, Y, A2)
        dW1, db1 = backward_prop(X.T, Y, A1)

        # Update parameters
        W3, b3 = update_parameters(W3, b3, dW3, db3, learning_rate)
        W2, b2 = update_parameters(W2, b2, dW2, db2, learning_rate)
        W1, b1 = update_parameters(W1, b1, dW1, db1, learning_rate)

        # Print the cost every 100 iterations
        if i % 100 == 0:
            print(f"Cost after iteration {i}: {cost}")

    return W1, b1, W2, b2, W3, b3

# Train the model
W1, b1, W2, b2, W3, b3 = model(x_train_normalized, y_train, W1, b1, W2, b2, W3, b3, 0.008, 10000)

# Save the trained model
np.savez('model.npz', W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)

# test the model
file_path = 'test_catvnoncat.h5'

# Initialize variables to store the data
x_test = None
y_test = None

# Check if the file exists
if os.path.exists(file_path):
    with h5py.File(file_path, 'r') as file:
        # Extract the datasets
        x_test = file['test_set_x'][:]
        y_test = file['test_set_y'][:]

# Print the shapes of the loaded data
print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Reshape x_test to a single-dimensional vector for each image
x_test_flattened = x_test.reshape(x_test.shape[0], -1)

# Print the new shape of x_test
print(f"x_test_flattened shape: {x_test_flattened.shape}")

# Normalize the pixel values
x_test_normalized = x_test_flattened / 255

# Reshape y_test to be a 2-dimensional array
y_test = y_test.reshape(1, -1)

# Load the trained model
model = np.load('model.npz')
W1 = model['W1']
b1 = model['b1']
W2 = model['W2']
b2 = model['b2']
W3 = model['W3']
b3 = model['b3']

# Forward propagation
A1 = forward_prop(x_test_normalized.T, W1, b1)
A2 = forward_prop(A1, W2, b2)
A3 = forward_prop(A2, W3, b3)

# Compute the accuracy
predictions = A3 > 0.5
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy}")





