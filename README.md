# DL-assignment-1

MNIST Neural Network Classifier

This repository contains an implementation of a flexible neural network for classifying handwritten digits from the MNIST dataset using TensorFlow and Keras.

Features

Loads and preprocesses the MNIST dataset.
Provides a customizable neural network architecture with adjustable layers and neurons.
Supports different optimizers and weight initializations.
Trains the model with user-defined hyperparameters.
Evaluates the model with accuracy and confusion matrix visualization.
Compares cross-entropy and mean squared error loss functions.
Uses train_test_split to create a validation set.
Implements various weight initializers such as Xavier (Glorot Normal) and Random Normal.


Code Overview:
load_data(): Loads and preprocesses the MNIST dataset by normalizing images and converting labels to categorical format.
plot_samples(x_train): Displays sample images from the dataset for verification.
create_model(hidden_layers, hidden_units, activation, optimizer, weight_init): Builds a flexible neural network with adjustable parameters.
train_model(x_train, y_train, x_val, y_val, model, batch_size, epochs): Trains the model with early stopping support.
evaluate_model(model, x_test, y_test): Evaluates the trained model, calculates accuracy, and plots a confusion matrix for performance analysis.
compare_loss_functions(): Trains two models separately with cross-entropy and mean squared error losses and compares their performance graphically.\


Model Training:

By default, the model is trained with:
3 hidden layers
64 neurons per layer
ReLU activation function
Adam optimizer for efficient training
Xavier (Glorot Normal) weight initialization for stable gradients
Mini-batch size of 32
10 training epochs
These settings can be modified in train.py when calling create_model().

Evaluation:
After training, the model is evaluated on the test set, displaying accuracy and a confusion matrix. Loss comparison between cross-entropy and mean squared error is also visualized.

Confusion Matrix:
A confusion matrix is plotted using Seaborn to analyze classification errors and improve model performance.

Hyperparameter Tuning:
You can modify the following hyperparameters in create_model():
Number of hidden layers
Number of neurons per layer
Activation functions (relu, sigmoid, tanh, etc.)
Optimizer choice (adam, sgd, rmsprop, etc.)
Weight initialization methods
Batch size and number of epochs

Possible Enhancements

Implement dropout for better regularization.
Introduce batch normalization to stabilize training.
Explore different activation functions.
Use data augmentation for improved generalization.
Try convolutional neural networks (CNNs) for better accuracy.
