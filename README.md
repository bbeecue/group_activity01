# CSC173 Activity 01 - Neural Network from Scratch

**Date:** October 09, 2025  
**Team:** [Paalisbo Kervin, Belvis Febe, Joshua Adlaon ]

## Project Overview

This project implements a simple neural network for binary classification using breast cancer diagnostic data. The network is built completely from scratch using only Python and NumPy, with no machine learning libraries. The goal is to deepen understanding of neural network fundamentals including forward propagation, loss computation, backpropagation, gradient descent training, and model evaluation.

## Data Preparation

We used the Breast Cancer Wisconsin Diagnostic dataset obtained from these sources:
- [Scikit-learn breast cancer dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
- [UCI Machine Learning Repository (Breast Cancer Wisconsin Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)  

We selected two features from the dataset for the input layer of the network, namely, concave_points3, and perimeter3. The selection of these features was done by identifying the two most correlated features with the target 'diagnosis'. Using panda's built-in cor function, we were able to see that concavepoints3 and perimeter3 correlated to the target the most.

## Network Architecture

- Input layer: 2 neurons (corresponding to selected features)
- Hidden layer: 2 to 4 neurons, activation function: Sigmoid, ReLU, or Tanh
- Output layer: 1 neuron to produce binary classification output

## Implementation Details

- Weight and bias parameters initialized randomly.
- Forward propagation implements layer-wise computations with chosen activation functions.
- Loss computed using Mean Squared Error (MSE).
- Backpropagation calculates gradients of weights and biases.
- Parameters updated using gradient descent.
- Training performed for 800 iterations(epochs)

## Results & Visualization
Final Training Loss: 0.03848911648284588
Test Accuracy: 92.98%

![diagram](images/Table%201.png)
The diagram above shows the training loss over the epochs. The curve is descending which means that the neural network is gradually improving which is showed by the decreasing loss over time.


![diagram](images/Table%202.png)
This second diagram shows how the model separates benign and malignant tumors using the two selected features: concave points and perimeter. We can see in the graph that the model has learned a dividing region in this space where points above the boundary classifies as  malignant and points below the boundary classifies as benign. This visually demonstrates how the model transforms the values of the features into class prediction(malignant or benign). 


## Team Collaboration

Each member contributed to different components of the network:
- Weight and bias initialization
- Forward propagation coding
- Loss function implementation
- Backpropagation and gradient computation
- Training loop and visualization

## How to Run

1. Clone the GitHub repository:
   ```
   git clone https://github.com/bbeecue/group_activity01
   ```
2. Open the Jupyter notebook or Colab file.
3. Run all cells sequentially.
4. Explore training loss plot and decision boundary visualizations.

## Summary

This activity provided hands-on experience in building a neural network without relying on high-level ML frameworks. The group collaboratively developed the model, analyzed its training behavior visually, and demonstrated understanding of fundamental AI concepts through both code and documentation.

Video: link
