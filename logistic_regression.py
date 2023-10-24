import math
import random
from collections import Counter, defaultdict
from matplotlib import pyplot as plt
import time
import numpy as np
from cmath import exp, pi, sqrt


class LogisticRegression:
    def __init__(self, num_classes, learning_rate=0.1, training_epochs=10):
        self.num_classes = num_classes
        self.weights = []
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs
        

    def softmax(self, weight, vector):
        dot_products = self.dot_product(self.weights, vector)

        _max = np.max(dot_products)
        exp_scores = np.exp(dot_products - _max)
        softmax_probs = exp_scores / sum(exp_scores)
        return softmax_probs

    def dot_product(self, weight, vector):
        return np.dot(weight, vector)
        # product = []
        # for i in range(len(matrix)):
        #     row_dot_product = 0
        #     for j in range(len(array)):
        #         row_dot_product += matrix[i][j] * array[j]
        #     product.append(row_dot_product)
        
        # return product
    
    def outer(self, error, vector):
        return np.outer(error, vector)
    
    
    def zeros(self, shape):
        rows, feature_size = shape
        return [[0 for j in range(feature_size)] for i in range(rows)]

    def train(self, feature_matrix, labels):
        feature_size = len(feature_matrix[0])
        self.weights = self.zeros((self.num_classes, feature_size))
        
        for _ in range(self.training_epochs):
            for features_index, features in enumerate(feature_matrix):

                target = [0]*self.num_classes
                target[labels[features_index]] = 1

                soft_max_score = self.softmax(self.weights, features)

                error = target - soft_max_score
                delta = self.learning_rate * self.outer(error, features)
                self.weights += delta


    def test(self, test_features):
        predictions = []

        for features in test_features:
            probabilities = self.softmax(self.weights, features)
            probabilities = list(probabilities)
            predictions.append(probabilities.index(max(probabilities)))
        
        return predictions
