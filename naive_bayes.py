import math
import random
from collections import Counter, defaultdict
from matplotlib import pyplot as plt



class NaiveBayes:
    def __init__(self, total_data, laplace_smoothing):
        self.laplace_smoothing = laplace_smoothing
        self.total_data = total_data
        self.memo = defaultdict(int)
        self.vocabulary = defaultdict(set)
        self.label_frequency = defaultdict(int) 

    def calculate_probability(self, frequency, num_events, voc_size):
        probability = (frequency + self.laplace_smoothing)/(num_events + self.laplace_smoothing*voc_size)
        probability = math.log(probability)
        return probability


    def train(self, feature_matrix, classes):
        
        for features, label in zip(feature_matrix, classes):
            self.label_frequency[label] += 1
        
            for feature_idx, feature_value in enumerate(features):
                memo_key = (feature_idx, feature_value, label)
                self.vocabulary[feature_idx].add(feature_value)
                self.memo[memo_key] += 1

    
    def test(self, feature_matrix):
        predictions = []
        num_unique_classes = len(self.label_frequency)

        for features in feature_matrix:
            probabilities = []
        
            for label, frequency in self.label_frequency.items():
                class_probability = self.calculate_probability(frequency, self.total_data, num_unique_classes)

                probability = class_probability
                for feature_idx, feature_value in enumerate(features):
                    # p(feature/label)
                    feature = (feature_idx, feature_value, label)
                    feature_probability = self.calculate_probability(self.memo[feature], frequency, len(self.vocabulary[feature_idx]))
                    probability += feature_probability

                probabilities.append((label, probability))

            predictions.append(max(probabilities, key=lambda x: x[1])[0])

        return predictions
