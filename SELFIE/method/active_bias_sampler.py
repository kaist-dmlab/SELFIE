import numpy as np
import time, os, math, operator, statistics, sys
import tensorflow as tf
from random import Random
from structure.minibatch import *
from structure.sample import *

class Sampler(object):
    def __init__(self, size_of_data, num_of_classes, smoothness = 0.0, loaded_data=None):
        self.size_of_data = size_of_data
        self.num_of_classes = num_of_classes
        self.smoothness = smoothness

        # prediction histories of samples
        self.all_probabilities = {}
        for i in range(size_of_data):
            self.all_probabilities[i] = []

        # Corrected weight map
        self.sample_weights = {}
        for i in range(size_of_data):
            self.sample_weights[i] = 0.0

        # For Logging
        self.loaded_data = None
        if loaded_data is not None:
            self.loaded_data = loaded_data

    def async_update_probability_matrix(self, ids, labels, softmax_matrix):
        for i in range(len(ids)):
            id = ids[i]
            label = labels[i]
            # prediction probability of target label
            probability = softmax_matrix[i][label]
            # append the prediction probability to the map
            self.all_probabilities[id].append(probability)

    def compute_sample_weights(self, ids, uniform=False):

        weights = []

        if uniform:
            for i in range(len(ids)):
                weights.append(1.0)
        else:
            total_sum = 0.0
            for i in range(len(ids)):
                id = ids[i]
                probabilities = self.all_probabilities[id]
                variance = np.var(probabilities)
                weight = variance + (variance * variance)/(float(len(probabilities))-1.0)
                weight = np.sqrt(weight) + self.smoothness
                weights.append(weight)
                total_sum += weight

            for i in range(len(weights)):
                # normalized and 1/N => 1/N*N ==> weighted average, not average
                weights[i] = (weights[i]*float(len(ids))/total_sum)
        
        # the output is not ordered by id, just follows the order of patch
        return weights

    def compute_new_noise_ratio(self):
        num_corrected_sample = 0
        for key, value in self.corrected_labels.items():
            if value != -1:
                num_corrected_sample += 1

        return 1.0 - float(num_corrected_sample) / float(self.size_of_data)
