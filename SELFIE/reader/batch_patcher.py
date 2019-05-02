import numpy as np
import os, math, sys
from structure.minibatch import *
from structure.sample import *

def bytes_to_int(bytes_array):
    result = 0
    for b in bytes_array:
        result = result * 256 + int(b)
    return result


class BatchPatcher(object):
    def __init__(self, size_of_data, batch_size, num_of_classes, replacement = False):
        self.size_of_data = size_of_data
        self.batch_size = batch_size
        self.num_of_classes = num_of_classes
        self.num_iters_per_epoch = int(math.ceil(float(size_of_data) / float(batch_size)))

        # For in-memory data
        self.loaded_data = []

        # Replacement in mini-batch for random batch selection
        self.replacement = replacement

        # Custom noise
        self.noise_rate = 0.0
        self.transition_matrix = None

    def bulk_load_in_memory(self, sess, ids, images, labels):
        # initialization
        self.loaded_data.clear()

        for i in range(self.size_of_data):
            self.loaded_data.append(None)

        # load data set in memory
        set_test = set()
        for i in range(self.num_iters_per_epoch * 2):
            mini_ids, mini_images, mini_labels = sess.run([ids, images, labels])

            for j in range(self.batch_size):
                id = bytes_to_int(mini_ids[j])

                if not id in set_test:
                    self.loaded_data[id] = Sample(id, mini_images[j], bytes_to_int(mini_labels[j]))
                    set_test.add(id)

        print("# of samples: ", len(self.loaded_data))

    def get_next_random_mini_batch(self, num_of_samples):
        selected_sample_ids = np.random.choice(self.size_of_data, num_of_samples, self.replacement)

        # Fetch mini-batch samples from loaded_data in main memory
        mini_batch = MiniBatch()
        for id in selected_sample_ids:
            sample = self.loaded_data[id]
            mini_batch.append(sample.id, sample.image, sample.label)

        return mini_batch.ids, mini_batch.images, mini_batch.labels

    def get_init_mini_batch(self, init_id):
        # init_id from 0~self.num_iters_per_epoch
        selected_sample_ids = list(range(init_id*self.batch_size, init_id*self.batch_size + self.batch_size))

        # Fetch mini-batch samples from loaded_data in main memory
        mini_batch = MiniBatch()
        for id in selected_sample_ids:
            if id >= self.size_of_data:
                continue
            else:
                sample = self.loaded_data[id]
                mini_batch.append(sample.id, sample.image, sample.label)

        return mini_batch.ids, mini_batch.images, mini_batch.labels

    def set_noise(self, noise_rate, noise_type):
        if noise_type != "symmetry" and noise_type != "pair" and noise_type != "none":
            print("Noise type error!")
            sys.exit(1)
        print("Noise Injection: ", noise_type)

        self.noise_rate = noise_rate
        self.transition_matrix = []
        for i in range(self.num_of_classes):
            self.transition_matrix.append([])
            for j in range(self.num_of_classes):
                self.transition_matrix[i].append(0.0)

        if noise_type == "symmetry":
            for i in range(self.num_of_classes):
                for j in range(self.num_of_classes):
                    if i == j:
                        self.transition_matrix[i][j] = 1.0 - self.noise_rate
                    else:
                        self.transition_matrix[i][j] = noise_rate / float(self.num_of_classes - 1)
        elif noise_type == "pair":
            for i in range(self.num_of_classes):
                self.transition_matrix[i][(i + 1) % self.num_of_classes] = noise_rate
                for j in range(self.num_of_classes):
                    if i == j:
                        self.transition_matrix[i][j] = 1.0 - noise_rate
        elif noise_type == "none":
            for i in range(self.num_of_classes):
                for j in range(self.num_of_classes):
                    if i == j:
                        self.transition_matrix[i][j] = 1.0

        for sample in self.loaded_data:
            sample.label = self.get_noise_label(self.transition_matrix[sample.true_label])
            sample.last_corrected_label = sample.label
            sample.first_label = sample.label

    def set_custom_noise(self, path):
        if not os.path.exists(path):
            print("Custom noise transition matrix is not exist in the path!")

        self.transition_matrix = []
        with open(path) as fp:
            for line in fp:
                temp = []
                for item in line.split(","):
                    temp.append(float(item))
                self.transition_matrix.append(temp)

        for sample in self.loaded_data:
            sample.label = self.get_noise_label(self.transition_matrix[sample.true_label])
            sample.last_corrected_label = sample.label
            sample.first_label = sample.label


    def get_noise_label(self, transition_array):

        return np.random.choice(self.num_of_classes, 1, True, p=transition_array)[0]

    def print_transition_matrix(self, transition_matrix, log = None):
        log_str = ""
        for i in range(len(transition_matrix)):
            for j in range(len(transition_matrix[i])):
                print(str(transition_matrix[i][j]) + " ", end=',')
                log_str += str(transition_matrix[i][j]) + ", "
            print("\n")
            log_str += "\n"
        log_str += "\n"
        if log is not None:
            log.append(log_str)

    def get_current_noise_matrix(self, entire=True):
        noise_matrix = np.zeros([self.num_of_classes, self.num_of_classes], dtype = int)
        for sample in self.loaded_data:
            if entire or sample.corrected:
                noise_matrix[sample.true_label][sample.last_corrected_label] += 1
        return noise_matrix
