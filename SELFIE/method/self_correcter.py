import numpy as np
import operator
from structure.minibatch import *
from structure.sample import *

class Correcter(object):
    def __init__(self, size_of_data, num_of_classes, queue_size, threshold, loaded_data=None):
        self.size_of_data = size_of_data
        self.num_of_classes = num_of_classes
        self.queue_size = queue_size
        self.threshold = threshold

        # prediction histories of samples
        self.all_predictions = {}
        for i in range(size_of_data):
            self.all_predictions[i] = np.zeros(queue_size, dtype=int)

        # Max Correctablility
        max_variance = (float(self.num_of_classes) - 1.0) / (float(self.num_of_classes) * float(self.num_of_classes))
        self.max_correctability = max_variance + max_variance * max_variance / (float(queue_size) - 1.0)
        self.max_certainty = -np.log(1.0/float(self.num_of_classes))

        # Corrected label map
        self.corrected_labels = {}
        for i in range(size_of_data):
            self.corrected_labels[i] = -1

        self.update_counters = np.zeros(size_of_data, dtype=int)

        # For Logging
        self.loaded_data = None
        if loaded_data is not None:
            self.loaded_data = loaded_data

    def async_update_prediction_matrix(self, ids, softmax_matrix):
        for i in range(len(ids)):
            id = ids[i]
            predicted_label = np.argmax(softmax_matrix[i])
            # append the predicted label to the prediction matrix
            cur_index = self.update_counters[id] % self.queue_size
            self.all_predictions[id][cur_index] = predicted_label
            self.update_counters[id] += 1

    def separate_clean_and_unclean_samples(self, ids, images, labels, loss_array, noise_rate):
        clean_batch = MiniBatch()
        unclean_batch = MiniBatch()
        num_clean_instances = int(np.ceil(float(len(ids)) * (1.0 - noise_rate)))

        loss_map = {}
        image_map = {}
        label_map = {}

        for i in range(len(ids)):
            loss_map[ids[i]] = loss_array[i]
            image_map[ids[i]] = images[i]
            label_map[ids[i]] = labels[i]

        # sort loss by descending order
        loss_map = dict(sorted(loss_map.items(), key=operator.itemgetter(1), reverse=False))

        index = 0
        for key in loss_map.keys():
            if index < num_clean_instances:
                clean_batch.append(key, image_map[key], label_map[key])
            else:
                unclean_batch.append(key, image_map[key], label_map[key])
            index += 1

        return clean_batch, unclean_batch

    def get_corrected_samples(self, ids, images):
        corrected_batch = MiniBatch()

        # check correctability for each sample
        accumulator = {}
        for i in range(len(ids)):
            id = ids[i]
            image = images[i]

            predictions = self.all_predictions[id]
            accumulator.clear()

            for prediction in predictions:
                if prediction not in accumulator:
                    accumulator[prediction] = 1
                else:
                    accumulator[prediction] = accumulator[prediction] + 1

            p_dict = np.zeros(self.num_of_classes, dtype=float)
            for key, value in accumulator.items():
                p_dict[key] = float(value) / float(self.queue_size)

            # based on variance
            #var = np.var(p_dict)
            #correctability = var + (var * var) / (float(self.queue_size) - 1.0)

            # based on entropy
            negative_entropy = 0.0
            for i in range(len(p_dict)):
                if p_dict[i] == 0:
                    negative_entropy += 0.0
                else:
                    negative_entropy += p_dict[i] * np.log(p_dict[i])
            certainty = - negative_entropy / self.max_certainty

            #if certainty <= self.threshold and not self.loaded_data[id].corrected:
            if certainty <= self.threshold:
                self.corrected_labels[id] = np.argmax(p_dict)
                corrected_batch.append(id, image, self.corrected_labels[id])

                # For logging ###########################################################
                if self.loaded_data is not None:
                    self.loaded_data[id].corrected = True
                    self.loaded_data[id].last_corrected_label = self.corrected_labels[id]
                #########################################################################
            
            #reuse
            elif self.corrected_labels[id] != -1:
                corrected_batch.append(id, image, self.corrected_labels[id])

        return corrected_batch

    def merge_clean_and_corrected_samples(self, clean_batch, corrected_batch):

        final_batch = MiniBatch()
        corrected_batch_ids = set()

        # Add corrected batch
        for i in range(len(corrected_batch.ids)):
            corrected_batch_ids.add(corrected_batch.ids[i])
            final_batch.append(corrected_batch.ids[i], corrected_batch.images[i], corrected_batch.labels[i])

        # Merge with clean batch
        # If clean samples are included in both clean_batch and corrected_batch, then the samples in corrected_batch are chosen.
        for i in range(len(clean_batch.ids)):
            if clean_batch.ids[i] in corrected_batch_ids:
                continue

            if self.corrected_labels[clean_batch.ids[i]] != -1:
                # if the sample was corrected at previous epoch, we reuse the corrected label for current mini-batch
                final_batch.append(clean_batch.ids[i], clean_batch.images[i], self.corrected_labels[clean_batch.ids[i]])
            else:
                final_batch.append(clean_batch.ids[i], clean_batch.images[i], clean_batch.labels[i])

        return final_batch.ids, final_batch.images, final_batch.labels

    def patch_clean_with_corrected_sample_batch(self, ids, images, labels, loss_array, noise_rate):
        # 1. separate clean and unclean samples
        clean_batch, unclean_batch = self.separate_clean_and_unclean_samples(ids, images, labels, loss_array,
                                                                             noise_rate)
        # 2. get corrected samples
        corrected_batch = self.get_corrected_samples(ids, images)
        # 3. merging
        return self.merge_clean_and_corrected_samples(clean_batch, corrected_batch)

    def compute_new_noise_ratio(self):
        num_corrected_sample = 0
        for key, value in self.corrected_labels.items():
            if value != -1:
                num_corrected_sample += 1

        return 1.0 - float(num_corrected_sample) / float(self.size_of_data)

    def predictions_clear(self):
        self.all_predictions.clear()
        for i in range(self.size_of_data):
            self.all_predictions[i] = np.zeros(self.queue_size, dtype=int)
