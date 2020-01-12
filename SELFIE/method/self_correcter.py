import numpy as np
import operator
from structure.minibatch import *
from structure.sample import *

class Correcter(object):
    def __init__(self, size_of_data, num_of_classes, history_length, threshold, loaded_data=None):
        self.size_of_data = size_of_data
        self.num_of_classes = num_of_classes
        self.history_length = history_length
        self.threshold = threshold

        # prediction histories of samples
        self.all_predictions = {}
        for i in range(size_of_data):
            self.all_predictions[i] = np.zeros(history_length, dtype=int)

        # Max predictive uncertainty
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
            cur_index = self.update_counters[id] % self.history_length
            self.all_predictions[id][cur_index] = predicted_label
            self.update_counters[id] += 1

    # low-loss separation
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

    def get_refurbishable_samples(self, ids, images):
        corrected_batch = MiniBatch()

        # check predictive uncertainty
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
                p_dict[key] = float(value) / float(self.history_length)

            # compute predictive uncertainty
            negative_entropy = 0.0
            for i in range(len(p_dict)):
                if p_dict[i] == 0:
                    negative_entropy += 0.0
                else:
                    negative_entropy += p_dict[i] * np.log(p_dict[i])
            certainty = - negative_entropy / self.max_certainty

            ############### correspond to the lines 12--19 of the paper ################
            # check refurbishable condition
            if certainty <= self.threshold:
                self.corrected_labels[id] = np.argmax(p_dict)
                corrected_batch.append(id, image, self.corrected_labels[id])

                # For logging ###########################################################
                if self.loaded_data is not None:
                    self.loaded_data[id].corrected = True
                    self.loaded_data[id].last_corrected_label = self.corrected_labels[id]
                #########################################################################

            # reuse previously classified refurbishalbe samples
            # As we tested, this part degraded the performance marginally around 0.3%p
            # because uncertainty of the sample may afftect the performance
            elif self.corrected_labels[id] != -1:
                corrected_batch.append(id, image, self.corrected_labels[id])

        return corrected_batch

    def merge_clean_and_corrected_samples(self, clean_batch, corrected_batch):

        final_batch = MiniBatch()
        corrected_batch_ids = set()

        # add clean batch
        for i in range(len(corrected_batch.ids)):
            corrected_batch_ids.add(corrected_batch.ids[i])
            final_batch.append(corrected_batch.ids[i], corrected_batch.images[i], corrected_batch.labels[i])

        # merge clean with refurbishable samples
        # If a sample is included in clean_batch and refurbishable_batch at the same time, then the samples is treated as refurbishable
        for i in range(len(clean_batch.ids)):
            if clean_batch.ids[i] in corrected_batch_ids:
                continue

            if self.corrected_labels[clean_batch.ids[i]] != -1:
                # if the sample was corrected at previous epoch, we reuse the corrected label for current mini-batch
                final_batch.append(clean_batch.ids[i], clean_batch.images[i], self.corrected_labels[clean_batch.ids[i]])
            else:
                final_batch.append(clean_batch.ids[i], clean_batch.images[i], clean_batch.labels[i])

        return final_batch.ids, final_batch.images, final_batch.labels

    def patch_clean_with_refurbishable_sample_batch(self, ids, images, labels, loss_array, noise_rate):
        # 1. separate clean and unclean samples
        clean_batch, unclean_batch = self.separate_clean_and_unclean_samples(ids, images, labels, loss_array, noise_rate)
        # 2. get refurbishable samples
        corrected_batch = self.get_refurbishable_samples(ids, images)
        # 3. merging
        return self.merge_clean_and_corrected_samples(clean_batch, corrected_batch)

    def predictions_clear(self):
        self.all_predictions.clear()
        for i in range(self.size_of_data):
            self.all_predictions[i] = np.zeros(self.history_length, dtype=int)

    def compute_new_noise_ratio(self):
        num_corrected_sample = 0
        for key, value in self.corrected_labels.items():
            if value != -1:
                num_corrected_sample += 1

        return 1.0 - float(num_corrected_sample) / float(self.size_of_data)