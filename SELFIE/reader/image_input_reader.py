from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.summary import summary
from tensorflow.python.training import input as tf_input
from six.moves import xrange
import tensorflow as tf

# Fixed Meta Data for Data Read
TRAIN_STR = "data_batch"
TEST_STR = "test_batch"

# Parsing Byte Information : [Image_Id, Image_Label, Red pixels, Green pixels, Blue pixels]
# Id, Label are header data for each image
ID_BYTES = 4
LABEL_BYTES = 4


class Record(object):
    pass


class ImageReader(object):
    def __init__(self, dataset_name, data_path, num_train_files, num_train_images, num_val_images, height, width, depth,
                 num_classes, crop_height=0, crop_width=0):
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.num_train_files = num_train_files
        self.num_train_images = num_train_images
        self.num_val_images = num_val_images
        self.height = height
        self.width = width
        self.depth = depth
        self.num_classes = num_classes
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.record_bytes = ID_BYTES + LABEL_BYTES + self.width * self.height * self.depth

    def read_data_set(self, filename_queue):
        record = Record()
        reader = tf.FixedLengthRecordReader(record_bytes=self.record_bytes)
        file_name, value = reader.read(filename_queue)

        byte_record = tf.decode_raw(value, tf.uint8)

        image_id = tf.strided_slice(byte_record, [0], [ID_BYTES])
        image_label = tf.strided_slice(byte_record, [ID_BYTES], [ID_BYTES + LABEL_BYTES])
        array_image = tf.strided_slice(byte_record, [ID_BYTES + LABEL_BYTES], [self.record_bytes])

        depth_major_image = tf.reshape(array_image, [self.depth, self.height, self.width])
        record.image = tf.transpose(depth_major_image, [1, 2, 0])

        # => height, width, depth order for augment
        # record.image = tf.reshape(array_image, [self.height, self.width, self.depth])

        record.id = image_id
        record.label = image_label

        return record

    def inputs(self, train=True, distort=False, normalize=True):
        filenames = []

        if train:
            for i in xrange(1, self.num_train_files + 1):
                filenames.append(os.path.join(self.data_path, TRAIN_STR + "_" + str(i) + ".bin"))
        else:
            filenames.append(os.path.join(self.data_path, TEST_STR + ".bin"))

        print("Now read following files.")
        print(filenames)

        filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
        record = self.read_data_set(filename_queue)

        # Type casting for nomalization
        record.image = tf.cast(record.image, tf.float32)

        if distort:
            print("Data augmentation is working.")
            if self.crop_width == 0 or self.crop_height == 0:
                print("Crop size is not defined yet.")
            # record.image = tf.random_crop(record.image, [self.crop_height, self.crop_width, self.depth])
            record.image = tf.image.random_flip_left_right(record.image)
            record.image = tf.image.random_brightness(record.image, max_delta=63)
            record.image = tf.image.random_contrast(record.image, lower=0.2, upper=1.8)

        # if not self.crop_height == 32:
        #    record.image = tf.image.resize_images(record.image, [32,32])

        # Normalization
        if normalize:
            record.image = tf.image.per_image_standardization(record.image)

        record.id.set_shape([4, ])
        record.label.set_shape([4, ])

        return record.id, record.image, record.label

    def data_read(self, batch_size, train=True, distort=False, normalize=True, shuffle=False):
        t_id, t_image, t_label = self.inputs(train=train, distort=distort, normalize=normalize)
        if train:
            return generate_image_and_label_batch(t_id, t_image, t_label, batch_size, self.num_train_images, 0.4, 16,
                                                  shuffle=shuffle)
        else:
            return generate_image_and_label_batch(t_id, t_image, t_label, batch_size, self.num_val_images, 0.4, 16,
                                                  shuffle=shuffle)


# Tensorflow API
def generate_image_and_label_batch(s_id, s_image, s_label, batch_size, num_images, min_fraction_of_examples_in_queue,
                                   num_preprocess_threads, shuffle=False):
    # Ensure that the random shuffling has good mixing properties.
    min_queue_examples = int(num_images * min_fraction_of_examples_in_queue)

    print("Filling queue with %d data before starting to train. This will take a few minutes." % min_queue_examples)
    t_id, t_images, t_label = shuffle_batch([s_id, s_image, s_label], batch_size=batch_size,
                                            num_threads=num_preprocess_threads,
                                            capacity=min_queue_examples + 3 * batch_size,
                                            min_after_dequeue=min_queue_examples,
                                            shuffle=shuffle)

    return t_id, t_images, t_label


# Tensorflow API
def shuffle_batch(tensors, batch_size, capacity, min_after_dequeue,
                  num_threads=1, seed=None, enqueue_many=False, shapes=None,
                  allow_smaller_final_batch=False, shared_name=None, name=None, shuffle=True):
    return _custom_shuffle_batch(
        tensors,
        batch_size,
        capacity,
        min_after_dequeue,
        keep_input=True,
        num_threads=num_threads,
        seed=seed,
        enqueue_many=enqueue_many,
        shapes=shapes,
        allow_smaller_final_batch=allow_smaller_final_batch,
        shared_name=shared_name,
        name=name,
        shuffle=shuffle)


# Modified Tensorflow API for FIFO queue, instead of SHUFFLE Queue
def _custom_shuffle_batch(tensors, batch_size, capacity, min_after_dequeue,
                          keep_input, num_threads=1, seed=None, enqueue_many=False,
                          shapes=None, allow_smaller_final_batch=False,
                          shared_name=None, name=None, shuffle=False):
    """Helper function for `shuffle_batch` and `maybe_shuffle_batch`."""

    if context.executing_eagerly():
        raise ValueError(
            "Input pipelines based on Queues are not supported when eager execution"
            " is enabled. Please use tf.data to ingest data into your model"
            " instead.")
    tensor_list = tf_input._as_tensor_list(tensors)
    with ops.name_scope(name, "shuffle_batch",
                        list(tensor_list) + [keep_input]) as name:
        if capacity <= min_after_dequeue:
            raise ValueError("capacity %d must be bigger than min_after_dequeue %d."
                             % (capacity, min_after_dequeue))
        tensor_list = tf_input._validate(tensor_list)
        keep_input = tf_input._validate_keep_input(keep_input, enqueue_many)
        tensor_list, sparse_info = tf_input._store_sparse_tensors(
            tensor_list, enqueue_many, keep_input)
        types = tf_input._dtypes([tensor_list])
        shapes = tf_input._shapes([tensor_list], shapes, enqueue_many)

        ###########################################################################################
        if shuffle:
            queue = data_flow_ops.RandomShuffleQueue(
                capacity=capacity, min_after_dequeue=min_after_dequeue, seed=seed,
                dtypes=types, shapes=shapes, shared_name=shared_name)
        else:
            # Remove shuffle property
            queue = data_flow_ops.FIFOQueue(capacity=capacity, dtypes=types, shapes=shapes, shared_name=shared_name)
        ###########################################################################################

        tf_input._enqueue(queue, tensor_list, num_threads, enqueue_many, keep_input)
        full = (math_ops.to_float(
            math_ops.maximum(0, queue.size() - min_after_dequeue)) *
                (1. / (capacity - min_after_dequeue)))

        summary_name = (
                "fraction_over_%d_of_%d_full" %
                (min_after_dequeue, capacity - min_after_dequeue))
        summary.scalar(summary_name, full)

        if allow_smaller_final_batch:
            dequeued = queue.dequeue_up_to(batch_size, name=name)
        else:
            dequeued = queue.dequeue_many(batch_size, name=name)

        dequeued = tf_input._restore_sparse_tensors(dequeued, sparse_info)

        return tf_input._as_original_type(tensors, dequeued)


def bytes_to_int(bytes_array):
    result = 0
    for b in bytes_array:
        result = result * 256 + int(b)
    return result




