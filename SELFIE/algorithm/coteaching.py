import os, sys, operator
import tensorflow as tf
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from reader import batch_patcher as patcher
from network.DenseNet.DenseNet import *
from network.VGG.VGG19 import *

class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.train_loss_op = None
        self.train_accuracy_op = None
        self.train_op = None
        self.train_xentropy_op = None
        self.train_prob_op = None
        self.test_loss_op = None
        self.test_accuracy_op = None
        self.test_xentropy_op = None
        self.test_prob_op = None

# Recommend T_k = 15 in coteaching paper
def coteaching(gpu_id, input_reader, model_type, total_epochs, batch_size, lr_boundaries, lr_values, optimizer_type, noise_rate, noise_type, T_k=15, log_dir="log"):

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # log list
    text_log = []
    text_log.append("epcoh, learning rate, training loss (network1), training error (network1), training loss (network2), training error (network2), "
                        "test loss (network1), test error (network1), test loss (network2), test error (network2)\n")

    num_train_images = input_reader.num_train_images
    num_test_images = input_reader.num_val_images
    num_label = input_reader.num_classes
    image_shape = [input_reader.height, input_reader.width, input_reader.depth]

    train_batch_patcher = patcher.BatchPatcher(num_train_images, batch_size, num_label)
    test_batch_patcher = patcher.BatchPatcher(num_test_images, batch_size, num_label)

    config = tf.ConfigProto(allow_soft_placement = True)
    config.gpu_options.visible_device_list = str(gpu_id)
    config.gpu_options.allow_growth = True
    graph = tf.Graph()

    with graph.as_default():
        with tf.device('/gpu:' + str(gpu_id)):
            with tf.Session(config = config) as sess:
                train_ids, train_images, train_labels = input_reader.data_read(batch_size, train = True)
                test_ids, test_images, test_labels = input_reader.data_read(batch_size, train = False)

                if model_type == "DenseNet-25-12":
                    model1 = DenseNet(25, 12, image_shape, num_label, scope='network1')
                    model2 = DenseNet(25, 12, image_shape, num_label, scope='network2')
                elif model_type == "DenseNet-40-12":
                    model1 = DenseNet(40, 12, image_shape, num_label, scope='network1')
                    model2 = DenseNet(40, 12, image_shape, num_label, scope='network2')
                elif model_type == "DenseNet-10-12":
                    model1 = DenseNet(10, 12, image_shape, num_label, scope='network1')
                    model2 = DenseNet(10, 12, image_shape, num_label, scope='network2')
                elif model_type == "VGG-19":
                    model1 = VGG19(image_shape, num_label, scope='network1')
                    model2 = VGG19(image_shape, num_label, scope='network2')

                # register training operations on Trainer class
                trainer1 = Trainer(model1)
                trainer1.train_loss_op, trainer1.train_accuracy_op, trainer1.train_op, trainer1.train_xentropy_op, trainer1.train_prob_op = model1.build_train_op(lr_boundaries, lr_values, optimizer_type)
                trainer1.test_loss_op, trainer1.test_accuracy_op, trainer1.test_xentropy_op, trainer1.test_prob_op = model1.build_test_op()

                trainer2 = Trainer(model2)
                trainer2.train_loss_op, trainer2.train_accuracy_op, trainer2.train_op, trainer2.train_xentropy_op, trainer2.train_prob_op = model2.build_train_op(lr_boundaries, lr_values, optimizer_type)
                trainer2.test_loss_op, trainer2.test_accuracy_op, trainer2.test_xentropy_op, trainer2.test_prob_op = model2.build_test_op()

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord = coord)

                # load data set in main memory
                train_batch_patcher.bulk_load_in_memory(sess, train_ids, train_images, train_labels)
                test_batch_patcher.bulk_load_in_memory(sess, test_ids, test_images, test_labels)

                # give noise on data set
                train_batch_patcher.set_noise(noise_rate, noise_type)

                ######################## main methodology for training #######################
                sess.run(tf.global_variables_initializer())

                # for model 1
                for epoch in range(total_epochs):
                    ratio = 1.0 - noise_rate * np.fmin(1.0, float(float(epoch) + 1.0) / float(T_k))
                    # print("Clean sample ratio: ", ratio)
                    for i in range(train_batch_patcher.num_iters_per_epoch):
                        ids, images, labels = train_batch_patcher.get_next_random_mini_batch(batch_size)

                        # losses of model 1
                        xentropy_array1 = sess.run(trainer1.train_xentropy_op, feed_dict={trainer1.model.train_image_placeholder: images, trainer1.model.train_label_placeholder: labels})
                        # losses of model 2
                        xentropy_array2 = sess.run(trainer2.train_xentropy_op, feed_dict={trainer2.model.train_image_placeholder: images, trainer2.model.train_label_placeholder: labels})

                        # choose R(T)% small-loss instances
                        ids_for_2, images_for_2, labels_for_2 = get_lower_loss_instances(batch_size, ids, images, labels, xentropy_array1, ratio)
                        ids_for_1, images_for_1, labels_for_1 = get_lower_loss_instances(batch_size, ids, images, labels, xentropy_array2, ratio)

                        # model update
                        _, _ = sess.run([trainer1.train_op, trainer2.train_op], feed_dict={trainer1.model.train_image_placeholder: images_for_1, trainer1.model.train_label_placeholder: labels_for_1, trainer2.model.train_image_placeholder: images_for_2, trainer2.model.train_label_placeholder: labels_for_2})


                    # inference and test error for logs
                    # test
                    avg_val_loss_1 = 0.0
                    avg_val_acc_1 = 0.0
                    avg_val_loss_2 = 0.0
                    avg_val_acc_2 = 0.0
                    for i in range(test_batch_patcher.num_iters_per_epoch):
                        ids, images, labels = test_batch_patcher.get_init_mini_batch(i)
                        val_loss_1, val_acc_1 = sess.run([trainer1.test_loss_op, trainer1.test_accuracy_op], feed_dict={trainer1.model.test_image_placeholder: images, trainer1.model.test_label_placeholder: labels})
                        val_loss_2, val_acc_2 = sess.run([trainer2.test_loss_op, trainer2.test_accuracy_op], feed_dict={trainer2.model.test_image_placeholder: images, trainer2.model.test_label_placeholder: labels})

                        avg_val_loss_1 += val_loss_1
                        avg_val_acc_1 += val_acc_1
                        avg_val_loss_2 += val_loss_2
                        avg_val_acc_2 += val_acc_2

                    avg_val_loss_1 /= test_batch_patcher.num_iters_per_epoch
                    avg_val_acc_1 /= test_batch_patcher.num_iters_per_epoch
                    avg_val_loss_2 /= test_batch_patcher.num_iters_per_epoch
                    avg_val_acc_2 /= test_batch_patcher.num_iters_per_epoch

                    # Inference
                    avg_train_loss_1 = 0.0
                    avg_train_acc_1 = 0.0
                    avg_train_loss_2 = 0.0
                    avg_train_acc_2 = 0.0

                    for i in range(train_batch_patcher.num_iters_per_epoch):
                        ids, images, labels = train_batch_patcher.get_init_mini_batch(i)
                        imp_loss_1, imp_acc_1 = sess.run([trainer1.train_loss_op, trainer1.train_accuracy_op], feed_dict={trainer1.model.train_image_placeholder: images, trainer1.model.train_label_placeholder: labels})
                        imp_loss_2, imp_acc_2 = sess.run([trainer2.train_loss_op, trainer2.train_accuracy_op], feed_dict={trainer2.model.train_image_placeholder: images, trainer2.model.train_label_placeholder: labels})

                        avg_train_loss_1 += imp_loss_1
                        avg_train_acc_1 += imp_acc_1
                        avg_train_loss_2 += imp_loss_2
                        avg_train_acc_2 += imp_acc_2

                    avg_train_loss_1 /= train_batch_patcher.num_iters_per_epoch
                    avg_train_acc_1 /= train_batch_patcher.num_iters_per_epoch
                    avg_train_loss_2 /= train_batch_patcher.num_iters_per_epoch
                    avg_train_acc_2 /= train_batch_patcher.num_iters_per_epoch


                    cur_lr = sess.run(trainer1.model.learning_rate)
                    print((epoch + 1), ", ", cur_lr, ", ", avg_train_loss_1, ", ", avg_train_acc_1,
                          ", ", avg_train_loss_2, ", ", avg_train_acc_2,
                          ", ", avg_val_loss_1, ", ", avg_val_acc_1,
                          ", ", avg_val_loss_2, ", ", avg_val_acc_2)
                    text_log.append(str(epoch + 1) + ", " + str(cur_lr) + ", " + str(avg_train_loss_1) + ", " + str(1.0 - avg_train_acc_1)
                                    + ", " + str(avg_train_loss_2) + ", " + str(1.0 - avg_train_acc_2)
                                    + ", " + str(avg_val_loss_1) + ", " + str(1.0 - avg_val_acc_1)
                                    + ", " + str(avg_val_loss_2) + ", " + str(1.0 - avg_val_acc_2))
                ##############################################################################

                coord.request_stop()
                coord.join(threads)
                sess.close()

    f = open(log_dir + "/log.csv", "w")
    for text in text_log:
        f.write(text + "\n")
    f.close()


def get_lower_loss_instances(batch_size, ids, images, labels, xentropy_array, ratio):

    num_instance = int(np.ceil(float(batch_size) * ratio))

    loss_map = {}
    image_map = {}
    label_map = {}

    for i in range(len(ids)):
        loss_map[ids[i]] = xentropy_array[i]
        image_map[ids[i]] = images[i]
        label_map[ids[i]] = labels[i]

    loss_map = dict(sorted(loss_map.items(), key=operator.itemgetter(1), reverse=False))

    new_ids = []
    new_images = []
    new_labels = []

    index = 0
    for key in loss_map.keys():
        if index >= num_instance:
            break

        new_ids.append(key)
        new_images.append(image_map[key])
        new_labels.append(label_map[key])
        index += 1

    return new_ids, new_images, new_labels

