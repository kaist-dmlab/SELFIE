import os, sys
import tensorflow as tf

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

def training(sess, training_epochs, batch_size, train_batch_patcher, validation_batch_patcher, trainer, cur_epoch, method, training_log=None):

    for epoch in range(training_epochs):
        avg_train_loss = 0.0
        avg_train_acc = 0.0

        if method == "default":
            for i in range(train_batch_patcher.num_iters_per_epoch):
                ids, images, labels = train_batch_patcher.get_next_random_mini_batch(batch_size)
                train_loss, train_acc, _ = sess.run([trainer.train_loss_op, trainer.train_accuracy_op, trainer.train_op], feed_dict={trainer.model.train_image_placeholder: images, trainer.model.train_label_placeholder: labels})
                avg_train_loss += train_loss
                avg_train_acc += train_acc

        avg_train_loss /= train_batch_patcher.num_iters_per_epoch
        avg_train_acc /= train_batch_patcher.num_iters_per_epoch

        # Validation
        avg_val_loss = 0.0
        avg_val_acc = 0.0
        for i in range(validation_batch_patcher.num_iters_per_epoch):
            ids, images, labels = validation_batch_patcher.get_init_mini_batch(i)
            val_loss, val_acc = sess.run([trainer.test_loss_op, trainer.test_accuracy_op], feed_dict={trainer.model.test_image_placeholder: images, trainer.model.test_label_placeholder: labels})
            avg_val_loss += val_loss
            avg_val_acc += val_acc
        avg_val_loss /= validation_batch_patcher.num_iters_per_epoch
        avg_val_acc /= validation_batch_patcher.num_iters_per_epoch

        # training log
        cur_lr = sess.run(trainer.model.learning_rate)
        print((epoch + cur_epoch + 1), ", ", cur_lr, ", ", avg_train_loss,  ", ", avg_train_acc, ", ", avg_val_loss, ", ", avg_val_acc)
        if training_log is not None:
            training_log.append(str(epoch + cur_epoch + 1) + ", " + str(cur_lr) + ", " + str(avg_train_loss) + ", " + str(1.0 - avg_train_acc) + ", " + str(avg_val_loss) + ", " + str(1.0 - avg_val_acc))

def default(gpu_id, input_reader, model_type, total_epochs, batch_size, lr_boundaries, lr_values, optimizer_type, noise_rate, noise_type, log_dir="log"):

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    method = "default"

    # log list
    training_log = []
    training_log.append("epcoh, learning rate, training loss, training error, test loss, test error\n")

    num_train_images = input_reader.num_train_images
    num_test_images = input_reader.num_val_images
    num_label = input_reader.num_classes
    image_shape = [input_reader.height, input_reader.width, input_reader.depth]

    # batch pathcer
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
                    model = DenseNet(25, 12, image_shape, num_label)
                elif model_type == "DenseNet-40-12":
                    model = DenseNet(40, 12, image_shape, num_label)
                elif model_type == "DenseNet-10-12":
                    model = DenseNet(10, 12, image_shape, num_label)
                elif model_type == "VGG-19":
                    model = VGG19(image_shape, num_label)

                # register training operations on Trainer class
                trainer = Trainer(model)
                trainer.train_loss_op, trainer.train_accuracy_op, trainer.train_op, trainer.train_xentropy_op, trainer.train_prob_op = model.build_train_op(lr_boundaries, lr_values, optimizer_type)
                trainer.test_loss_op, trainer.test_accuracy_op, trainer.test_xentropy_op, trainer.test_prob_op = model.build_test_op()
                trainer.init_op = tf.global_variables_initializer()

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord = coord)

                # load data set in main memory
                train_batch_patcher.bulk_load_in_memory(sess, train_ids, train_images, train_labels)
                test_batch_patcher.bulk_load_in_memory(sess, test_ids, test_images, test_labels)

                # give noise on data set
                train_batch_patcher.set_noise(noise_rate, noise_type)

                ######################## main methodology for training #######################
                sess.run(trainer.init_op)

                train_batch_patcher.print_transition_matrix(train_batch_patcher.get_current_noise_matrix(entire=True))
                training(sess, total_epochs, batch_size, train_batch_patcher, test_batch_patcher, trainer, 0, method, training_log=training_log)
                ##############################################################################

                coord.request_stop()
                coord.join(threads)
                sess.close()

    f = open(log_dir + "/log.csv", "w")
    for text in training_log:
        f.write(text + "\n")
    f.close()
