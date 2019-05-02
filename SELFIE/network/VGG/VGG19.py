import sys, os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from network.VGG.utils import *
import tensorflow as tf

weight_decay = 0.0005

class VGG19(object):
    def __init__(self, image_shape, num_labels, scope="VGG-19"):

        self.image_shape = image_shape
        self.num_labels = num_labels
        self.scope = scope

        [height, width, channels] = image_shape


        train_batch_shape = [None, height, width, channels]
        self.train_image_placeholder = tf.placeholder(
            tf.float32,
            shape=train_batch_shape,
            name='train_images'
        )
        self.train_label_placeholder = tf.placeholder(
            tf.int32,
            shape=[None, ],
            name='train_labels'
        )

        test_batch_shape = [None, height, width, channels]
        self.test_image_placeholder = tf.placeholder(
            tf.float32,
            shape=test_batch_shape,
            name='test_images'
        )
        self.test_label_placeholder = tf.placeholder(
            tf.int32,
            shape=[None, ],
            name='test_labels'
        )

    def build_network(self, images, is_training, reuse):
        with tf.variable_scope(self.scope, reuse=reuse):

            batch_size = images.get_shape().as_list()[0]
            if is_training:
                keep_prob = 0.5
            else:
                keep_prob = 1.0

            #########################
            #conv_1 = conv2d(scope='conv_1', input_layer=images, output_dim=64, use_bias=True, filter_size=3, strides=[1, 1, 1, 1])
            conv_1 = conv('conv_1', images, 64)
            conv_1 = batch_norm('conv_1_bn', conv_1, is_training, reuse)
            conv_1 = lrelu(conv_1)
            #conv_2 = conv2d(scope='conv_2', input_layer=conv_1, output_dim=64, use_bias=True, filter_size=3, strides=[1, 1, 1, 1])
            conv_2 = conv('conv_2', conv_1, 64)
            conv_2 = batch_norm('conv_2_bn', conv_2, is_training, reuse)
            conv_2 = lrelu(conv_2)
            conv_2 = tf.nn.max_pool(conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

            #########################
            #conv_3 = conv2d(scope='conv_3', input_layer=conv_2, output_dim=128, use_bias=True, filter_size=3, strides=[1, 1, 1, 1])
            conv_3 = conv('conv_3', conv_2, 128)
            conv_3 = batch_norm('conv_3_bn', conv_3, is_training, reuse)
            conv_3 = lrelu(conv_3)
            #conv_4 = conv2d(scope='conv_4', input_layer=conv_3, output_dim=128, use_bias=True, filter_size=3, strides=[1, 1, 1, 1])
            conv_4 = conv('conv_4', conv_3, 128)
            conv_4 = batch_norm('conv_4_bn', conv_4, is_training, reuse)
            conv_4 = lrelu(conv_4)
            conv_4 = tf.nn.max_pool(conv_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

            #########################
            #conv_5 = conv2d(scope='conv_5', input_layer=conv_4, output_dim=256, use_bias=True, filter_size=3, strides=[1, 1, 1, 1])
            conv_5 = conv('conv_5', conv_4, 256)
            conv_5 = batch_norm('conv_5_bn', conv_5, is_training, reuse)
            conv_5 = lrelu(conv_5)
            #conv_6 = conv2d(scope='conv_6', input_layer=conv_5, output_dim=256, use_bias=True, filter_size=3, strides=[1, 1, 1, 1])
            conv_6 = conv('conv_6', conv_5, 256)
            conv_6 = batch_norm('conv_6_bn', conv_6, is_training, reuse)
            conv_6 = lrelu(conv_6)
            #conv_7 = conv2d(scope='conv_7', input_layer=conv_6, output_dim=256, use_bias=True, filter_size=3, strides=[1, 1, 1, 1])
            conv_7 = conv('conv_7', conv_6, 256)
            conv_7 = batch_norm('conv_7_bn', conv_7, is_training, reuse)
            conv_7 = lrelu(conv_7)
            #conv_8 = conv2d(scope='conv_8', input_layer=conv_7, output_dim=256, use_bias=True, filter_size=3, strides=[1, 1, 1, 1])
            conv_8 = conv('conv_8', conv_7, 256)
            conv_8 = batch_norm('conv_8_bn', conv_8, is_training, reuse)
            conv_8 = lrelu(conv_8)
            conv_8 = tf.nn.max_pool(conv_8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

            #########################
            #conv_9 = conv2d(scope='conv_9', input_layer=conv_8, output_dim=512, use_bias=True, filter_size=3, strides=[1, 1, 1, 1])
            conv_9 = conv('conv_9', conv_8, 512)
            conv_9 = batch_norm('conv_9_bn', conv_9, is_training, reuse)
            conv_9 = lrelu(conv_9)
            #conv_10 = conv2d(scope='conv_10', input_layer=conv_9, output_dim=512, use_bias=True, filter_size=3, strides=[1, 1, 1, 1])
            conv_10 = conv('conv_10', conv_9, 512)
            conv_10 = batch_norm('conv_10_bn', conv_10, is_training, reuse)
            conv_10 = lrelu(conv_10)
            #conv_11 = conv2d(scope='conv_11', input_layer=conv_10, output_dim=512, use_bias=True, filter_size=3, strides=[1, 1, 1, 1])
            conv_11 = conv('conv_11', conv_10, 512)
            conv_11 = batch_norm('conv_11_bn', conv_11, is_training, reuse)
            conv_11 = lrelu(conv_11)
            #conv_12 = conv2d(scope='conv_12', input_layer=conv_11, output_dim=512, use_bias=True, filter_size=3, strides=[1, 1, 1, 1])
            conv_12 = conv('conv_12', conv_11, 512)
            conv_12 = batch_norm('conv_12_bn', conv_12, is_training, reuse)
            conv_12 = lrelu(conv_12)
            conv_12 = tf.nn.max_pool(conv_12, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

            #########################
            #conv_13 = conv2d(scope='conv_13', input_layer=conv_12, output_dim=512, use_bias=True, filter_size=3, strides=[1, 1, 1, 1])
            conv_13 = conv('conv_13', conv_12, 512)
            conv_13 = batch_norm('conv_13_bn', conv_13, is_training, reuse)
            conv_13 = lrelu(conv_13)
            #conv_14 = conv2d(scope='conv_14', input_layer=conv_13, output_dim=512, use_bias=True, filter_size=3, strides=[1, 1, 1, 1])
            conv_14 = conv('conv_14', conv_13, 512)
            conv_14 = batch_norm('conv_14_bn', conv_14, is_training, reuse)
            conv_14 = lrelu(conv_14)
            #conv_15 = conv2d(scope='conv_15', input_layer=conv_14, output_dim=512, use_bias=True, filter_size=3, strides=[1, 1, 1, 1])
            conv_15 = conv('conv_15', conv_14, 512)
            conv_15 = batch_norm('conv_15_bn', conv_15, is_training, reuse)
            conv_15 = lrelu(conv_15)
            #conv_16 = conv2d(scope='conv_16', input_layer=conv_15, output_dim=512, use_bias=True, filter_size=3, strides=[1, 1, 1, 1])
            conv_16 = conv('conv_16', conv_15, 512)
            conv_16 = batch_norm('conv_16_bn', conv_16, is_training, reuse)
            conv_16 = lrelu(conv_16)
            conv_16 = tf.nn.max_pool(conv_16, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

            #########################

            fc_17 = fully_connected('fc_17', conv_16, 4096)
            fc_17 = batch_norm('fc_17_bn', fc_17, is_training, reuse)
            fc_17 = lrelu(fc_17)
            fc_17 = tf.nn.dropout(fc_17, keep_prob)

            fc_18 = fully_connected('fc_18', fc_17, 4096)
            fc_18 = batch_norm('fc_18_bn', fc_18, is_training, reuse)
            fc_18 = lrelu(fc_18)
            fc_18 = tf.nn.dropout(fc_18, keep_prob)

            fc_19 = fully_connected('fc_19', fc_18, self.num_labels)

            return tf.nn.softmax(fc_19), fc_19

    def build_train_op(self, lr_boundaries, lr_values, optimizer_type):
        train_step = tf.Variable(initial_value=0, trainable=False)

        self.train_step = train_step

        prob, logits = self.build_network(self.train_image_placeholder, True, False)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.train_label_placeholder,
            logits=logits
        )

        prediction = tf.equal(tf.cast(tf.argmax(prob, axis=1), tf.int32), self.train_label_placeholder)
        prediction = tf.cast(prediction, tf.float32)

        l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

        self.train_loss = tf.reduce_mean(loss) + l2_loss * weight_decay
        self.train_accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
        self.learning_rate = tf.train.piecewise_constant(train_step, lr_boundaries, lr_values)

        if optimizer_type == "momentum":
            optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9, use_nesterov=True)
        elif optimizer_type == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)


        train_vars = [x for x in tf.trainable_variables() if self.scope in x.name]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(self.train_loss, global_step=train_step, var_list=train_vars)

        return self.train_loss, self.train_accuracy, train_op, loss, prob

    def build_test_op(self):
        prob, logits = self.build_network(self.test_image_placeholder, False, True)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.test_label_placeholder,
            logits=logits
        )

        prediction = tf.equal(tf.cast(tf.argmax(prob, axis=1), tf.int32), self.test_label_placeholder)
        prediction = tf.cast(prediction, tf.float32)

        self.test_loss = tf.reduce_mean(loss)
        self.test_accuracy = tf.reduce_mean(prediction)

        return self.test_loss, self.test_accuracy, loss, prob

