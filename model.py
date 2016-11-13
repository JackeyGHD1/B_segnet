# encoding: utf-8

# general
import os
import re
import sys

# tensorflow
import tensorflow as tf
from tensorflow.python.training import moving_averages

import numpy as np
import math

# settings
import settings
FLAGS = settings.FLAGS

NUM_CLASSES = FLAGS.num_classes
IMAGE_HEIGHT = FLAGS.image_height
IMAGE_WIDTH = FLAGS.image_width
LEARNING_RATE_DECAY_FACTOR = FLAGS.learning_rate_decay_factor
INITIAL_LEARNING_RATE = FLAGS.learning_rate

# multiple GPU's prefix
TOWER_NAME = FLAGS.tower_name

# Used to keep the update ops done by batch_norm.
UPDATE_OPS_COLLECTION = '_update_ops_'

def debug(images):
    return images

def _variable_with_weight_decay(name, shape, stddev, wd, trainable=True):
    '''
    重み減衰を利用した変数の初期化
    '''
    var = _variable_on_gpu(name, shape, tf.truncated_normal_initializer(stddev=stddev), trainable=trainable)
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _variable_on_cpu(name, shape, initializer, trainable=True):
    '''
    CPUメモリに変数をストアする
    '''
    #with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer, trainable=trainable)
    return var

def _variable_on_gpu(name, shape, initializer, trainable=True):
    '''
    GPUメモリに変数をストアする
    '''
    #with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer, trainable=trainable)
    return var

def _activation_summary(x):
    '''
    可視化用のサマリを作成
    '''
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def conv2d(scope_name, inputs, shape, bias_shape, stride, padding='VALID', wd=0.0, reuse=False, trainable=True):
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        kernel = _variable_with_weight_decay(
            'weights',
            shape=shape,
            stddev=0.01,
            wd=0.0,  # not use weight decay
            trainable=trainable
        )
        conv = tf.nn.conv2d(inputs, kernel, stride, padding=padding)
        #biases = _variable_on_gpu('biases', bias_shape, tf.constant_initializer(0.1), trainable=trainable)
        #bias = tf.nn.bias_add(conv, biases)
        bn = batch_norm(conv)
        conv_ = tf.nn.relu(bn, name=scope.name)
        return conv_


def conv2d_transpose(scope_name, inputs, shape, output_shape, bias_shape, stride, padding='VALID', reuse=False, trainable=True):
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        kernel = _variable_with_weight_decay(
            'weights',
            shape=shape,
            stddev=0.01,
            wd=0.0,  # not use weight decay
            trainable=trainable
        )
        tf_output_shape = tf.pack(output_shape)
        deconv = tf.nn.conv2d_transpose(inputs, kernel, tf_output_shape, stride, padding=padding)
        deconv.set_shape(output_shape)
        bn = batch_norm(deconv)
        #biases = _variable_on_gpu('biases', bias_shape, tf.constant_initializer(0.1), trainable=trainable)
        #bias = tf.nn.bias_add(deconv, biases)
        deconv_ = tf.nn.relu(bn, name=scope.name)
        return deconv_


def batch_norm(inputs,
                       decay=0.999,
                       center=True,
                       scale=False,
                       epsilon=0.001,
                       moving_vars='moving_vars',
                       activation=None,
                       is_training=True,
                       trainable=True,
                       restore=True,
                       scope=None,
                       reuse=None):
    """Adds a Batch Normalization layer.
    Args:
      inputs: a tensor of size [batch_size, height, width, channels]
              or [batch_size, channels].
      decay: decay for the moving average.
      center: If True, subtract beta. If False, beta is not created and ignored.
      scale: If True, multiply by gamma. If False, gamma is
        not used. When the next layer is linear (also e.g. ReLU), this can be
        disabled since the scaling can be done by the next layer.
      epsilon: small float added to variance to avoid dividing by zero.
      moving_vars: collection to store the moving_mean and moving_variance.
      activation: activation function.
      is_training: whether or not the model is in training mode.
      trainable: whether or not the variables should be trainable or not.
      restore: whether or not the variables should be marked for restore.
      scope: Optional scope for variable_op_scope.
      reuse: whether or not the layer and its variables should be reused. To be
        able to reuse the layer scope must be given.
    Returns:
      a tensor representing the output of the operation.
    """
    inputs_shape = inputs.get_shape()
    with tf.variable_op_scope([inputs], scope, 'BatchNorm', reuse=reuse):
        axis = list(range(len(inputs_shape) - 1))
        params_shape = inputs_shape[-1:]
        # Allocate parameters for the beta and gamma of the normalization.
        beta, gamma = None, None
        if center:
            beta = tf.get_variable('beta',
                                      params_shape,
                                      initializer=tf.zeros_initializer,
                                      trainable=trainable)
        if scale:
            gamma = tf.get_variable('gamma',
                                       params_shape,
                                       initializer=tf.ones_initializer,
                                       trainable=trainable)
        # 移動平均と移動分散を作成する(明示的にリストアが必要)
        # Create moving_mean and moving_variance add them to
        # GraphKeys.MOVING_AVERAGE_VARIABLES collections. (restoreに使う)
        moving_collections = [moving_vars, tf.GraphKeys.MOVING_AVERAGE_VARIABLES]
        moving_mean = tf.get_variable('moving_mean',
                                         params_shape,
                                         initializer=tf.zeros_initializer,
                                         trainable=False)
        moving_variance = tf.get_variable('moving_variance',
                                             params_shape,
                                             initializer=tf.ones_initializer,
                                             trainable=False)

        if is_training:
            # Calculate the moments based on the individual batch.
            mean, variance = tf.nn.moments(inputs, axis)

            update_moving_mean = moving_averages.assign_moving_average(
                moving_mean, mean, decay)
            tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
            update_moving_variance = moving_averages.assign_moving_average(
                moving_variance, variance, decay)
            tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)
        else:
            # Just use the moving_mean and moving_variance.
            mean = moving_mean
            variance = moving_variance
        # Normalize the activations.
        outputs = tf.nn.batch_normalization(
            inputs, mean, variance, beta, gamma, epsilon)
        outputs.set_shape(inputs.get_shape())
        if activation:
            outputs = activation(outputs)
        return outputs


def print_shape(name, layer):
    print "="*100
    print name
    print layer.get_shape()
    print "="*100

def test_conv(images):
    en_conv1_1 = conv2d('en_conv1_1', images, [3, 3, 3, 21], [21], [1, 1, 1, 1], padding='SAME')

    return en_conv1_1

def inference_segnet_former(images, keep_conv, keep_hidden, reuse=False):
    '''
    SegNet encoder bases VGG16
    :param images:
    :param keep_conv:
    :param keep_hidden:
    :param reuse:
    :return: pool layer's feature map
    '''
    en_conv1_1 = conv2d('en_conv1_1', images, [3, 3, 3, 64], [64], [1, 1, 1, 1], padding='SAME', reuse=reuse)
    en_conv1_2 = conv2d('en_conv1_2', en_conv1_1, [3, 3, 64, 64], [64], [1, 1, 1, 1], padding='SAME', reuse=reuse)
    en_pool1 = tf.nn.max_pool(en_conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    en_conv2_1 = conv2d('en_conv2_1', en_pool1, [3, 3, 64, 128], [128], [1, 1, 1, 1], padding='SAME', reuse=reuse)
    en_conv2_2 = conv2d('en_conv2_2', en_conv2_1, [3, 3, 128, 128], [128], [1, 1, 1, 1], padding='SAME', reuse=reuse)
    en_pool2 = tf.nn.max_pool(en_conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')


    return (en_pool1, en_pool2)

def inference_segnet_latter(images, feature_maps, class_num, keep_conv, keep_hidden, batch_size, reuse=False):
    '''
    SegNet decoder bases FCN
    :param feature_maps:
    :param keep_conv:
    :param keep_hidden:
    :param reuse:
    :return:
    '''
    en_pool1 = feature_maps[0]
    en_pool2 = feature_maps[1]

    # resolution check
    print("resolution check")
    #print_shape("images", images)
    print_shape("en_pool1", en_pool1)
    print_shape("en_pool2", en_pool2)


    # SegNet original size
    # == == == == == == == == == == ==
    # images
    # (5, 360, 480, 3)
    # en_pool5
    # (5, 11, 15, 512)
    # en_pool4
    # (5, 23, 30, 512)
    # en_pool3
    # (5, 45, 60, 256)
    # en_pool2
    #
    # en_pool1
    # (5, 180, 240, 64)
    # == == == == == == == == == == ==

    # dementionaly reduction
    # dim_reduct_feature5 = conv2d('dim_reduct_feature5', en_pool5, [1, 1, 512, class_num], [class_num], [1, 1, 1, 1])
    # dim_reduct_feature1 = conv2d('dim_reduct_feature1', en_pool1, [1, 1, 64, class_num], [class_num], [1, 1, 1, 1], reuse=reuse)

    en_conv3_1 = conv2d('en_conv3_1', en_pool2, [3, 3, 128, 256], [256], [1, 1, 1, 1], padding='SAME', reuse=reuse)
    en_conv3_2 = conv2d('en_conv3_2', en_conv3_1, [3, 3, 256, 256], [256], [1, 1, 1, 1], padding='SAME', reuse=reuse)
    en_conv3_3 = conv2d('en_conv3_3', en_conv3_2, [3, 3, 256, 256], [256], [1, 1, 1, 1], padding='SAME', reuse=reuse)
    en_pool3 = tf.nn.max_pool(en_conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
    en_pool3_dropout = tf.nn.dropout(en_pool3, keep_conv)

    en_conv4_1 = conv2d('en_conv4_1', en_pool3_dropout, [3, 3, 256, 512], [512], [1, 1, 1, 1], padding='SAME', reuse=reuse)
    en_conv4_2 = conv2d('en_conv4_2', en_conv4_1, [3, 3, 512, 512], [512], [1, 1, 1, 1], padding='SAME', reuse=reuse)
    en_conv4_3 = conv2d('en_conv4_3', en_conv4_2, [3, 3, 512, 512], [512], [1, 1, 1, 1], padding='SAME', reuse=reuse)
    en_pool4 = tf.nn.max_pool(en_conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
    en_pool4_dropout = tf.nn.dropout(en_pool4, keep_conv)

    en_conv5_1 = conv2d('en_conv5_1', en_pool4_dropout, [3, 3, 512, 512], [512], [1, 1, 1, 1], padding='SAME', reuse=reuse)
    en_conv5_2 = conv2d('en_conv5_2', en_conv5_1, [3, 3, 512, 512], [512], [1, 1, 1, 1], padding='SAME', reuse=reuse)
    en_conv5_3 = conv2d('en_conv5_3', en_conv5_2, [3, 3, 512, 512], [512], [1, 1, 1, 1], padding='SAME', reuse=reuse)
    en_pool5 = tf.nn.max_pool(en_conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')
    en_pool5_dropout = tf.nn.dropout(en_pool5, keep_conv)

    de_deconv1 = conv2d_transpose('de_deconv1', en_pool5_dropout, [3, 3, 256, 512], [batch_size, 23, 30, 256], [256],
                                  [1, 2, 2, 1], padding='SAME', reuse=reuse)
    print_shape("de_deconv1", de_deconv1)
    de_deconv1_feature = tf.concat(3, [de_deconv1, en_pool4])
    de_conv1_1 = conv2d('de_conv1_1', de_deconv1_feature, [3, 3, 768, 512], [512], [1, 1, 1, 1], padding='SAME',
                        reuse=reuse)
    de_conv1_2 = conv2d('de_conv1_2', de_conv1_1, [3, 3, 512, 512], [512], [1, 1, 1, 1], padding='SAME', reuse=reuse)
    de_conv1_2_dropout = tf.nn.dropout(de_conv1_2, keep_conv)

    de_deconv2 = conv2d_transpose('de_deconv2', de_conv1_2_dropout, [3, 3, 256, 512], [batch_size, 45, 60, 256], [256],
                                  [1, 2, 2, 1], padding='SAME', reuse=reuse)
    de_deconv2_feature = tf.concat(3, [de_deconv2, en_pool3])
    de_conv2_1 = conv2d('de_conv2_1', de_deconv2_feature, [3, 3, 512, 256], [256], [1, 1, 1, 1], padding='SAME',
                        reuse=reuse)
    de_conv2_2 = conv2d('de_conv2_2', de_conv2_1, [3, 3, 256, 256], [256], [1, 1, 1, 1], padding='SAME', reuse=reuse)
    de_conv2_2_dropout = tf.nn.dropout(de_conv2_2, keep_conv)

    de_deconv2 = conv2d_transpose('de_deconv3', de_conv2_2_dropout, [3, 3, 128, 256], [batch_size, 90, 120, 128], [128],
                                  [1, 2, 2, 1], padding='SAME', reuse=reuse)
    de_deconv3_feature = tf.concat(3, [de_deconv2, en_pool2])
    de_conv3_1 = conv2d('de_conv3_1', de_deconv3_feature, [3, 3, 256, 128], [128], [1, 1, 1, 1], padding='SAME', reuse=reuse)
    de_conv3_2 = conv2d('de_conv3_2', de_conv3_1, [3, 3, 128, 128], [128], [1, 1, 1, 1], padding='SAME', reuse=reuse)
    de_conv3_2_dropout = tf.nn.dropout(de_conv3_2, keep_conv)

    de_deconv4 = conv2d_transpose('de_deconv4', de_conv3_2_dropout, [3, 3, 64, 128], [batch_size, 180, 240, 64], [64],
                                  [1, 2, 2, 1], padding='SAME', reuse=reuse)
    de_deconv4_feature = tf.concat(3, [de_deconv4, en_pool1])
    de_conv4_1 = conv2d('de_conv4_1', de_deconv4_feature, [3, 3, 128, 64], [64], [1, 1, 1, 1], padding='SAME', reuse=reuse)
    de_conv4_2 = conv2d('de_conv4_2', de_conv4_1, [3, 3, 64, 64], [64], [1, 1, 1, 1], padding='SAME', reuse=reuse)

    de_deconv5 = conv2d_transpose('de_deconv5', de_conv4_2, [3, 3, 64, 64], [batch_size, 360, 480, 64], [64],
                                  [1, 2, 2, 1], padding='SAME', reuse=reuse)
    de_conv5_1 = conv2d('de_conv5_1', de_deconv5, [3, 3, 64, 64], [64], [1, 1, 1, 1], padding='SAME', reuse=reuse)
    de_conv5_2 = conv2d('de_conv5_2', de_conv5_1, [3, 3, 64, class_num], [class_num], [1, 1, 1, 1], padding='SAME', reuse=reuse)

    print_shape("de_conv5_2", de_conv5_2)

    return de_conv5_2, pixelwise_argmax(de_conv5_2)


def pixelwise_softmax(target, name=None):
    with tf.op_scope([target], name, 'softmax'):
        max_axis = tf.reduce_max(target, reduction_indices=[3], keep_dims=True)
        target_exp = tf.exp(target - max_axis)
        normalize = tf.reduce_sum(target_exp, [3], keep_dims=True)
        softmax = target_exp / normalize
    return softmax


def pixelwise_argmax(target, name=None):
    with tf.op_scope([target], name, 'argmax'):
        max_axis = tf.reduce_max(target, reduction_indices=[3], keep_dims=True)
        target_exp = tf.exp(target - max_axis)
        normalize = tf.reduce_sum(target_exp, [3], keep_dims=True)
        softmax = target_exp / normalize
    return tf.argmax(softmax, dimension=3)


def samples_mean(samples_list):
    samples_tensor = tf.pack([x for x in samples_list])
    return tf.reduce_mean(samples_tensor, reduction_indices=[0])


def samples_var(samples_list):
    mean_output_vec = samples_mean(samples_list)
    sum = tf.zeros([IMAGE_HEIGHT, IMAGE_WIDTH, FLAGS.num_classes])
    for sample in samples_list:
        sum = tf.add(sum, tf.square(tf.sub(sample, mean_output_vec)))
    var = tf.div(sum, tf.fill([IMAGE_HEIGHT, IMAGE_WIDTH, FLAGS.num_classes], float(FLAGS.num_classes)))
    return mean_output_vec, tf.reduce_mean(var, reduction_indices=[3])

def loss_segnet(logits, targets):
    print "=" * 100
    print "loss logits:"
    print logits.get_shape()
    print "loss depths"
    print targets.get_shape()
    print "=" * 100

    logits_reshape = tf.reshape(logits, [-1, FLAGS.num_classes])
    targets_reshape = tf.reshape(targets, [-1])


    cost = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(tf.cast(logits_reshape, tf.float32),
                                                                               tf.cast(targets_reshape,
                                                                                       tf.int32))) / FLAGS.image_height * FLAGS.image_width

    tf.add_to_collection('losses', cost)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.scalar_summary(l.op.name + ' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))

    return loss_averages_op
