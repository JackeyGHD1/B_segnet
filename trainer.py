#encoding: utf-8

from datetime import datetime
import os.path
import time

import tensorflow.python.platform
from tensorflow.python.platform import gfile

from PIL import Image

import numpy as np
import tensorflow as tf

# model
import model

# train operation
import train_operation as op

# inputs
import dataset
from dataset import DataSet

# settings
import settings
FLAGS = settings.FLAGS

TRAIN_DIR = FLAGS.train_dir
EX_DIR = FLAGS.experiment_dir
LOG_DIR = FLAGS.log_dir

MAX_STEPS = FLAGS.max_steps
LOG_DEVICE_PLACEMENT = FLAGS.log_device_placement
TF_RECORDS = FLAGS.train_tfrecords
TF_RECORDS_VAL = FLAGS.eval_tfrecords
BATCH_SIZE = FLAGS.batch_size

def test():
    with tf.Graph().as_default():
        image_input = DataSet()
        csv_train = FLAGS.train_csv
        csv_test = FLAGS.eval_csv
        #images, depths = image_input.csv_inputs(csv_train, 2)
        images, depths = image_input.csv_inputs_augumentation(csv_train, 2)

        keep_conv  = 0.8
        keep_hidden = 0.5

        # inference
        # logits_encoder = model.inference_segnet_encoder(images, keep_conv, keep_hidden)
        # logits_decoder = model.inference_segnet_fcn_decoder(images, logits_encoder, 10, keep_conv, keep_hidden, 5)


        # 初期化オペレーション
        init_op = tf.initialize_all_variables()

        # Session
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=LOG_DEVICE_PLACEMENT))
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        images_val, depths_val = sess.run([images, depths])

        row = 5
        col = 5
        ones = np.ones((2, row, col, 3))
        for i in xrange(row):
            for j in xrange(col):
                ones.itemset((0, i, j, 0), 0)
                ones.itemset((0, i, j, 1), 0)
                ones.itemset((0, i, j, 2), 255)
        for i in xrange(row):
            for j in xrange(col):
                ones.itemset((1, i, j, 0), 255)
                ones.itemset((1, i, j, 1), 0)
                ones.itemset((1, i, j, 2), 0)
        zero_images = tf.convert_to_tensor(ones)
        print ones

        target = np.zeros((2, 5, 5, 1))
        target_images = tf.convert_to_tensor(target)
        #target[0] = 0
        print target

        zeros_val = np.zeros_like(images_val)


        print "zero images shape"
        print zero_images.get_shape
        print "zeros val shape"
        print zeros_val.shape

        image_input.output_images(sess.run(zero_images), "test")
        image_input.output_depths(sess.run(target_images), "test")

        loss = model.loss_segnet(images, depths)


        print "get shape"
        print zero_images.get_shape()

        init_op = tf.initialize_all_variables()
        sess.run(init_op)

        images_re = tf.reshape(ones, [-1, 3])

        print "get shape flat"
        print images_re.get_shape()

        target_re = tf.reshape(target, [-1])

        image_re_val = sess.run(images_re)
        print image_re_val.shape



        sparse_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(tf.cast(images_re, tf.float32), tf.cast(target_re, tf.int32)))/FLAGS.image_height*FLAGS.image_width
        sparse_loss_val = sess.run(sparse_loss)

        print sparse_loss_val

        # session
        coord.request_stop()
        coord.join(threads)
        sess.close()


def train():
    '''
    Train
    '''
    with tf.Graph().as_default():
        # globalなstep数
        global_step = tf.Variable(0, trainable=False)

        csv_train = FLAGS.train_csv
        csv_test = FLAGS.eval_csv

        image_input = DataSet()
        #images, targets = image_input.csv_inputs(csv_train, FLAGS.batch_size)
        #images_val, targets_val = image_input.csv_inputs(csv_test, FLAGS.num_examples)
        images, targets = image_input.csv_inputs_augumentation(csv_train, FLAGS.batch_size)
        images_val, targets_val = image_input.csv_inputs_augumentation(csv_test, FLAGS.num_examples)
        images_val_debug = model.debug(images_val)
        targets_val_debug = model.debug(targets_val)

        keep_conv = tf.placeholder(tf.float32)
        keep_hidden = tf.placeholder(tf.float32)

        # graphのoutput
        print("train.")
        encoder_output = model.inference_segnet_former(images, keep_conv, keep_hidden)
        encoder_output_val = model.inference_segnet_former(images_val, keep_conv, keep_hidden, reuse=True)
        logits, logits_argmax = model.inference_segnet_latter(images, encoder_output, FLAGS.num_classes, keep_conv, keep_hidden, batch_size=FLAGS.batch_size)
        logits_val, logits_argmax_val = model.inference_segnet_latter(images_val, encoder_output_val, FLAGS.num_classes, keep_conv, keep_hidden, batch_size=FLAGS.num_examples, reuse=True)


        # loss graphのoutputとlabelを利用
        loss = model.loss_segnet(logits, targets)
        loss_val = model.loss_segnet(logits_val, targets_val)
        tf.scalar_summary("validation", loss_val)
        # 学習オペレーション
        train_op = op.train(loss, global_step)

        # サマリー
        summary_op = tf.merge_all_summaries()


        # 初期化オペレーション
        init_op = tf.initialize_all_variables()

        # Session
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=LOG_DEVICE_PLACEMENT))

        summary_writer = tf.train.SummaryWriter(LOG_DIR, graph_def=sess.graph_def)

        # saver
        #saver = tf.train.Saver(tf.all_variables())

        sess.run(init_op)

        segnet_params = {}
        #if FLAGS.refine_train:
        #    for variable in tf.all_variables():
        #        variable_name = variable.name
        #        print("parameter: %s" % (variable_name))
        #        if variable_name.find("/") < 0 or variable_name.count("/") != 1:
        #            print("ignore.")
        #            continue
        #        if variable_name.find('coarse') >= 0:
        #            print("coarse parameter: %s" % (variable_name))
        #            coarse_params[variable_name] = variable
        #        print("parameter: %s" %(variable_name))
        #        if variable_name.find('fine') >= 0:
        #            print("refine parameter: %s" % (variable_name))
        #            refine_params[variable_name] = variable
        #else:
        print("Create variable list for saver.")
        for variable in tf.trainable_variables():
            variable_name = variable.name
            if variable_name.find("/") < 0 or variable_name.count("/") > 2:
                print("ignore parameter: %s" % (variable_name))
                continue
            if variable_name.find('en_conv') >= 0:
                print("en_conv parameter: %s" % (variable_name))
                segnet_params[variable_name] = variable
            if variable_name.find('de_conv') >= 0:
                print("de_conv parameter: %s" % (variable_name))
                segnet_params[variable_name] = variable

        print("=="*100)

        #for variable in tf.get_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES):
        for variable in tf.all_variables():
            variable_name = variable.name
            #print("MOVING_AVERAGE_VARIABLES collection: %s" % (variable_name))
            if variable_name.find("moving_mean") >= 0:
                print("moving mean parameter: %s" % (variable_name))
                segnet_params[variable_name] = variable
            elif variable_name.find("moving_variance") >= 0:
                print("moving variance parameter: %s" % (variable_name))
                segnet_params[variable_name] = variable

        print("=="*100)

        # define saver
        #saver = tf.train.Saver(segnet_params)
        saver = tf.train.Saver(tf.all_variables())

        # fine tune
        if FLAGS.fine_tune:
            # load coarse paramteters
            segnet_ckpt = tf.train.get_checkpoint_state(TRAIN_DIR)
            if segnet_ckpt and segnet_ckpt.model_checkpoint_path:
                print("Pretrained segnet Model Loading.")
                print("model path: %s" % (segnet_ckpt.model_checkpoint_path))
                saver.restore(sess, segnet_ckpt.model_checkpoint_path)
                print("Pretrained segnet Model Restored.")
            else:
                print("No Pretrained segnet Model.")
            
        # TODO train coarse or refine (change trainable)
        #if not FLAGS.coarse_train:
        #    for val in coarse_params:
        #        print val
        #if not FLAGS.refine_train:
        #    for val in coarse_params:
        #        print val

        # train refine
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        dropout_flag = False
        # max_stepまで繰り返し学習
        avg_loss_list = []
        previous_loss = 10000000
        saturate_step = 0
        for step in xrange(MAX_STEPS):
            start_time = time.time()
            previous_time = start_time
            index = 0
            for i in xrange(100):
                loss_list = []
                if dropout_flag:
                    _, loss_value = sess.run([train_op, loss], feed_dict={keep_conv: 0.5, keep_hidden: 0.5})
                else:
                    _, loss_value = sess.run([train_op, loss], feed_dict={keep_conv: 1.0, keep_hidden: 1.0})
                #_, loss_value = sess.run([train_op, loss], feed_dict={keep_conv: 0.5, keep_hidden: 0.5})

                if i == 0:
                    if dropout_flag:
                        print("------------------- using Dropout ---------------------")
                        print("saturate step is %d" % saturate_step)

                if index % 10 == 0:
                    end_time = time.time()
                    duration = end_time - previous_time
                    num_examples_per_step = BATCH_SIZE * 10
                    examples_per_sec = num_examples_per_step / duration
                    print("%s: %d[epoch]: %d[iteration]: train loss %f: %d[examples/iteration]: %f[examples/sec]: %f[sec/iteration]" % (datetime.now(), step, index, loss_value, num_examples_per_step, examples_per_sec, duration))
                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                previous_time = end_time
                index += 1
                loss_list.append(loss_value)

            else:
                avg_loss_list.append(sum(loss_list)/len(loss_list))

                summary_str = sess.run(summary_op, feed_dict={keep_conv: 1.0, keep_hidden: 1.0})
                summary_writer.add_summary(summary_str, step)

            if step % 1 == 0 or (step * 1) == MAX_STEPS:
                images_debug_eval, depths_debug_eval, output_vec, output_vec_argmax, cost_value = sess.run([images_val_debug, targets_val_debug, logits_val, logits_argmax_val, loss_val], feed_dict={keep_conv: 1.0, keep_hidden: 1.0})
                print("%s: %d[epoch]: %d[iteration]: validation loss: %f" % (datetime.now(), step, index, cost_value))

            if step % 10 == 0 or (step * 1) == MAX_STEPS:
                output_dir = "predicts_%05dstep" % (step)
                print("predicts output: %s" % output_dir)
                image_input.output_predict(output_vec_argmax, output_dir)
                image_input.output_images(images_debug_eval, output_dir)
                image_input.output_depths(depths_debug_eval, output_dir)

                # if FLAGS.scale3_train:
                #     scale1and2_output_dir = "scale1and2_%05dstep" % (step)
                #     print("scale1and2 output: %s" % scale1and2_output_dir)
                #     dataset.output_predict(coarse_output_vec, coarse_output_dir)

            if step % 30 == 0 or (step * 1) == MAX_STEPS:
                checkpoint_path = TRAIN_DIR + '/model.ckpt'
                saver.save(sess, checkpoint_path, global_step=step)

                if previous_loss < sum(avg_loss_list) / len(avg_loss_list) and saturate_step == 0:
                    checkpoint_path = EX_DIR + '/model.ckpt'
                    saver.save(sess, checkpoint_path, global_step=step)
                    dropout_flag = True
                    saturate_step = step
                previous_loss = sum(avg_loss_list) / len(avg_loss_list)
                print("30steps average loss: %f " % previous_loss)
                avg_loss_list = []

        coord.request_stop()
        coord.join(threads)
        sess.close()


def main(argv=None):
    #test()

    if not gfile.Exists(TRAIN_DIR):
        gfile.MakeDirs(TRAIN_DIR)

    train()



if __name__ == '__main__':
    tf.app.run()
