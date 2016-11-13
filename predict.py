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
TEST_DIR = FLAGS.test_dir
SAMPLES = FLAGS.num_samples

LOG_DEVICE_PLACEMENT = FLAGS.log_device_placement

#accuracy
import segmentation_accuracy as seg_acc

def predict():
    '''
    Predict
    '''
    #with tf.Graph().as_default():
    # globalなstep数
    global_step = tf.Variable(0, trainable=False)

    csv_test = FLAGS.test_csv

    image_input = DataSet()
    lines = image_input.load_csv(csv_test)
    images_val, targets_val = image_input.csv_test_inputs(csv_test, FLAGS.test_examples)

    images_val_debug = model.debug(images_val)

    keep_conv = tf.placeholder(tf.float32)
    keep_hidden = tf.placeholder(tf.float32)

    # graphのoutput
    images_ph = tf.placeholder(tf.float32, [1, 360, 480, 3])
    former_output_val = model.inference_segnet_former(images_ph, keep_conv, keep_hidden)
    en_pool1 = tf.placeholder(tf.float32, [1, 180, 240, 64])
    en_pool2 = tf.placeholder(tf.float32, [1, 90, 120, 128])
    feature_map = (en_pool1, en_pool2)
    logits_val, logits_argmax_val = model.inference_segnet_latter(images_ph, feature_map, FLAGS.num_classes, keep_conv, keep_hidden, batch_size=FLAGS.test_examples)

    # サマリー
    summary_op = tf.merge_all_summaries()

    # 初期化オペレーション
    init_op = tf.initialize_all_variables()

    # Session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=LOG_DEVICE_PLACEMENT))

    # saver
    #saver = tf.train.Saver(tf.all_variables())

    sess.run(init_op)

    # make up list of saving parameter
    segnet_params = {}
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
    saver = tf.train.Saver(tf.all_variables())

    # load model parameters
    segnet_ckpt = tf.train.get_checkpoint_state(TRAIN_DIR)
    if segnet_ckpt and segnet_ckpt.model_checkpoint_path:
        print("Pretrained segnet Model Loading.")
        print("model path: %s" % (segnet_ckpt.model_checkpoint_path))
        saver.restore(sess, segnet_ckpt.model_checkpoint_path)
        print("Pretrained segnet Model Restored.")
    else:
        print("No Pretrained segnet Model.")
        exit()

    # get threads coordinator
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    annotation_list = []
    predict_list = []
    for i, line in enumerate(lines):
        pil_img = Image.open(line[0])
        img_array = np.asarray(pil_img)
        print "img shape:"
        print img_array.shape

        pil_annot_img = Image.open(line[1])
        annot_img_array = np.asarray(pil_annot_img)

        print "input shape:"
        img_array = img_array[None, ...]
        print img_array.shape
        with tf.Graph().as_default():
            feature_vec = sess.run(former_output_val, feed_dict={images_ph: img_array, keep_conv: 1.0, keep_hidden: 1.0})
        samples_list = []

        for sample in range(SAMPLES):
            with tf.Graph().as_default():
                output_vec, output_vec_argmax = sess.run([logits_val, logits_argmax_val], feed_dict={en_pool1: feature_vec[0], en_pool2: feature_vec[1], keep_conv: 0.5, keep_hidden: 0.5})
            tf.reset_default_graph()
            samples_list.append(output_vec)

        mean_output_vec, var_output_vec = model.samples_var(samples_list)
        mean_output_vec_argmax = model.pixelwise_argmax(mean_output_vec)

        #convert tensor to numpy_array
        with tf.Session():
            mean_output_vec_argmax = mean_output_vec_argmax.eval()
            var_output_vec = var_output_vec.eval()


        print("predicts output: %s" % TEST_DIR)
	imagename = line[0].split("/")[3]
        image_input.output_test(img_array, mean_output_vec_argmax, var_output_vec, TEST_DIR, i, imagename)


        print "annot_img_array shape:"
        print annot_img_array.shape
        print "mean_output_vec_argmax shape:"
        print mean_output_vec_argmax.shape
        annotation_list.append(annot_img_array)
        predict_list.append(mean_output_vec_argmax)


    filename = "accuracy_result/" + segnet_ckpt.model_checkpoint_path.replace("train/model.ckpt-", "") + "-bayesian.txt"
    with open(filename, "w") as f:
        seg_acc.global_accuracy(annotation_list, predict_list, f)
        seg_acc.class_average_accuracy(annotation_list, predict_list, f)
        seg_acc.mean_interaction_over_union(annotation_list, predict_list, f)



    coord.request_stop()
    coord.join(threads)
    sess.close()


def main(argv=None):
    if not gfile.Exists(TEST_DIR):
        gfile.MakeDirs(TEST_DIR)

    predict()


if __name__ == '__main__':
    tf.app.run()
