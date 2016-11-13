#encoding: utf-8
import os
import random
from glob import glob
import tensorflow as tf
import numpy as np
import csv

from tensorflow.python.platform import gfile

from PIL import Image
import time

# settings
import settings
FLAGS = settings.FLAGS
BATCH_SIZE = FLAGS.batch_size
IS_CROP = FLAGS.is_crop
IS_RESIZE = FLAGS.is_resize
IMAGE_HEIGHT = FLAGS.image_height
IMAGE_WIDTH = FLAGS.image_width

DEPTH_HEIGHT = FLAGS.depth_height
DEPTH_WIDTH = FLAGS.depth_width

NUM_THREADS = FLAGS.num_threads

Animal = [64, 128, 64]
Archway = [192, 0, 128]
Bicyclist = [0, 128, 192]
Bridge = [0, 128, 64]
Building = [128, 0, 0]
Car = [64, 0, 128]
CartLuggagePram = [64, 0, 192]
Child = [192, 128, 64]
Column_Pole = [192, 192, 128]
Fence = [64, 64, 128]
LaneMkgsDriv = [128, 0, 192]
LaneMkgsNonDriv = [192, 0, 64]
Misc_Text = [128, 128, 64]
MotorcycleScooter = [192, 0, 192]
OtherMoving = [128, 64, 64]
ParkingBlock = [64, 192, 128]
Pedestrian = [64, 64, 0]
Road = [128, 64, 128]
RoadShoulder = [128, 128, 192]
Sidewalk = [0, 0, 192]
SignSymbol = [192, 128, 128]
Sky = [128, 128, 128]
SUVPickupTruck = [64, 128, 192]
TrafficCone = [0, 0, 64]
TrafficLight = [0, 64, 64]
Train = [192, 64, 128]
Tree = [128, 128, 0]
Truck_Bus = [192, 128, 192]
Tunnel = [64, 0, 64]
VegetationMisc = [192, 192, 0]
Void = [0, 0, 0]
Wall = [64, 192, 0]


class DataSet:
    def __init__(self):
        pass

    def _generate_image_and_label_batch_pare(self, image, depth, min_queue_examples, batch_size):
        '''
        imageとlabelのmini batchを生成
        '''
        num_preprocess_threads = NUM_THREADS
        images, depths = tf.train.batch(
            [image, depth],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size
        )

        # Display the training images in the visualizer
        # tf.image_summary('images', images, max_images=BATCH_SIZE)
        return images, depths

    def _generate_image_and_label_batch(self, image, depth, invalid_depth, min_queue_examples, batch_size):
        '''
        imageとlabelのmini batchを生成
        '''
        num_preprocess_threads = NUM_THREADS
        images, depths, invalid_depths = tf.train.shuffle_batch(
            [image, depth, invalid_depth],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * BATCH_SIZE,
            min_after_dequeue=min_queue_examples
        )
    
        # Display the training images in the visualizer
        #tf.image_summary('images', images, max_images=BATCH_SIZE)
        return images, depths, invalid_depths

    def tfrecords_inputs(self, tfrecords, batch_size):
        filename_queue = tf.train.string_input_producer([tfrecords])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        datas = tf.parse_single_example(
            serialized_example,
            features={
                "depth": tf.FixedLenFeature([], dtype=tf.string),
                "image_raw": tf.FixedLenFeature([], dtype=tf.string),
            }
        )
        image = tf.image.decode_png(datas['image_raw'], channels=3)
        image = tf.image.resize_images(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
        image = tf.cast(image, tf.float32)
        depth = tf.image.decode_png(datas['depth'], channels=1)
        depth = tf.image.resize_images(depth, (DEPTH_HEIGHT, DEPTH_WIDTH))
        depth = tf.cast(depth, tf.float32)
        depth = tf.div(depth, [255.0])
        invalid_depth = tf.sign(depth)

        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(100 * min_fraction_of_examples_in_queue)
        print ('filling queue with %d train images before starting to train.  This will take a few minutes.' % min_queue_examples)
    
        return self._generate_image_and_label_batch(image, depth, invalid_depth, min_queue_examples, batch_size)

    def load_csv(self, path):
        print("load csv: %s" % (path))
        images = []
        with open(path, 'r') as f:
            rows = csv.reader(f)
            for row in rows:
                print row
                images.append(row)
        return images


    def csv_test_inputs(self, csv, example_size):
        print("test csv: %s" % (csv))
        filename_queue = tf.train.string_input_producer([csv], shuffle=False)
        reader = tf.TextLineReader()
        _, serialized_example = reader.read(filename_queue)
        filename, depth_filename = tf.decode_csv(serialized_example, [["path"], ["annotation"]])

        png = tf.read_file(filename)
        image = tf.image.decode_png(png, channels=3)
        print image.dtype
        image = tf.cast(image, tf.float32)
        print image.dtype
        #image.set_shape([IMAGE_HEIGHT_ORG, IMAGE_WIDTH_ORG, IMAGE_DEPTH_ORG])
        image = tf.image.resize_images(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
       
        depth_png = tf.read_file(depth_filename)
        depth = tf.image.decode_png(depth_png, channels=1)
        print depth.dtype
        depth = tf.cast(depth, tf.int64)
        print depth.dtype
        #depth.set_shape([IMAGE_HEIGHT_ORG, IMAGE_WIDTH_ORG, 1])
        depth = tf.image.resize_images(depth, (DEPTH_HEIGHT, DEPTH_WIDTH))
 
        min_fraction_of_examples_in_queue = 0.4
        #min_fraction_of_examples_in_queue = 1
        min_queue_examples = int(example_size * min_fraction_of_examples_in_queue)
        print ('filling queue with %d train images before starting to train.  This will take a few minutes.' % min_queue_examples)
    
        return self._generate_image_and_label_batch_pare(image, depth, min_queue_examples, example_size)

    def csv_inputs(self, csv, batch_size):
        print csv
        filename_queue = tf.train.string_input_producer([csv], shuffle=True)
        reader = tf.TextLineReader()
        _, serialized_example = reader.read(filename_queue)
        filename, depth_filename = tf.decode_csv(serialized_example, [["path"], ["annotation"]])

        png = tf.read_file(filename)
        image = tf.image.decode_png(png, channels=3)
        print image.dtype
        image = tf.cast(image, tf.float32)
        print image.dtype
        #image.set_shape([IMAGE_HEIGHT_ORG, IMAGE_WIDTH_ORG, IMAGE_DEPTH_ORG])
        image = tf.image.resize_images(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
       
        depth_png = tf.read_file(depth_filename)
        depth = tf.image.decode_png(depth_png, channels=1)
        print depth.dtype
        depth = tf.cast(depth, tf.int64)
        print depth.dtype
        #depth.set_shape([IMAGE_HEIGHT_ORG, IMAGE_WIDTH_ORG, 1])
        depth = tf.image.resize_images(depth, (DEPTH_HEIGHT, DEPTH_WIDTH))
 
        min_fraction_of_examples_in_queue = 0.4
        #min_fraction_of_examples_in_queue = 1
        min_queue_examples = int(100 * min_fraction_of_examples_in_queue)
        print ('filling queue with %d train images before starting to train.  This will take a few minutes.' % min_queue_examples)
    
        return self._generate_image_and_label_batch_pare(image, depth, min_queue_examples, batch_size)

    def csv_inputs_augumentation(self, csv, batch_size):
        print("trainig csv: %s" % (csv))
        filename_queue = tf.train.string_input_producer([csv], shuffle=True)
        reader = tf.TextLineReader()
        _, serialized_example = reader.read(filename_queue)
        filename, gt_filename = tf.decode_csv(serialized_example, [["path"], ["segmentation"]])

        # file load, decode and resize
        png = tf.read_file(filename)
        image = tf.image.decode_png(png, channels=3)
        image = tf.image.resize_images(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
        print("input images types: %s" % image.dtype)

        gt_png = tf.read_file(gt_filename)
        gt = tf.image.decode_png(gt_png, channels=1)
        print("ground truth types: %s" % (gt.dtype))

        # change data type for network inputs and targets
        gt = tf.cast(gt, tf.int64)
        gt = tf.image.resize_images(gt, [IMAGE_HEIGHT, IMAGE_WIDTH])

        # flip augumentation
        distortions = tf.random_uniform([2], 0, 1.0, dtype=tf.float32)
        image = self.image_distortions(image, distortions)
        gt = self.image_distortions(gt, distortions)

        # brightness augumentation
        image = tf.image.random_brightness(image, max_delta=0.6)

        # contrast augumentation
        image = tf.image.random_contrast(image, lower=0.2, upper=1.0)

        # satutatin augumentation
        image = tf.image.random_saturation(image, lower=0.2, upper=1.8)

        # hue augumentation
        image = tf.image.random_hue(image, max_delta=0.3)


        min_fraction_of_examples_in_queue = 0.4
        # min_fraction_of_examples_in_queue = 1
        min_queue_examples = int(100 * min_fraction_of_examples_in_queue)
        print('filling queue with %d train images before starting to train.  This will take a few minutes.' % min_queue_examples)

        return self._generate_image_and_label_batch_pare(image, gt, min_queue_examples, batch_size)

    def image_distortions(self, image, distortions):
        distort_left_right_random = distortions[0]
        mirror = tf.less(tf.pack([1.0, distort_left_right_random, 1.0]), 0.5)
        image = tf.reverse(image, mirror)
        # distort_up_down_random = distortions[1]
        # mirror = tf.less(tf.pack([distort_up_down_random, 1.0, 1.0]), 0.5)
        # image = tf.reverse(image, mirror)
        return image


    def output_images(self, images, output_dir):
        for i, image in enumerate(images):
            print type(image)
            print image.shape
            pilimg = Image.fromarray(np.uint8(image))
            image_name = "%s/%05d_org.png" % (output_dir, i)
            pilimg.save(image_name)

    def output_depths(self, depths, output_dir):
        for i, depth in enumerate(depths):
            depth = depth.transpose(2, 0, 1)
            print type(depth)
            print depth.shape
            depth = depth[0]

            r = depth.copy()
            g = depth.copy()
            b = depth.copy()

            label_colours = np.array([Sky, Building, Column_Pole, Road, Sidewalk, Tree, SignSymbol, Fence, Car
                                         , Pedestrian, Bicyclist, LaneMkgsDriv, Void])

            for l in range(0,13):
                r[depth==l] = label_colours[l,0]
                g[depth==l] = label_colours[l,1]
                b[depth==l] = label_colours[l,2]

            rgb = np.zeros((depth.shape[0], depth.shape[1], 3))
            rgb[:,:,0] = r
            rgb[:,:,1] = g
            rgb[:,:,2] = b

            pilimg = Image.fromarray(np.uint8(rgb))
            image_name = "%s/%05d_target.png" % (output_dir, i)
            pilimg.save(image_name)

    def output_predict(self, depths, output_dir):
        if not gfile.Exists(output_dir):
            gfile.MakeDirs(output_dir)
    
        print("the number of output predict: %d" % len(depths))
        for i, depth in enumerate(depths):
            class_set = set()
            #depth = depth.transpose(2, 0, 1)
            print depth.shape
            for l in xrange(depth.shape[0]):
                for m in xrange(depth.shape[1]):
                    class_set.add(depth[l][m])   
            print class_set

            r = depth.copy()
            g = depth.copy()
            b = depth.copy()

            label_colours = np.array([Sky, Building, Column_Pole, Road, Sidewalk, Tree, SignSymbol, Fence, Car
                                         , Pedestrian, Bicyclist, LaneMkgsDriv, Void])

            for l in range(0, 13):
                r[depth==l] = label_colours[l,0]
                g[depth==l] = label_colours[l,1]
                b[depth==l] = label_colours[l,2]

            rgb = np.zeros((depth.shape[0], depth.shape[1], 3))
            rgb[:,:,0] = r
            rgb[:,:,1] = g
            rgb[:,:,2] = b

            if np.max(depth) != 0:
                ra_depth = (depth/float(np.max(depth)))*255.0
            else:
                ra_depth = depth*255.0
            depth_pil = Image.fromarray(np.uint8(rgb))
            depth_name = "%s/%05d.png" % (output_dir, i)
            depth_pil.save(depth_name)


    def output_test(self, images, predicts, vars, output_dir, pred, imagename):
        for i, (image, depth, uncertainty) in enumerate(zip(images, predicts, vars)):
            print type(image)
            print image.shape
            pilimg = Image.fromarray(np.uint8(image))

            class_set = set()
            #depth = depth.transpose(2, 0, 1)
            print depth.shape
            for l in xrange(depth.shape[0]):
                for m in xrange(depth.shape[1]):
                    class_set.add(depth[l][m])
            print class_set

            r = depth.copy()
            g = depth.copy()
            b = depth.copy()

            label_colours = np.array([Sky, Building, Column_Pole, Road, Sidewalk, Tree, SignSymbol, Fence, Car
                                         , Pedestrian, Bicyclist, LaneMkgsDriv, Void])

            for l in range(0, 13):
                r[depth==l] = label_colours[l,0]
                g[depth==l] = label_colours[l,1]
                b[depth==l] = label_colours[l,2]

            rgb = np.zeros((depth.shape[0], depth.shape[1], 3))
            rgb[:,:,0] = r
            rgb[:,:,1] = g
            rgb[:,:,2] = b

            if np.max(depth) != 0:
                ra_depth = (depth/float(np.max(depth)))*255.0
            else:
                ra_depth = depth*255.0
            depth_pil = Image.fromarray(np.uint8(rgb))

            max_intensity = np.max(uncertainty)
            for h in xrange(IMAGE_HEIGHT):
                for w in xrange(IMAGE_WIDTH):
                    uncertainty[h][w] = 255 - (uncertainty[h][w] / max_intensity * 255)
            uncertainty_pil = Image.fromarray(np.uint8(uncertainty))

            canvas = Image.new('RGB', (1440, 360))
            canvas.paste(pilimg, (0, 0))
            canvas.paste(depth_pil, (480, 0))
            canvas.paste(uncertainty_pil, (960, 0))

            #canvas = Image.new('RGB', (480, 360))
            #canvas.paste(uncertainty_pil, (0, 0))

            #image_name = "%s/%05d_%05d_test.png" % (output_dir, pred, i)
            image_name = "%s/%s" % (output_dir, imagename)
            canvas.save(image_name, 'PNG', quality=100, optimize=True)


    # def output_test(self, images, predicts, output_dir, pred):
    #     for i, (image, depth) in enumerate(zip(images, predicts)):
    #         print type(image)
    #         print image.shape
    #         pilimg = Image.fromarray(np.uint8(image))
    #
    #         class_set = set()
    #         # depth = depth.transpose(2, 0, 1)
    #         print depth.shape
    #         for l in xrange(depth.shape[0]):
    #             for m in xrange(depth.shape[1]):
    #                 class_set.add(depth[l][m])
    #         print class_set
    #
    #         r = depth.copy()
    #         g = depth.copy()
    #         b = depth.copy()
    #
    #         label_colours = np.array(
    #             [Animal, Archway, Bicyclist, Bridge, Building, Car, CartLuggagePram, Child, Column_Pole,
    #              Fence, LaneMkgsDriv, LaneMkgsNonDriv, Misc_Text, MotorcycleScooter, OtherMoving,
    #              ParkingBlock, Pedestrian, Road, RoadShoulder, Sidewalk, SignSymbol, Sky, SUVPickupTruck,
    #              TrafficCone, TrafficLight, Train, Tree, Truck_Bus, Tunnel, VegetationMisc, Void, Wall])
    #         for l in range(0, 32):
    #             r[depth == l] = label_colours[l, 0]
    #             g[depth == l] = label_colours[l, 1]
    #             b[depth == l] = label_colours[l, 2]
    #
    #         rgb = np.zeros((depth.shape[0], depth.shape[1], 3))
    #         rgb[:, :, 0] = r
    #         rgb[:, :, 1] = g
    #         rgb[:, :, 2] = b
    #
    #         if np.max(depth) != 0:
    #             ra_depth = (depth / float(np.max(depth))) * 255.0
    #         else:
    #             ra_depth = depth * 255.0
    #         depth_pil = Image.fromarray(np.uint8(rgb))
    #
    #         canvas = Image.new('RGB', (960, 360))
    #         canvas.paste(pilimg, (0, 0))
    #         canvas.paste(depth_pil, (480, 0))
    #
    #         image_name = "%s/%05d_%05d_test.png" % (output_dir, pred, i)
    #         canvas.save(image_name, 'PNG', quality=100, optimize=True)




