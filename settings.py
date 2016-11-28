# encoding: utf-8

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# train settings
flags.DEFINE_integer('batch_size', 3, 'the number of images in a batch.')
flags.DEFINE_string('train_tfrecords', 'data/make3d_depth_max_norm.tfrecords', 'path to tfrecords file for training.')
flags.DEFINE_string('train_csv', 'data/CamVid/train.txt', 'path to tf csv file for training.')
flags.DEFINE_integer('image_height', 360, 'image height.')
flags.DEFINE_integer('image_width', 480, 'image width.')
flags.DEFINE_integer('depth_height', 360, 'depth height')
flags.DEFINE_integer('depth_width', 480, 'depth width')
flags.DEFINE_integer('image_depth', 3, 'image depth.')
flags.DEFINE_boolean('is_crop', False, 'is crop.')
flags.DEFINE_integer('crop_size_height', 240, 'crop size of image height.')
flags.DEFINE_integer('crop_size_width', 320, 'crop size of image height.')
flags.DEFINE_boolean('is_resize', False, 'is resize')
flags.DEFINE_float('learning_rate', 1e-4, 'initial learning rate.')
flags.DEFINE_float('learning_rate_decay_factor', 0.9, 'learning rate decay factor.')
flags.DEFINE_float('num_epochs_per_decay', 30.0, 'epochs after which learning rate decays.')
flags.DEFINE_float('moving_average_decay', 0.999999, 'decay to use for the moving averate.')
flags.DEFINE_integer('num_examples_per_epoch_for_train', 500, 'the number of examples per epoch train.')
flags.DEFINE_integer('num_examples_per_epoch_for_eval', 20, 'the number of examples per epoch eval.')
flags.DEFINE_string('tower_name', 'tower', 'multiple GPU prefix.')
flags.DEFINE_integer('num_classes', 13, 'the number of classes.')
flags.DEFINE_integer('num_threads', 4, 'the number of threads.')
flags.DEFINE_boolean('fine_tune', False, 'is use pre-trained model in trained_model')
flags.DEFINE_boolean('train', True, 'train coarse parameters')
flags.DEFINE_string('trained_model', 'trained_model', 'where to saved trained model for fine tuning.')
flags.DEFINE_float('si_lambda', 0.5, 'rate of the scale invaliant error.')

# output logs settings
flags.DEFINE_string('train_dir', 'train', 'directory where to write even logs and checkpoint')
flags.DEFINE_integer('max_steps', 1000000000, 'the number of batches to run.')
flags.DEFINE_boolean('log_device_placement', False, 'where to log device placement.')
flags.DEFINE_string('log_dir', 'log', 'directory where to write logs for tensorboard')


# evaluate settings
flags.DEFINE_string('eval_dir', 'eval', 'directory where to write event logs.')
flags.DEFINE_string('eval_tfrecords', 'data/make3d_depth_max_norm_val.tfrecords', 'path to tfrecords file for eval')
flags.DEFINE_string('eval_csv', 'data/CamVid/val.txt', 'path to tf csv file for validation')
flags.DEFINE_string('checkpoint_dir', 'train', 'directory where to read model checkpoints.')
flags.DEFINE_integer('eval_interval_secs', 30, 'How to often to run the eval.'),
flags.DEFINE_integer('num_examples', 3, 'the number of examples to run.')
flags.DEFINE_boolean('run_once', False, 'whether to run eval only once.')

# test settings
flags.DEFINE_string('test_dir', 'test', 'directory where to write test result')
flags.DEFINE_string('test_csv', 'data/CamVid/test.txt', 'path to tf csv file for test.')
flags.DEFINE_integer('test_examples', 1, 'number of examples to test')
flags.DEFINE_integer('num_samples', 40, 'the number of Monte Carlo samples.')

# experiment settings
flags.DEFINE_string('experiment_dir', 'experiment', 'directory where to write experiment result')
