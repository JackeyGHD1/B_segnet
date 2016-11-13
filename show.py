
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


filename_queue = tf.train.string_input_producer(['data/CamVid/rgb.png']) #  list of files to read
filename_queue2 = tf.train.string_input_producer(['data/CamVid/uncertainty.png']) #  list of files to read

reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)
key, value2 = reader.read(filename_queue2)


my_img = tf.image.decode_png(value, channels=3) # use png or jpg decoder based on your files.
gray = tf.image.decode_png(value2, channels=3) # use png or jpg decoder based on your files.
gray = tf.image.rgb_to_grayscale(gray)
#my_img = tf.cast(my_img, tf.float32)
#my_img.set_shape([200, 200, 3])
#my_img = tf.random_crop(my_img, [200,400,3])
print my_img.get_shape()
#my_img = tf.image.resize_images(my_img, (300, 480))
print gray.get_shape()

four = tf.concat(2, [my_img, gray])
print four.get_shape()
#my_img = tf.image.random_hue(my_img, max_delta=0.3)
init_op = tf.initialize_all_variables()
with tf.Session() as sess:
  sess.run(init_op)

  # Start populating the filename queue.

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  for i in range(1): #length of your filename list
    image = four.eval() #here is your image Tensor :)

  print(image)
  #plt.imshow(image)
  #plt.imshow(Image.fromarray(np.asarray(image)))
  plt.show()

  coord.request_stop()
  coord.join(threads)
