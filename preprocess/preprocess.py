from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.python.ops import control_flow_util, control_flow_ops
control_flow_util.ENABLE_CONTROL_FLOW_V2 = True



def apply_with_random_selector(x, func, num_cases):

  sel = tf.random.uniform([], maxval=num_cases, dtype=tf.int32)
  # Pass the real x only to one of the func calls.
  return control_flow_ops.merge([
      func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
      for case in range(num_cases)])[0]



def distort_color(image, color_ordering=0, fast_mode=True, scope=None):

  if fast_mode:
    if color_ordering == 0:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    else:
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_brightness(image, max_delta=32. / 255.)

  return tf.clip_by_value(image, 0.0, 1.0)



def preprocess_for_train(images, labels, image_name, width, height, num_label, fast_mode=True, distort_color=False, is_training=True, cifar10_mode=False):

  if images.dtype != tf.float32:
    images = tf.image.convert_image_dtype(images, dtype=tf.float32)

  # We select only 1 case for fast_mode bilinear.
  # Default: ResizeMethod.BILINEAR
  distorted_images = tf.image.resize(images, [height, width])

  # Normalization
  if distort_color:
    num_distort_cases = 1 if fast_mode else 4
    distorted_images = apply_with_random_selector(
        distorted_images, \
        lambda x, ordering: distort_color(x, ordering, fast_mode), \
        num_cases=num_distort_cases)
  distorted_images = tf.subtract(distorted_images, 0.5)
  distorted_images = tf.multiply(distorted_images, 2.0)

  if is_training:
    # Randomly filp images
    distorted_images = tf.image.random_flip_left_right(distorted_images)

    # Label process
    labels = tf.one_hot(labels, num_label)
    if cifar10_mode:
      labels = tf.squeeze(labels, axis=1)
    return distorted_images, labels

  else:
    return distorted_images, labels, image_name

