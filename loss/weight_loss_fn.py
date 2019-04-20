from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from itertools import product



def create_loss_weight(num_label, weight_category, weight):
  """
  Args:
    weight_category: Which category you want to weight.
    weight: How much the predict confidence will be scale.
  """
  w_array = np.ones((num_label, num_label))
  w_array[:, weight_category] = weight
  return w_array



def create_weight_mask(y_true, y_pred, weights):
  nb_cl = len(weights)
  weights = tf.cast(weights, dtype=tf.float32)
  final_mask = K.zeros_like(y_pred[:, 0])
  y_pred_max = K.max(y_pred, axis=1)
  y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
  y_pred_max_mat = K.equal(y_pred, y_pred_max)
  y_pred_max_mat = tf.cast(y_pred_max_mat, dtype=tf.float32)
  for c_p, c_t in product(range(nb_cl), range(nb_cl)):
      final_mask += weights[c_t, c_p] * y_true[:, c_t] * y_pred_max_mat[:, c_p]
      
  return final_mask


