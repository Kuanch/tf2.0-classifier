from __future__ import absolute_import, division, print_function, unicode_literals

from argparse import ArgumentParser

import tensorflow as tf
from tensorflow.python.ops import control_flow_util
control_flow_util.ENABLE_CONTROL_FLOW_V2 = True

from model.model import *
from dataset.create_dataset import create_dataset
from preprocess.preprocess import preprocess_for_train


parser = ArgumentParser()

parser.add_argument('--tfrecord_path')

parser.add_argument('--train_image_size', type=int)

parser.add_argument('--image_width', type=int)

parser.add_argument('--image_height', type=int)

parser.add_argument('--num_label', type=int)

parser.add_argument('--batch_size', type=int)

parser.add_argument('--num_epoch', type=int)

args = parser.parse_args()


TOTAL_IMAGE_NUM = 196086


def _total_step_one_epoch():
  return TOTAL_IMAGE_NUM / args.batch_size



def train_one_step(model, optimizer, loss_fn, x_train, y_train, metrics):

  with tf.GradientTape() as tape:
    logits = model(x_train)  # Logits for this minibatch
    # Loss value for this minibatch
    loss_value = loss_fn(y_true=y_train, y_pred=logits)
    # Add extra losses created during this forward pass:
    loss_value += sum(model.losses)

  grads = tape.gradient(loss_value, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
  metrics(y_train, logits)

  return loss_value



# Iterate over the batches of a dataset.
def train(model, optimizer, loss_fn, dataset, metrics, epoch):

  loss = 0.0
  accuracy = 0.0
  for e in range(epoch):
    step = 0
    for images, labels in dataset:
      step += 1
      
      loss = train_one_step(model, optimizer, loss_fn, images, labels, metrics)

      if tf.equal(step % 10, 0):
        tf.print('step', step, 'loss', loss, 'accuracy', metrics.result())
    tf.print('epoch', e, 'step', step, 'loss', loss, 'accuracy', metrics.result())
  tf.print('Final:step', step, 'loss', loss, 'accuracy', metrics.result())



def main():

  # Create data from iteration
  dataset = create_dataset(args.tfrecord_path, args.batch_size)
  
  # Preprocess on every images
  dataset = dataset.map(lambda images, labels:preprocess_for_train(images, labels,
                                       args.train_image_size, args.train_image_size,
                                       args.num_label), num_parallel_calls=8)


  tf.keras.backend.clear_session()
  learning_rate_decay = tf.optimizers.schedules.ExponentialDecay(0.001, 400, 0.9)
  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_decay)
  loss_fn = tf.keras.losses.CategoricalCrossentropy()
  compute_accuracy = tf.keras.metrics.CategoricalAccuracy()
  model = Inception(args.num_label)

  train(model, optimizer, loss_fn, dataset, compute_accuracy, args.num_epoch)

if __name__ == '__main__':
  main()


