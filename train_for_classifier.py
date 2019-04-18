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

parser.add_argument('--model_save_path')

parser.add_argument('--train_image_size', type=int)

parser.add_argument('--image_width', type=int)

parser.add_argument('--image_height', type=int)

parser.add_argument('--num_label', type=int)

parser.add_argument('--batch_size', type=int)

parser.add_argument('--num_epoch', type=int)

parser.add_argument('--cifar10_test_mode', dest='feature', action='store_true', help='if activating cifar test mode')
parser.set_defaults(cifar10_test_mode=True)

args = parser.parse_args()


TOTAL_IWILD_TRAIN_IMAGE = 196086



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
# There is no tf.function is used here, since the incompability
# on dataset api and some eager mode features.
#@tf.function
def train(model, optimizer, loss_fn, dataset, metrics, epoch):

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

  if args.cifar10_test_mode:
    dataset = create_dataset(args.tfrecord_path, args.batch_size, cifar10_test_mode=True)
  
    train_image_size = 32
    num_label = 10
    test_mode = True

  else:
    dataset = create_dataset(args.tfrecord_path, args.batch_size)

    train_image_size = args.train_image_size
    num_label = args.num_label
    test_mode = False

  dataset = dataset.map(lambda images, labels:preprocess_for_train(images, labels,
                                       train_image_size, train_image_size,
                                       num_label, cifar10_test_mode=test_mode), num_parallel_calls=8)

  # Clearing the session removes all the nodes left over
  tf.keras.backend.clear_session()
  
  # Learning rate config
  learning_rate_decay = tf.optimizers.schedules.ExponentialDecay(0.001, 400, 0.9)
  
  # Create Optimizer
  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_decay)
  
  # Cross-entorpy loss for classification
  loss_fn = tf.keras.losses.CategoricalCrossentropy()
  
  # Metrics for classification accuracy
  compute_accuracy = tf.keras.metrics.CategoricalAccuracy()
  
  # Create model
  model = Inception(args.num_label)

  # Training process
  train(model, optimizer, loss_fn, dataset, compute_accuracy, args.num_epoch)

  # Export the model to a SavedModel
  tf.keras.experimental.export_saved_model(model, args.model_save_path)


if __name__ == '__main__':
  main()


