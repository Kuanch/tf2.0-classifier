from __future__ import absolute_import, division, print_function, unicode_literals

from argparse import ArgumentParser

import tensorflow as tf
from tensorflow.python.ops import control_flow_util
control_flow_util.ENABLE_CONTROL_FLOW_V2 = True

from model.model import *
from dataset.create_dataset import create_dataset
from preprocess.preprocess import preprocess_for_train
from loss.weight_loss_fn import create_loss_weight, create_weight_mask

from evaluation.evaluation import evaluate_for_iwild


parser = ArgumentParser()

parser.add_argument('--tfrecord_path')

parser.add_argument('--model_save_path')

parser.add_argument('--save_eval_result_path')

parser.add_argument('--train_image_size', type=int)

parser.add_argument('--image_width', type=int)

parser.add_argument('--image_height', type=int)

parser.add_argument('--num_label', type=int)

parser.add_argument('--batch_size', type=int)

parser.add_argument('--num_epoch', type=int)

parser.add_argument('--test_tfrecord_path')

parser.add_argument('--cifar10_mode', dest='cifar10_mode', action='store_true', help='if activating cifar test mode')

args = parser.parse_args()


TOTAL_IWILD_TRAIN_IMAGE = 196086



def train_one_step(model, optimizer, loss_fn, x_train, y_train, metrics, num_label):

  with tf.GradientTape() as tape:
    logits = model(x_train)  # Logits for this minibatch

    # Weights loss
    """ 
    When assign a specific class, it's confidence will always times weight. 
    If weight > 1, the model tend to not to allocate confidence on the class.
    Otherwise, predicitons will tend to fall into it.
    """
    weight_array = create_loss_weight(num_label, weight_category=0, weight=10)
    loss_mask = create_weight_mask(y_train, logits, weights=weight_array)
    # Loss value for this minibatch
    loss_value = loss_fn(y_true=y_train, y_pred=logits,sample_weight=loss_mask)
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
def train(model, optimizer, loss_fn, dataset, metrics, epoch, num_label):

  step = 0
  for images, labels in dataset:
    step += 1
    
    loss = train_one_step(model, optimizer, loss_fn, images, labels, metrics, num_label)

    if tf.equal(step % 10, 0):
      tf.print('step', step, 'loss', loss, 'accuracy', metrics.result())
  tf.print('Final:step', step, 'loss', loss, 'accuracy', metrics.result())


def main():

  if args.cifar10_mode:

    train_image_size = 32
    num_label = 10

    dataset = create_dataset(args.tfrecord_path, preprocess_for_train, num_label, args.batch_size,
                             args.num_epoch, train_image_size, cifar10_mode=True)
  

  else:
    
    train_image_size = args.train_image_size
    num_label = args.num_label

    dataset = create_dataset(args.tfrecord_path, preprocess_for_train, num_label, args.batch_size,                             args.num_epoch, train_image_size)


  # Clearing the session removes all the nodes left over
  tf.keras.backend.clear_session()
  
  # Learning rate config
  learning_rate_decay = tf.optimizers.schedules.ExponentialDecay(0.001, 100, 0.9)
  
  # Create Optimizer
  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_decay)
  
  # Cross-entorpy loss for classification
  loss_fn = tf.keras.losses.CategoricalCrossentropy()
  
  
  # Metrics for classification accuracy
  compute_accuracy = tf.keras.metrics.CategoricalAccuracy()

  # Create model
  num_label = 10 if args.cifar10_mode else args.num_label
  model = Inception(num_label)

  # Set up checkpoint
  checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

  # Training process
  train(model, optimizer, loss_fn, dataset, compute_accuracy, args.num_epoch, num_label)

  # Export the model to a checkpoint
  #checkpoint.save(file_prefix=args.model_save_path + 'ckpt')

  # Evaluate results
  if args.test_tfrecord_path:
    test_dataset = create_dataset(args.test_tfrecord_path, 1,
                                  train_image_size, num_label,
                                  preprocess_for_train, is_training=False)
    evaluate_for_iwild(model, test_dataset, args.save_eval_result_path)


if __name__ == '__main__':
  main()


