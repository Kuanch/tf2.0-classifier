
import argparse

import tensorflow as tf


def _parse_function(proto):
  # define your tfrecord again. Remember that you saved your image as a string.
  keys_to_features = {'encoded_image': tf.io.FixedLenFeature([], tf.string),
                        'category_id': tf.io.FixedLenFeature([], tf.int64)}
    
  # Load one example
  parsed_features = tf.io.parse_single_example(proto, keys_to_features)
    
  # Decode bytes into RGB jpeg
  parsed_features['decoded_image'] = tf.io.decode_jpeg(
        parsed_features['encoded_image'], channels=3)

  #parsed_features['class'] = tf.io.decode_raw(
  #      parsed_features['category_id'], tf.uint8)
    
  return (parsed_features['decoded_image'], parsed_features['category_id'])



def create_dataset(tfrecord_path, batch_size, shuffle_buffer=100):
    
  # This works with arrays as well
  dataset = tf.data.TFRecordDataset(tfrecord_path)
    
  # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
  dataset = dataset.map(_parse_function, num_parallel_calls=8)
    
  # Set the number of datapoints you want to load and shuffle 
  dataset = dataset.shuffle(shuffle_buffer)

  # This dataset will go on forever
  dataset = dataset.repeat()
  
  dataset = dataset.prefetch(buffer_size=batch_size*5)
  
  # Set the batchsize
  dataset = dataset.batch(batch_size)
    
  return dataset


