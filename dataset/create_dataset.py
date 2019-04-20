import tensorflow as tf


def _parse_function(proto):
  # define your tfrecord again. Remember that you saved your image as a string.
  keys_to_features = {'encoded_image': tf.io.FixedLenFeature([], tf.string),
                        'category_id': tf.io.FixedLenFeature([], tf.int64),
                        'image_name':tf.io.FixedLenFeature([], tf.string)}
    
  # Load one example
  parsed_features = tf.io.parse_single_example(proto, keys_to_features)
    
  # Decode bytes into RGB jpeg
  parsed_features['decoded_image'] = tf.io.decode_jpeg(
        parsed_features['encoded_image'], channels=3)

  parsed_features['category_id'] = tf.cast(
        parsed_features['category_id'], dtype=tf.int64)

  
  return (parsed_features['decoded_image'], parsed_features['category_id'], parsed_features['image_name'])



def create_dataset(tfrecord_path, batch_size=32, num_epoch=1, train_image_size=224, num_label,
                   preprocess_fn, shuffle_buffer=100, is_training=True, 
                   cifar10_mode=False):

 
  if cifar10_mode:
    (images, labels), _ = tf.keras.datasets.cifar10.load_data()

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    cifar10_mode = True
    tf.print('Cifar10 mode')

  else:
    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    
    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(_parse_function, num_parallel_calls=8)

    tf.print('Regular mode')
    

  dataset = dataset.map(lambda image, label, image_name:preprocess_fn(image, label,
                                       image_name, train_image_size, train_image_size,                                                                    num_label, is_training=is_training,
                                       cifar10_mode=cifar10_mode),
                                       num_parallel_calls=8)


  # Set the number of datapoints you want to load and shuffle 
  dataset = dataset.shuffle(shuffle_buffer)

  # This dataset will go on forever
  dataset = dataset.repeat(num_epoch)
  
  dataset = dataset.prefetch(buffer_size=batch_size*2)
  
  # Set the batchsize
  dataset = dataset.batch(batch_size, drop_remainder=False)
    
  return dataset


