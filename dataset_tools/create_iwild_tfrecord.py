import os
from argparse import ArgumentParser

import tensorflow as tf

import dataset_util


parser = ArgumentParser()

parser.add_argument('--info_file_path', help='path of images info file')

parser.add_argument('--images_path', help='path of folder containing images')

parser.add_argument('--tfrecord_output_path', help='where tfrecord is stored')

parser.add_argument('--is_training', dest='feature', action='store_true', help='if creating a training dataset')
parser.set_defaults(is_training=False)

args = parser.parse_args()




def read_image_name_and_category(info_file_path, is_training=True):

  num_image = 0
  folder_path = '/'.join(info_file_path.split('/')[:-1])

  with open(info_file_path, 'r') as f:
    
    for line in f.readlines():
      if not num_image:
        num_image += 1 #Skip first line
        continue
    
      line_split = line.split(',')

      if is_training:
        category_id = int(line_split[0])
        image_name = line_split[2]
        image_path = os.path.join(folder_path, 'train_images/' + image_name)
      
      else:
        category_id = 0
        image_name = line_split[1]
        image_path = os.path.join(folder_path, 'test_images/' + image_name)

      _encode_image_to_tfrecord(image_path, category_id)

      num_image += 1

      if num_image % 1000 == 0:
        print(num_image, 'have been processed')

    print('Total processed images:{}'.format(num_image))
    
  
  

def _encode_image_to_tfrecord(image_path, category_id):

  with tf.io.gfile.GFile(image_path, 'rb') as fid:
    encoded_jpg = fid.read()

  image_name = image_path.split('/')[-1]

  feature_dict = {

    'image_name':
      dataset_util.bytes_feature(image_name.encode('utf8')),

    'encoded_image':
      dataset_util.bytes_feature(encoded_jpg),

    'category_id':
      dataset_util.int64_feature(category_id),

    'format':
      dataset_util.bytes_feature('jpeg'.encode('utf8')) 

  }

  _create_tfrecord(tf.train.Example(features=tf.train.Features(feature=feature_dict)))



def _create_tfrecord(tf_example):
  
  tf_writer = tf.io.TFRecordWriter(args.tfrecord_output_path)
  tf_writer.write(tf_example.SerializeToString())
  tf_writer.close()



if __name__ == '__main__':

  if args.is_training:
    read_image_name_and_category(args.info_file_path)
  else:
    read_image_name_and_category(args.info_file_path, is_training=False)

