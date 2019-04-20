from __future__ import absolute_import, division, print_function

import tensorflow as tf


def evaluate_for_iwild(model, dataset, save_eval_result_path):
  
  with open(save_eval_result_path, 'w') as csv:
    
    step = 0
    csv.write('Id,Predicted\n')
    for image, _, image_name in dataset:

      # Prediction
      predict = model(image)
      
      # Tensor to scalar and string
      pred_class = tf.argmax(predict, axis=1)
      pred_class = pred_class.numpy()[0]
      image_name = image_name.numpy()[0]

      # Write result
      result = str(image_name) + ',' + str(pred_class) + '\n'
      csv.write(result)

      step += 1
      if tf.equal(step % 100, 0):
        tf.print('step', step, 'has been complete')

    csv.close()
