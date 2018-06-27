"""
Test unpool operator
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import array_ops


# Copy me, use me, lead the path to the unpooling world :)
# Snippet to add to your model file
@ops.RegisterGradient("MaxPoolGradWithArgmax")
def _MaxPoolGradGradWithArgmax(op, grad):
  print(len(op.outputs))
  print(len(op.inputs))
  print(op.name)
  return (array_ops.zeros(
      shape=array_ops.shape(op.inputs[0]),
      dtype=op.inputs[0].dtype), array_ops.zeros(
          shape=array_ops.shape(op.inputs[1]), dtype=op.inputs[1].dtype),
          gen_nn_ops._max_pool_grad_grad_with_argmax(
              op.inputs[0],
              grad,
              op.inputs[2],
              op.get_attr("ksize"),
              op.get_attr("strides"),
              padding=op.get_attr("padding")))
 

def rec(img):
    """
    Reconstruct part of the image after simple pooling
    Args:
        img: input image
    """
    with tf.Graph().as_default():
        img_op = tf.placeholder(dtype=tf.float32, shape=[1,300,300,3],name='input')
        pool_op = tf.nn.max_pool(img_op, ksize=[1,2,2,1], strides=[1,2,2,1],padding='VALID', name='pool')
        unpool_op = gen_nn_ops._max_pool_grad(img_op, pool_op, pool_op, [1,2,2,1], [1,2,2,1],'VALID')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            pool_out, unpool_out = sess.run([pool_op, unpool_op],feed_dict={img_op:img})
            cv2.imwrite('img/pool_out.png', pool_out[0,:,:,:])
            cv2.imwrite('img/unpool_out.png', unpool_out[0,:,:,:])
            
if __name__ == '__main__':  
    img_fn = 'img/img_test.jpg'
    img = cv2.imread(img_fn)[0:300,0:300]
    img = np.expand_dims(img, axis=0)
    print(img.shape)
    rec(img)
