"""
Builds a simple encoder decoder, loss and optimizer
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import urllib
import tensorflow as tf
from math import sqrt
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import array_ops

IMAGE_SIZE=300

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
              
def net(img, is_training):
    size_filter=3
    i=1
    layers=[]
    conv_=[]
    arg_=[]
 
    with tf.variable_scope('l%d' %i) as scope: #1
        conv = tf.layers.conv2d(inputs=img,filters=1,kernel_size=(3, 3),padding="same",
                kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.contrib.layers.batch_norm(conv, fused=True, decay=0.9, is_training=is_training)
        out = tf.nn.relu(conv)
        layers.append(out)
        print(scope.name+'::out', out.get_shape())
        i+=1
  
    with tf.variable_scope('l%d' %i) as scope: #2
        conv = tf.layers.conv2d(inputs=layers[-1],filters=1,kernel_size=(3, 3),padding="same",
                kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.contrib.layers.batch_norm(conv, fused=True, decay=0.9, is_training=is_training)
        conv = tf.nn.relu(conv)
        out = tf.nn.max_pool(conv, ksize=[1,2,2,1], strides=[1,2,2,1],padding='VALID', name='pool')
        conv_.append(conv)
        layers.append(out)
        print(scope.name+'::out', out.get_shape())
        i+=1
  
    with tf.variable_scope('l%d' %i) as scope:#3
        conv = tf.layers.conv2d(inputs=layers[-1],filters=1,kernel_size=(3, 3),padding="same",
                kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.contrib.layers.batch_norm(conv, fused=True, decay=0.9, is_training=is_training)
        out = tf.nn.relu(conv)
        layers.append(out)
        print(scope.name+'::out', out.get_shape())
        i+=1

    with tf.variable_scope('l%d' %i) as scope:#4
        conv = tf.layers.conv2d(inputs=layers[-1],filters=1,kernel_size=(3,3),padding="same",
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.contrib.layers.batch_norm(conv, fused=True, decay=0.9, is_training=is_training)
        conv = tf.nn.relu(conv)
        out = tf.nn.max_pool(conv, ksize=[1,2,2,1], strides=[1,2,2,1],padding='VALID', name='pool')
        conv_.append(conv)
        layers.append(out)
        print(scope.name + '::out', out.get_shape())
        i+=1
  
    with tf.variable_scope('l%d' %i) as scope:#9
        conv=conv_[-1]
        unpool = gen_nn_ops._max_pool_grad(conv, layers[-1], layers[-1], [1,2,2,1], [1,2,2,1],'VALID')
        out = tf.layers.conv2d_transpose(unpool,filters= 1,kernel_size=(3,3),strides=(1,1),padding='same', 
                kernel_initializer=tf.contrib.layers.xavier_initializer())
        layers.append(out)
        print(scope.name+'::out', out.get_shape())
        i+=1
     
    with tf.variable_scope('l%d' %i) as scope:#11
        conv=conv_[0]
        pool=layers[1]
        unpool = gen_nn_ops._max_pool_grad(conv, pool, layers[-1], [1,2,2,1], [1,2,2,1],'VALID')
        out = tf.layers.conv2d_transpose(unpool,filters= 1,kernel_size=(3,3),strides=(1,1),padding='same', 
                kernel_initializer=tf.contrib.layers.xavier_initializer())
        layers.append(out)
        print(scope.name+'::out', out.get_shape())
        i+=1
    
    output = layers[-1]
    tf.summary.image('output',output) 
    return output


def loss(img, out):
  loss = tf.nn.l2_loss(img - out) /IMAGE_SIZE
  tf.add_to_collection('losses', loss)
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])
  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name +' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))
  return loss_averages_op


def train(total_loss, global_step):
  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)
  
  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.AdamOptimizer(1e-3,0.99, 0.999, 1e-8)
    update_ops =  tf.get_collection(tf.GraphKeys.UPDATE_OPS) #line for BN
    with tf.control_dependencies(update_ops):
        grads = opt.compute_gradients(total_loss)
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      0.9999, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())
  
  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')
  return train_op


