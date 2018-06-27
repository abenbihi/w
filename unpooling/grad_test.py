"""A binary to train using a single GPU.
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

import model

def grad_test(img):
    """
    Encoder-Decoder mode. Shows how to fill the input of unpooling when
    using convolutions. Test of the gradient flow is available using
    tensorboard.
    Args:
        img: input image
    """
    with tf.Graph().as_default():
        # Generate training graph
        global_step = tf.Variable(0, trainable=False) 
        img_op = tf.placeholder(dtype=tf.float32, shape=[1,300,300,3],name='input')
        out_op = model.net(img_op, is_training=True)
        loss_op = model.loss(img_op, out_op)
        train_op = model.train(loss_op, global_step)
        
        # Setup session
        summary_op = tf.summary.merge_all()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # Start the queue runners.
            coord = tf.train.Coordinator()
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True,start=True))
                summary_writer = tf.summary.FileWriter('log/',graph_def=sess.graph_def)
   
                for step in range(10):
                    _, loss = sess.run([train_op, loss_op],feed_dict={img_op:img})
                    summary_str = sess.run(summary_op,feed_dict={img_op:img})
                    summary_writer.add_summary(summary_str, step)

            except Exception as e:  # pylint: disable=broad-except
                coord.request_stop(e)
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)

if __name__ == '__main__':  
    img_fn = 'img/img_test.jpg'
    img = cv2.imread(img_fn)[0:300,0:300]
    img = np.expand_dims(img, axis=0)
    print(img.shape)
    grad_test(img)
