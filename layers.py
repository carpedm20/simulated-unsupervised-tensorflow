import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.framework import add_arg_scope

def _update_dict(layer_dict, scope, layer):
  name = "{}/{}".format(tf.get_variable_scope().name, scope)
  layer_dict[name] = layer

@add_arg_scope
def resnet_block(
    inputs, scope, num_outputs=64, kernel_size=[3, 3],
    stride=[1, 1], padding="SAME", layer_dict={}):
  with tf.variable_scope(scope):
    layer = slim.conv2d(
        inputs, num_outputs, kernel_size, stride,
        padding=padding, activation_fn=tf.nn.relu, scope="conv1")
    layer = slim.conv2d(
        inputs, num_outputs, kernel_size, stride,
        padding=padding, scope="conv2")
    outputs = tf.nn.relu(tf.add(inputs, layer))
  _update_dict(layer_dict, scope, outputs)
  return outputs

@add_arg_scope
def repeat(inputs, repetitions, layer, layer_dict={}, **kargv):
  outputs = slim.repeat(inputs, repetitions, layer, **kargv)
  _update_dict(layer_dict, kargv['scope'], outputs)
  return outputs

@add_arg_scope
def conv2d(inputs, num_outputs, kernel_size, stride, layer_dict={}, **kargv):
  outputs = slim.conv2d(inputs, num_outputs, kernel_size, stride, **kargv)
  _update_dict(layer_dict, kargv['scope'], outputs)
  return outputs

@add_arg_scope
def max_pool2d(inputs, kernel_size=[3, 3], stride=[1, 1], layer_dict={}, **kargv):
  outputs = slim.max_pool2d(inputs, kernel_size, stride, **kargv)
  _update_dict(layer_dict, kargv['scope'], outputs)
  return outputs
