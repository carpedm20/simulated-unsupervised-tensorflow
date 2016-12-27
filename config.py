#-*- coding: utf-8 -*-
import argparse

def str2bool(v):
  return v.lower() in ('true', '1')

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
  arg = parser.add_argument_group(name)
  arg_lists.append(arg)
  return arg

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--kernel_dims', type=eval, default='[]', help='')
net_arg.add_argument('--stride_size', type=eval, default='[]', help='')
net_arg.add_argument('--channel_dims', type=eval, default='[]', help='')

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--data_set', type=str, default='gaze')
data_arg.add_argument('--data_dir', type=str, default='data')

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--optimizer', default='rmsprop', help='')
train_arg.add_argument('--batch_size', type=int, default=512, help='')
train_arg.add_argument('--num_epochs', type=int, default=12, help='')
train_arg.add_argument('--random_seed', type=int, default=123, help='')
train_arg.add_argument('--learning_rate', type=float, default=0.001, help='')

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--log_dir', type=str, default='log')
misc_arg.add_argument('--debug', type=str2bool, default=True)

def get_config():
  config, unparsed = parser.parse_known_args()
  return config, unparsed
