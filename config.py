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
data_arg.add_argument('--input_height', type=int, default=35)
data_arg.add_argument('--input_width', type=int, default=55)
data_arg.add_argument('--input_channel', type=int, default=1)
data_arg.add_argument('--max_synthetic_num', type=int, default=1200000)
data_arg.add_argument('--mpiigaze_dir', type=str, default="MPIIGaze")
data_arg.add_argument('--unityeye_dir', type=str, default="UnityEyes")

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--task', type=str, default='generative', 
                       choices=['generative', 'estimation'], help='')
train_arg.add_argument('--is_train', type=str2bool, default=True, help='')
train_arg.add_argument('--optimizer', type=str, default='rmsprop', help='')
train_arg.add_argument('--max_step', type=int, default=200, help='')
train_arg.add_argument('--reg_scale', type=float, default=1, help='')
train_arg.add_argument('--initial_K_d', type=int, default=200, help='')
train_arg.add_argument('--initial_K_g', type=int, default=1000, help='')
train_arg.add_argument('--K_d', type=int, default=1, help='')
train_arg.add_argument('--K_g', type=int, default=2, help='')
train_arg.add_argument('--batch_size', type=int, default=512, help='')
train_arg.add_argument('--buffer_size', type=int, default=5120, help='')
train_arg.add_argument('--num_epochs', type=int, default=12, help='')
train_arg.add_argument('--random_seed', type=int, default=123, help='')
train_arg.add_argument('--learning_rate', type=float, default=0.001, help='')
train_arg.add_argument('--checkpoint_secs', type=int, default=300, help='')

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--log_step', type=int, default=20, help='')
misc_arg.add_argument('--log_dir', type=str, default='logs')
misc_arg.add_argument('--sample_dir', type=str, default='samples')
misc_arg.add_argument('--load_path', type=str, default='')
misc_arg.add_argument('--debug', type=str2bool, default=True)

def get_config():
  config, unparsed = parser.parse_known_args()
  return config, unparsed
