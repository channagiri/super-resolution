# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 12:07:02 2020

@author: MOHAMED
"""



from solver import *
import tensorflow as tf
import argparse

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

 

# parser.add_argument('--sample_dir', default='samples', type=str, help='Directory to save samples to.')
def main(_):
  solver = Solver()
  solver.train()

if __name__ == '__main__':
  tf.compat.v1.app.run()  