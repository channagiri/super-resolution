    # -*- coding: utf-8 -*-
"""
Created on Tue May 12 09:56:31 2020

@author: MOHAMED
"""

from __future__ import absolute_import, division, print_function
import os
import tensorflow as tf
FLAGS = tf.compat.v1.flags.Flag
import numpy as np
import matplotlib.pyplot as plt



 
class Dataset:
    def __init__(self, dataset, num_epoch, batch_size=5, image_size=128, split='train'):
        # initialize the attributes of the class Dataset
        self.height = self.width = image_size
        self.channels = 3
        self.dataset = dataset
        self.num_epoch = num_epoch
        self.split = split
        self.idx = 0
        #join one or more path components
        self.basepath = os.path.join('./data', dataset, split)
        #batch size = the number of training examples in one forward/backward pass.
        #The higher the batch size, the more memory space we will need.
        self.batch_size = batch_size
        print("batch_size = " + str(self.batch_size)+"\n")
        

        self.base_fnames = sorted([os.path.basename(fname)[:-4] for fname in os.listdir(self.basepath) if 'jpg' in fname])[:50000]
        self.fnames = [os.path.join(self.basepath, fname + '.jpg') for fname in self.base_fnames] 
        self.batches_per_epoch = len(self.fnames) // self.batch_size
        print("batches_per_epoch = " + str(self.batches_per_epoch)+"\n") 
        self.number_of_batches = self.batches_per_epoch * self.num_epoch
        print("number_of_batches = " + str(self.number_of_batches)+"\n")
        min_after_dequeue = 1000
        capacity = min_after_dequeue + 10 * self.batch_size

        # Convert fnames to Tensor
        fnames = tf.convert_to_tensor(self.fnames, dtype=tf.string)
        
        # Create a queue with jpg filenames  
        queue = tf.compat.v1.train.slice_input_producer([fnames], num_epochs= num_epoch) 
        
        # Grab individual fnames from queue
        ims = tf.io.read_file( queue[0] )
             
        # Read in images  
        ims = tf.image.decode_jpeg(ims, channels=3) 
        ims = tf.cast(ims, tf.float32)
        #ims = (ims / 128.0) - 0.5


        # Extract random crops of size image_size from the larger patch
        self.hr_ims = tf.stack([tf.cast(tf.image.random_crop(ims, [128, 128, 3], seed=i), tf.float32) for i in range(32)], 0)
        self.hr_ims = tf.compat.v1.image.resize_bilinear(self.hr_ims, (32, 32))
        self.lr_ims = tf.compat.v1.image.resize_bicubic(self.hr_ims, (8, 8))
        self.hr_ims = tf.cast(self.hr_ims, tf.float32)
        self.lr_ims = tf.cast(self.lr_ims, tf.float32)
        
  

        min_after_dequeue = 1000
        capacity = min_after_dequeue + 400 * batch_size

        # batches images of shape [batch_size, 32, 32, 3],[batch_size, 8, 8, 3]

        self.hr_images, self.lr_images = tf.compat.v1.train.shuffle_batch([self.hr_ims, self.lr_ims], batch_size=batch_size,min_after_dequeue=min_after_dequeue, capacity=capacity, enqueue_many=True)
        

        

