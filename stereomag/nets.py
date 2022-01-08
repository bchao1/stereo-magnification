#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Network definitions for multiplane image (MPI) prediction networks.
"""
from __future__ import division
import numpy as np
import tensorflow as tf

from tensorflow.contrib import slim

def mpi_net(inputs, num_outputs, num_mpi_planes, ngf=64, vscope='net', reuse_weights=False):
    """Network definition for multiplane image (MPI) inference.

    Args:
        inputs: stack of input images [batch, height, width, input_channels]
        num_outputs: number of output channels
        ngf: number of features for the first conv layer
        vscope: variable scope
        reuse_weights: whether to reuse weights (for weight sharing)
    Returns:
        pred: network output at the same spatial resolution as the inputs.
    """
    with tf.compat.v1.variable_scope(vscope, reuse=reuse_weights):
        with slim.arg_scope(
            [slim.conv2d, slim.conv2d_transpose], normalizer_fn=slim.layer_norm):
            cnv1_1 = slim.conv2d(inputs, ngf, [3, 3], scope='conv1_1', stride=1)
            cnv1_2 = slim.conv2d(cnv1_1, ngf * 2, [3, 3], scope='conv1_2', stride=2)

            cnv2_1 = slim.conv2d(cnv1_2, ngf * 2, [3, 3], scope='conv2_1', stride=1)
            cnv2_2 = slim.conv2d(cnv2_1, ngf * 4, [3, 3], scope='conv2_2', stride=2)

            cnv3_1 = slim.conv2d(cnv2_2, ngf * 4, [3, 3], scope='conv3_1', stride=1)
            cnv3_2 = slim.conv2d(cnv3_1, ngf * 4, [3, 3], scope='conv3_2', stride=1)
            cnv3_3 = slim.conv2d(cnv3_2, ngf * 8, [3, 3], scope='conv3_3', stride=2)

            cnv4_1 = slim.conv2d(
                cnv3_3, ngf * 8, [3, 3], scope='conv4_1', stride=1, rate=2)
            cnv4_2 = slim.conv2d(
                cnv4_1, ngf * 8, [3, 3], scope='conv4_2', stride=1, rate=2)
            cnv4_3 = slim.conv2d(
                cnv4_2, ngf * 8, [3, 3], scope='conv4_3', stride=1, rate=2)

            # Adding skips
            skip = tf.concat([cnv4_3, cnv3_3], axis=3)
            cnv5_1 = slim.conv2d_transpose(
                skip, ngf * 4, [4, 4], scope='conv6_1', stride=2)
            cnv5_2 = slim.conv2d(cnv5_1, ngf * 4, [3, 3], scope='conv6_2', stride=1)
            cnv5_3 = slim.conv2d(cnv5_2, ngf * 4, [3, 3], scope='conv6_3', stride=1)

            skip = tf.concat([cnv5_3, cnv2_2], axis=3)
            cnv6_1 = slim.conv2d_transpose(
                skip, ngf * 2, [4, 4], scope='conv7_1', stride=2)
            cnv6_2 = slim.conv2d(cnv6_1, ngf * 2, [3, 3], scope='conv7_2', stride=1)

            skip = tf.concat([cnv6_2, cnv1_2], axis=3)
            cnv7_1 = slim.conv2d_transpose(
                skip, ngf, [4, 4], scope='conv8_1', stride=2)
            cnv7_2 = slim.conv2d(cnv7_1, ngf, [3, 3], scope='conv8_2', stride=1)

            feat = cnv7_2

            pred = slim.conv2d(
                feat,
                num_outputs, [1, 1],
                stride=1,
                activation_fn=tf.nn.tanh,
                normalizer_fn=None,
                scope='color_pred')

            # Add predict depth
            flattened = tf.compat.v1.layers.Flatten()(cnv4_3)
            
            lin5_1 = tf.compat.v1.layers.Dense(1024, activation="relu", name="lin5_1")(flattened)
            lin5_2 = tf.compat.v1.layers.Dense(512, activation="relu", name="lin5_2")(lin5_1)
            lin5_3 = tf.compat.v1.layers.Dense(256, activation="relu", name="lin5_3")(lin5_2)
            lin5_4 = tf.compat.v1.layers.Dense(128, activation="relu", name="lin5_4")(lin5_3)
            lin_final = tf.compat.v1.layers.Dense(num_mpi_planes - 1, activation="sigmoid", name="lin_final")(lin5_4)
        
            return pred, lin_final
