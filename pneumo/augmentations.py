# Copyright 2021 Stephen Baek. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utility functions for data sets
"""

import tensorflow as tf


def random_crop(size):
    def func(image, mask):
        shape = tf.shape(image, out_type=tf.int64)
        crop_size = tf.cast(tf.cast(shape, tf.float32) * 0.9, tf.int64)
        ymin = tf.random.uniform([], 0, shape[0] - crop_size[1], dtype=tf.int64)
        xmin = tf.random.uniform([], 0, shape[1] - crop_size[0], dtype=tf.int64)

        image_sliced = tf.slice(image, [ymin, xmin, 0], [crop_size[1], crop_size[0], 3])
        mask_sliced = tf.slice(mask, [ymin, xmin, 0], [crop_size[1], crop_size[0], 1])

        return tf.image.resize(image_sliced, size), tf.image.resize(mask_sliced, size)

    return func


def random_intensity(stddev):
    def func(image, mask):
        noise = tf.random.normal([], stddev=stddev, dtype=tf.float32)
        return tf.clip_by_value(image + noise, 0, 255), mask

    return func


def random_noise(stddev):
    def func(image, mask):
        noise = tf.random.normal(image.shape, stddev=stddev, dtype=tf.float32)
        return tf.clip_by_value(image + noise, 0, 255), mask

    return func
