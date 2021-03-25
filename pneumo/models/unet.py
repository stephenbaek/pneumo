"""
https://www.tensorflow.org/tutorials/images/segmentation
"""

import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix


class UNet(tf.keras.Model):
    def __init__(self):
        super(UNet, self).__init__()
        # self.input_shape = (None, None, 3)
        backbone = tf.keras.applications.MobileNetV2(
            include_top=False,
            input_shape=(None, None, 3),
            weights='imagenet'
        )
        layer_names = [
            'block_1_expand_relu',
            'block_3_expand_relu',
            'block_6_expand_relu',
            'block_13_expand_relu',
            'block_16_project',
        ]
        layers = [backbone.get_layer(name).output for name in layer_names]

        self.down_stack = tf.keras.Model(inputs=backbone.input, outputs=layers)
        self.down_stack.trainable = True
        
        self.up_stack = [
            pix2pix.upsample(512, 3),  # 4x4 -> 8x8
            pix2pix.upsample(256, 3),  # 8x8 -> 16x16
            pix2pix.upsample(128, 3),  # 16x16 -> 32x32
            pix2pix.upsample(64, 3),   # 32x32 -> 64x64
        ]
        self.last = tf.keras.layers.Conv2DTranspose(
            1, 3, strides=2,
            padding='same')  #64x64 -> 128x128



    def call(self, inputs):
        # inputs = tf.keras.layers.Input(shape=input_shape)
        x = inputs

        # Downsampling through the model
        skips = self.down_stack(x)
        x = skips[-1]
        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])

        return self.last(x)
