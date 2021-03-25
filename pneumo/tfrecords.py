import tensorflow as tf


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(image, mask):
    """Serialize image and mask data to create tfrecord
    """
    if image.ndim == 2:
        image = tf.expand_dims(image, -1)

    image = tf.image.encode_png(image)
    mask = tf.expand_dims(mask, -1)
    mask = tf.image.encode_png(mask)
    feature = {
        'image': _bytes_feature(image),
        'mask': _bytes_feature(mask),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def decode_example(example):
    """Decode a serialized example.
    """
    example = tf.io.parse_single_example(example, {
        'image': tf.io.FixedLenFeature([], tf.string),
        'mask': tf.io.FixedLenFeature([], tf.string),
    })
    image = tf.image.decode_png(example['image'])
    image = tf.image.grayscale_to_rgb(image)
    mask = tf.image.decode_png(example['mask'])
    return tf.cast(image, tf.float32), tf.cast(mask, tf.float32)/255.
