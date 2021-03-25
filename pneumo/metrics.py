import tensorflow as tf


def dice(y_true, y_pred):
    y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred) + 1
    return numerator / denominator
