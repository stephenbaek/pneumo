import tensorflow as tf

# def weighted_categorical_crossentropy(weights):
#     # weights = [0.9,0.05,0.04,0.01]
#     def wcce(y_true, y_pred):
#         Kweights = tf.constant(weights, dtype=tf.float32)
#         y_true = tf.cast(y_true, y_pred.dtype)
#         return tf.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True) * tf.reduce_sum(tf.squeeze(tf.one_hot(tf.cast(y_true, tf.int32),2)) * Kweights, axis=-1)
#     return wcce


def dice_loss(y_true, y_pred):
    y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred) + 1

    return 1 - numerator / denominator


def focal_loss(y_true, y_pred, alpha=0.8, gamma=2):
    # flatten label and prediction tensors
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    # first compute binary cross-entropy
    ce = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)
    ce = tf.reduce_mean(ce)
    ce_exp = tf.exp(-ce)
    focal_loss = alpha * (1 - ce_exp) ** gamma * ce

    return focal_loss


def combined_loss(dice_weight, focal_weight, ce_weight):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        ce = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)
        ce = tf.reduce_mean(ce)
        dc = dice_loss(y_true, y_pred)
        fo = focal_loss(y_true, y_pred)

        return dice_weight * dc + focal_weight * fo + ce_weight * ce

    return loss
