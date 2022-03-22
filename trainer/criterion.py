import tensorflow as tf

@tf.function
def sparse_categorical_crossentropy(y_pred, y_true):
    mask = tf.not_equal(y_true, 0)

    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)

    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true_masked, y_pred_masked, from_logits=True)
#    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    loss = tf.reduce_mean(loss)
    return loss

@tf.function
def masked_sparse_categorical_crossentropy(y_pred, y_true):
    y_true_masked = tf.boolean_mask(y_true, tf.not_equal(y_true, -1))
    y_pred_masked = tf.boolean_mask(y_pred, tf.not_equal(y_true, -1))
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true_masked, y_pred_masked, from_logits=True)
    loss = tf.math.reduce_mean(loss)
    return loss
