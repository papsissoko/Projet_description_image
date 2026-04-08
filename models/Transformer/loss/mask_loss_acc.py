import tensorflow as tf 



def  mask_loss(y_true, y_pred) :  
    mask =  tf.math.logical_not( tf.math.equal(y_true,0))
    loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction="none")(y_true, y_pred)
    loss *= tf.cast(mask, dtype=loss.dtype)
    return tf.reduce_mean(loss)


def mask_accuracy(y_true, y_pred) :  
    mask =  tf.math.logical_not( tf.math.equal(y_true,0))
    accuracy = tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
    accuracy *= tf.cast(mask, dtype=accuracy.dtype)
    return tf.reduce_mean(accuracy)