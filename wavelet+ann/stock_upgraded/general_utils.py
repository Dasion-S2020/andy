import tensorflow as tf
#from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.keras.layers import BatchNormalization, Flatten
#from tensorflow.contrib.framework import arg_scope

def batch_normalization(x, scope, training=True):
    return tf.cond(
        pred=training,
        true_fn = lambda: BatchNormalization(center=True, scale=True, zero_debias_moving_mean=True, decay=0.9, updates_collections=None, scope=scope, inputs=x, is_training=training, reuse=None),
        false_fn = lambda: BatchNormalization(center=True, scale=True, zero_debias_moving_mean=True, decay=0.9, updates_collections=None, scope=scope, inputs=x, is_training=training, reuse=True)
    )


'''def batch_normalization(x, scope, training=True):
    with arg_scope([BatchNormalization],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        return tf.cond(
            pred=training,
            true_fn=lambda: BatchNormalization(inputs=x, is_training=training, reuse=None),
            false_fn=lambda: BatchNormalization(inputs=x, is_training=training, reuse=True)
        )
'''

