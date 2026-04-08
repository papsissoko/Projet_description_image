from  keras_nlp.layers import  SinePositionEncoding
import tensorflow as tf 
from  tensorflow import keras

@tf.keras.utils.register_keras_serializable(package="Custom")
class Positional_enc( tf.keras.layers.Layer) :
    def __init__(self,**kwargs) : 
        super().__init__(**kwargs)
        self.postional_enc = SinePositionEncoding(
        )
        self.supports_masking = True  # corrige 
        """
        was passed an input with a mask attached to it. However, this layer does not support masking 
        and will therefore destroy the mask information. Downstream layers will not see the mask.

        """
    def get_config(self):
        config = super().get_config()
        return config
    
    def compute_mask(self, inputs, mask=None) :
        return mask
    """
    def call(self, embedder_output, mask=None) :  
        return self.postional_enc(embedder_output)
    """

    def call(self, embedder_output, mask=None):
        seq_len = tf.shape(embedder_output)[1]
        d_model = tf.shape(embedder_output)[2]

        positions = tf.cast(tf.range(seq_len)[:, tf.newaxis], tf.float32) 
        dims = tf.cast(tf.range(d_model)[tf.newaxis, :], tf.float32)      

        angles = positions / tf.pow(
        10000.0, (2 * (dims // 2)) / tf.cast(d_model, tf.float32)
          ) 

        sin_enc = tf.sin(angles[:, 0::2])  
        cos_enc = tf.cos(angles[:, 1::2])  

        pe = tf.reshape(
         tf.stack([sin_enc, cos_enc], axis=-1),
         [seq_len, -1]
        )  

        pe = pe[tf.newaxis, :, :] 

        return embedder_output + pe



