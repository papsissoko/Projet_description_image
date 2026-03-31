from  keras_nlp.layers import  SinePositionEncoding
import tensorflow as tf 
from  tensorflow import keras

@tf.keras.utils.register_keras_serializable(package="Custom")
class Positional_enc( tf.keras.layers.Layer) :
    def __init__(self,**kwargs) : 
        super().__init__(**kwargs)
        self.postional_enc = SinePositionEncoding(
        )
    def get_config(self):
        config = super().get_config()
        return config

    def call(self, embedder_output) :  
        return self.postional_enc(embedder_output)