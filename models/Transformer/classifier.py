import tensorflow as tf
from tensorflow import  keras 
from  tensorflow.keras.layers import Dense

@tf.keras.utils.register_keras_serializable(package="Custom")
class Classifier(tf.keras.layers.Layer) : 
    def __init__(self, vocab_size,**kwargs) : 
        super().__init__(**kwargs)
        self.out= Dense(vocab_size+1, activation="softmax")
    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size
        })
        return  config
    def call( self,  inputs) : 
        return self.out(inputs)
    

