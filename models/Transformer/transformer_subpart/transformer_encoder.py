import  numpy  as np  
import  matplotlib.pyplot as plt 
import tensorflow as tf 
from  tensorflow import keras 
from  tensorflow.keras.layers import  Dense, MultiHeadAttention, LayerNormalization, Layer


@tf.keras.utils.register_keras_serializable(package="Custom")
class TransformerEncoderBlock(tf.keras.layers.Layer ) : 
    def __init__(self,num_heads : int =8,embed_dim : int =512,**kwargs) :
        super().__init__(**kwargs)
        
        self.attention_layer = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.mlp1 = Dense(embed_dim*2, activation="relu")
        self.mlp2 = Dense(embed_dim, activation="relu")
        self.Norm_layer = LayerNormalization()
    
    def build(self, input_shape):
        return super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "embed_dim": self.embed_dim
        })
        return config



    def call(self, inputs):
        x1 =  self.attention_layer(inputs, inputs)
        x2 = self.Norm_layer(x1+ inputs)
        x3 = self.mlp2(x2)
        x4 =self.Norm_layer(x2+x3)
        return x4 

