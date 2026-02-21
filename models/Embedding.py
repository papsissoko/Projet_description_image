import  tensorflow as tf 
from  tensorflow import  keras 
from  tensorflow.keras.layers import  Embedding 
from  tensorflow.keras.models import  Model  

@tf.keras.utils.register_keras_serializable(package="Custom")
class Embedder( Model) :  
    def __init__(self ,  input_dim : int,  out_dim :  int ,  **kwargs ) :  
        super().__init__(**kwargs) 
        self.input_dim=input_dim
        self.out_dim=out_dim
        self.embedder = Embedding(
            input_dim = input_dim ,  
            output_dim = out_dim ,
            mask_zero= True
        )
    def get_config(self) :  
        config= super().get_config()
        config.update({
            "input_dim" : self.input_dim,  
            "out_dim" : self.out_dim

        })
        return config 
    def call(self,  x) :  
        return self.embedder(x)
    