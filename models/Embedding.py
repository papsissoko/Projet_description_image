import  tensorflow as tf 
from  tensorflow import  keras 
from  tensorflow.keras.layers import  Embedding 
from  tensorflow.keras.models import  Model  


class Embedder( Model) :  
    def __init__(self ,  input_dim : int,  out_dim :  int  ) :  
        super().__init__() 
        self.embedder = Embedding(
            input_dim = input_dim ,  
            output_dim = out_dim 
        )
    def call(self,  x) :  
        return self.embedder(x)
    