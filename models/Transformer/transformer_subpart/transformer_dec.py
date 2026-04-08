import sys
import os
sys.path.append(r"C:\Users\papch\project1")
from  Preprocessing.Images import  preprocesser
import numpy as np 
import tensorflow as tf 
from  tensorflow.keras.layers  import  Embedding, Dense, MultiHeadAttention, LayerNormalization
from  keras_nlp.layers import  SinePositionEncoding



from utils.load_data import  loader
from  utils.Tokenizer import  tokenization
"""
captions,captions_val,  images,  val_images  = loader ( r"archive",  "captions.txt" , "Images" )


tokens, word_index, max_len = tokenization(caption=captions)
tokens_val, word_index_val, max_len_val = tokenization(caption=captions_val)
vocab_size = len(word_index) + 1
"""

@tf.keras.utils.register_keras_serializable(package="Custom")
class Trasformer_dec (tf.keras.Model) : 
    def __init__(self,num_heads :  int=8, out_proj_dim : int = 512,**kwargs ) : 
        super().__init__(**kwargs) 
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=out_proj_dim)
        self.feed_forward =  Dense(out_proj_dim, activation="relu")
        self.norm = LayerNormalization(epsilon=1e-4)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "out_proj_dim": self.out_proj_dim
            
        })
        return config

        
    def  call(self,  tokens_embed_pos:  tf.Tensor, attention_mask) :
        x = self.norm(tokens_embed_pos) 
        x_att = self.attention(x,x, attention_mask=attention_mask)
        x_att_norm= self.norm(x_att+x)
        x_feed =  self.feed_forward(x_att_norm)
        return  x_feed






