import  tensorflow as tf
from  tensorflow import keras 
from  tensorflow.keras.layers import Embedding, Dense
from  tensorflow.keras.preprocessing.text import  Tokenizer

@tf.keras.utils.register_keras_serializable(package="Custom")
class PatchEncoding(tf.keras.layers.Layer) : 
    def __init__(self, nb_patch, proj_dim : int =512,**kwargs) : 
        super().__init__(**kwargs)
        self.proj_dim  = proj_dim
        self.nb_patch = nb_patch
        w_init =tf.random_normal_initializer()
        class_token = w_init( shape=(1,proj_dim),dtype=tf.float32)
        self.w = tf.Variable(initial_value=class_token, trainable=True)
        self.proj = Dense(proj_dim)
        self.tokenizer = Tokenizer(
            num_words = nb_patch, 
        )
        self.Embedder =Embedding(
            input_dim= nb_patch+1,
            output_dim=proj_dim

        )
    def get_config(self):
        config = super().get_config()
        config.update({
            "nb_patch": self.nb_patch,
            "proj_dim": self.proj_dim
        })
        return config


    def call(self, inputs) :  
        b,nb_patch,h_w_c = tf.shape(inputs)[0],tf.shape(inputs)[1],tf.shape(inputs)[2]
        # on  applatit les images 
        inputs = tf.reshape(inputs, [b,-1,h_w_c])

        class_token = tf.tile(self.w, multiples = [b,1]) 
        class_token =  tf.reshape(class_token,  [b,1,self.proj_dim]) # shape = ( b, 1,  proj_dim)
        patchs_embed= self.proj(inputs) # projection  de h*w*c vers proj_dim
        patchs_embed =  tf.concat([class_token, patchs_embed], axis=1)

        # Positional embedding 
        positions = tf.range(start = 0, limit=self.nb_patch+1, delta=1 )
        postion_embed = self.Embedder(positions)
        postion_embed = tf.expand_dims(postion_embed, axis=0)


        enc = patchs_embed+ postion_embed

        return  enc 


if __name__ == "__main__" : 
    import  sys 
    sys.path.append("c:\\Users\\papch\\project1")
    from models.Transformer.patcher_imp import test_embd
    

    
