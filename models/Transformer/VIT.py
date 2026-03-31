import tensorflow as tf
from  tensorflow import keras
from tensorflow.keras.layers import  MultiHeadAttention,Dense
import  sys 
sys.path.append("c:\\Users\\papch\\project1")
from models.Transformer.patcher_imp import test_embd, plot_patch,patchEmbedd
from models.Transformer.patch_encod import PatchEncoding
from models.Transformer.transformer_encoder import TransformerEncoderBlock
#from models.Transformer.classifier import Classifier
from models.Transformer.transformer_dec import Trasformer_dec
from models.Transformer.Positional_enc import Positional_enc
from models.CCN_LSTM.Embedding import Embedder


@tf.keras.utils.register_keras_serializable(package="Custom")
class VIT(tf.keras.Model) :  
    def __init__(self,vocab_size :  int, nb_block,  proj_dim :  int =512, patch_size: int =32,image_size: int =256,**kwargs):
        super().__init__(**kwargs)
        self.nb_block  =  nb_block
        self.proj_dim = proj_dim
        self.patch_size = patch_size
        self.image_size = image_size
        self.n_h, self.n_w = image_size//patch_size, image_size//patch_size
        self.nb_patch = self.n_h*self.n_w

        self.Embedder = Embedder(vocab_size,proj_dim)
        self.vocab_size = vocab_size
        self.patch_size= patch_size
        self.cross_multi_head = MultiHeadAttention(num_heads=8, key_dim=proj_dim)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-4)
        self.encodeur = TransformerEncoderBlock(embed_dim=proj_dim)
        self.decodeur = Trasformer_dec()
        self.patch_enc = PatchEncoding(nb_patch=self.nb_patch, proj_dim=proj_dim)
        self.positional_enc = Positional_enc()

        self.patcher = patchEmbedd(patch_size)
        self.out1 = Dense(proj_dim, activation="relu")
        self.out2 = Dense(vocab_size, activation="softmax")


    def get_config(self):
        config = super().get_config()
        config.update({
            "nb_block": self.nb_block,
            "proj_dim": self.proj_dim,
            "patch_size": self.patch_size,
            "image_size": self.image_size,
            "vocab_size": self.vocab_size
        })
        return config




    def  call(self, inputs  ):
        inputs_im, tokens= inputs
        tokens_emb = self.Embedder(tokens)
        tokens_emd_pos = self.positional_enc(tokens_emb)
        tokens_pos_emb =tokens_emb+ tokens_emd_pos
        patches, _,_= self.patcher(inputs_im)
        patches_encoded =self.patch_enc(patches)
        # premier passage dans le transformeur
        patches_attention_score = self.encodeur(patches_encoded)
        text_attention = self.decodeur(tokens_pos_emb)
        cross_attention = self.cross_multi_head(text_attention, patches_attention_score)
        cross_attention_norm = self.norm(cross_attention+text_attention)

        # Passage dans les autres blocks construits par la boucle ci-dessous

        for i in  range(self.nb_block) : 
            patches_attention_score = self.encodeur(patches_encoded)
            text_attention = self.decodeur(cross_attention_norm)
            cross_attention = self.cross_multi_head(text_attention, patches_attention_score)
            cross_attention_norm = self.norm(cross_attention+text_attention)

                     
        out1= self.out1(cross_attention_norm)
        out2 = self.out2(out1)

        return out2

        


if __name__ == "__main__" : 
    vit =VIT(512,8,)
    vit.summary()

        
        


        
