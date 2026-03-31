"""
im_patch,n_h,n_w = test_embd()
decision = input("do you want to plot an example of patch (y/n)")
if decision =="y" : 
    plot_patch(im_patch,n_h,n_w)
else :
    pass

embedd = PatchEncoding(nb_patch=n_h*n_w, proj_dim=512)
im_patch = tf.convert_to_tensor(im_patch)
im_patch = tf.expand_dims(im_patch, axis=0)
patch_emm = embedd(im_patch)
transformer_enc = TransformerEncoderBlock()
attention_out = transformer_enc(patch_emm)

import sys
import os
sys.path.append(r"C:\Users\papch\project1")

from loading.load_data import  loader
from  utils.Tokenizer import  tokenization
from models.CCN_LSTM.Embedding import Embedder
from  models.Transformer.Positional_enc import Positional_enc
from models.Transformer.transformer_dec import Trasformer_dec

captions,captions_val,  images,  val_images  = loader ( r"archive",  "captions.txt" , "Images" )


tokens, word_index, max_len = tokenization(caption=captions)
#tokens_val, word_index_val, max_len_val = tokenization(caption=captions_val)
vocab_size = len(word_index) + 1

tokens_embedd = Embedder(vocab_size,512)(tokens)
positions = Positional_enc()(tokens_embedd)
transf_dec = Trasformer_dec()
emb_pos=tokens_embedd+ positions
att = transf_dec(emb_pos)
"""