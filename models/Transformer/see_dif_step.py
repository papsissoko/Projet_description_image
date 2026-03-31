import  sys
sys.path.append("c:\\Users\\papch\\project1")
import  tensorflow as tf 

from models.Transformer.patcher_imp import test_embd, plot_patch,patchEmbedd
from models.Transformer.patch_encod import PatchEncoding
from models.Transformer.transformer_encoder import TransformerEncoderBlock
from utils.Tokenizer import tokenization
from  loading.load_data import loader
from  Preprocessing.Images import  preprocesser
from models.Transformer.VIT import VIT

preprocesser =preprocesser()


captions,captions_val,  images,  val_images  = loader ( r"archive",  "captions.txt" , "Images" )


tokens, word_index, max_len = tokenization(caption=captions)
tokens_val, word_index_val, max_len_val = tokenization(caption=captions_val)
vocab_size = len(word_index) + 1
images_preprocessed = images.map(preprocesser.call, num_parallel_calls=tf.data.AUTOTUNE)  # car images est  prefetch  non  tensor fixe



vit =VIT(vocab_size,8)

for im in images_preprocessed.take(1):
    im = tf.convert_to_tensor(im)
    batch_size = tf.shape(im)[0]
    tokens = tokens[:batch_size]
    
    pred = vit(im,tokens)
    break

# Reste plus quà  faire l'entraînement du  modèle.


