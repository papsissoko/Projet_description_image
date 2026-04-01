import  sys
import  pickle 
import os 
sys.path.append("c:\\Users\\papch\\project1")

from models.Transformer.patcher_imp import test_embd, plot_patch,patchEmbedd
from models.Transformer.patch_encod import PatchEncoding
from models.Transformer.transformer_encoder import TransformerEncoderBlock
from utils.Tokenizer import tokenization
from  utils.load_data import loader
from  Preprocessing.Images import  preprocesser
from models.Transformer.VIT import VIT
import  tensorflow as tf 

preprocesser =preprocesser()


captions,captions_val,  images,  val_images  = loader ( r"archive",  "captions.txt" , "Images" )


data = pickle.load(open("tokens/tokenizer_data_train.pkl", "rb"))
word_index, max_len,tokens = data["word_index"], data["max_len"], data["tokens_padded"]
vocab_size = len(word_index) + 1
images_preprocessed = images.map(preprocesser.call, num_parallel_calls=tf.data.AUTOTUNE)  # car images est  prefetch  non  tensor fixe


"""
vit =VIT(vocab_size,8)

for im in images_preprocessed.take(1):
    im = tf.convert_to_tensor(im)
    batch_size = tf.shape(im)[0]
    tokens = tokens[:batch_size]
    
    pred = vit(im,tokens)
    break

# Reste plus quà  faire l'entraînement du  modèle.
"""



vit =VIT(vocab_size,0)
#images_preprocessed = images_preprocessed.unbatch()
images_preprocessed=images_preprocessed.unbatch()
tokens_dataset = tf.data.Dataset.from_tensor_slices(tokens)
dataset_train = tf.data.Dataset.zip((images_preprocessed, tokens_dataset))
dataset_train = dataset_train.map(
    lambda img, tok: ((img, tok[:-1]), tok[1:])  # ← ce map est là ?
)
dataset_train = dataset_train.batch(32).prefetch(tf.data.AUTOTUNE)

with tf.device('/GPU:0') :
    vit.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]) 
    vit.fit(dataset_train,epochs=1)#, batch_size=32)

vit.save("transformer.keras")