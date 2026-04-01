import tensorflow as tf 
import  matplotlib.pyplot as plt 

from  PIL import Image
import numpy as np 
import pickle
import sys 
import traceback


from  Preprocessing.Images import  preprocesser
from models.Transformer.transformer_subpart.patcher_imp import test_embd, plot_patch,patchEmbedd
from models.Transformer.transformer_subpart.patch_encod import PatchEncoding
from models.Transformer.transformer_subpart.transformer_encoder import TransformerEncoderBlock
#from models.Transformer.classifier import Classifier
from models.Transformer.transformer_subpart.transformer_dec import Trasformer_dec
from models.Transformer.transformer_subpart.Positional_enc import Positional_enc
from models.CCN_LSTM.Embedding import Embedder
from models.Transformer.VIT import VIT




my_model = tf.keras.models.load_model("transformer.keras", compile=False)

im = tf.convert_to_tensor(Image.open(r"archive\Images\2599903773_0f724d8f63.jpg"))
im_resized =tf.keras.layers.Resizing(256,256)(im)
im_= tf.keras.layers.Normalization(mean=0.5, variance=0.5)(im_resized)
im=tf.expand_dims(im_, axis=0)
token = ["<start>"]

data = pickle.load(open("tokens/tokenizer_data_train.pkl", "rb"))
word_index, max_len,tokens = data["word_index"], data["max_len"], data["tokens_padded"]
vocab_size = len(word_index) + 1


inv_dic = {v: k for k, v in word_index.items()} # on veut {token: mot} pour pouvoir faire un get  dessus


image_test =tf.convert_to_tensor(Image.open(r"archive\Images\2599903773_0f724d8f63.jpg"))
image_test = tf.expand_dims(image_test, axis=0)

print(inv_dic)
image_processed = preprocesser()(image_test)

text_input = ["<start>"] # On commence avec la balise de début
final_caption = []
print( "génération de la description en cours ...")

for i in range(max_len - 1): 
    print(i)
    seq =[word_index.get(word,0) for word in text_input]
    seq_padded = tf.keras.preprocessing.sequence.pad_sequences([seq], maxlen=max_len - 1, padding='post')
    prediction = my_model.predict([image_processed, seq_padded], verbose=0)
    mot_predit = np.argmax(prediction[0,i,:])
    mot = inv_dic.get(mot_predit,"")
    print(mot)
    if mot == "<end>" or mot == "": 
        break
    text_input.append(mot)
    print(mot)
    final_caption.append(mot)

print("Description générée : " + " ".join(final_caption))



"""
image_test =tf.convert_to_tensor(Image.open(r"archive\Images\2599903773_0f724d8f63.jpg"))
image_processed = preprocesser(image_test)

text_input = ["<start>"] # On commence avec la balise de début
final_caption = []

pred = my_model.predict(image_processed)    
print(pred)

"""