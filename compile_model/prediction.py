import  tensorflow as tf 
import os
import sys 
from PIL import  Image
import numpy as np 
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(r"C:\Users\papch\project1")
if not project_root in  sys.path :  
    sys.path.append(project_root)
from  Preprocessing.Images import  preprocesser

# Import des classes nécessaires pour le chargement du modèle
from models.my_model import Mymodel
from models.CNN_model import CNN_layer
from models.Embedding import Embedder
from models.LSTM_model import LSTM_model

preprocesser =preprocesser()
path = "mon_modele_sauvegarde"


try : 
    with open("tokenizer_data.pkl", "rb" ) as f:
        data= pickle.load(f)
        word_index = data["word_index"]
        max_len = data["max_len"]
        keys= list(word_index.keys())
        values= list(word_index.values())
        inv_dic = {v: k for k, v in word_index.items()} # on veut {token: mot} pour pouvoir faire un get  dessus

except FileNotFoundError : 
    sys.exit(1)



my_model = tf.keras.models.load_model("mon_modele.keras", compile=False)

# Chargement et prétraitement de l'image
image_test =np.array(Image.open(r"archive\Images\2599903773_0f724d8f63.jpg"))
image_processed = preprocesser(image_test)

text_input = ["<start>"] # On commence avec la balise de début
final_caption = []
print( "génération de la description en cours ...")

for i in range(max_len - 1): 
    seq =[word_index.get(word,0) for word in text_input]
    seq_padded = pad_sequences([seq], maxlen=max_len - 1, padding='post')
    prediction = my_model.predict([image_processed, seq_padded], verbose=0)
    mot_predit = np.argmax(prediction[0,i,:])
    mot = inv_dic.get(mot_predit,"")
    if mot == "<end>" or mot == "": 
        break
    text_input.append(mot)
    print(mot)
    final_caption.append(mot)

print("Description générée : " + " ".join(final_caption))
