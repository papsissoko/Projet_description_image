import sys
sys.path.append("c:\\Users\\papch\\project1")
from  Preprocessing.Images import  preprocesser
import numpy as np 
import tensorflow as tf 
from tensorflow.keras.optimizers import Adam


from loading.load_data import  loader
from  utils.Tokenizer import  tokenization
from models.my_model import Mymodel
from  models.CNN_model import CNN_layer
from models.Embedding import Embedder
from models.LSTM_model import LSTM_model
import yaml
import pickle
from  PIL import  Image
# Initialisation  des classes  

preprocesser =preprocesser()
lstm_model = LSTM_model()
cnn_model = CNN_layer()

with open("config\config.yml", 'r') as f :  
               params = yaml.safe_load(f)
               params = params.get("MyModel", {})
               hidden_dim = params["hidden_dim"]
               lr = params["learning_rate"]
               batch_size= params["batch_size"]
captions,captions_val,  images = loader ( r"archive",  "captions.txt" , "Images" )

images_preprocessed = images.map(preprocesser.call, num_parallel_calls=tf.data.AUTOTUNE)  # car images est  prefetch  non  tensor fixe . 
tokens, word_index, max_len = tokenization(caption=captions)

vocab_size = len(word_index) + 1
embedder =  Embedder(input_dim=vocab_size, out_dim=hidden_dim )

my_model = Mymodel(cnn_model=cnn_model,LSTM_model=lstm_model, embedder=embedder, dim_proj=vocab_size)


optim = Adam(learning_rate = float(lr) )

loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction = 'none')
my_model.compile(
        optimizer = optim, 
        loss = tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
)

images_array = np.concatenate([x for x in images_preprocessed], axis=0)

text_in_ds = tokens[:len(images_array), :-1]
text_out_ds = tokens[: len(images_array),1:]


if __name__ == "__main__":
    with open("tokenizer_data.pkl", "wb") as f:
        pickle.dump({"word_index": word_index, "max_len": max_len, "tokens_padded": tokens}, f)
    print("Données du tokenizer sauvegardées dans tokenizer_data.pkl")

    my_model.fit(x=[images_array,text_in_ds],y=text_out_ds, epochs=1,batch_size=batch_size)
    my_model.save("mon_modele.keras")
    print("Modèle sauvegardé.")