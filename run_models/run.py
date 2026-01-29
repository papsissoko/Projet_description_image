import sys
sys.path.append("c:\\Users\\papch\\project1")
from  Preprocessing.Images import  preprocesser
from  Preprocessing.Text  import  Textpreprocesseur


from lodading.load_data import  loader
from  utils.Tokenizer import  tokenization
from models.my_model import Mymodel
from  models.CNN_model import CNN_layer
from models.Embedding import Embedder
from models.LSTM_model import LSTM_model
import yaml

# Initialisation  des classes  
preprocesser =preprocesser()
Textpreprocesseur= Textpreprocesseur()
lstm_model = LSTM_model()
cnn_model = CNN_layer()

with open("config\config.yml", 'r') as f :  
               params = yaml.safe_load(f)
               params = params.get("CNN", {})
               hidden_dim = params["hidden_dim"]
captions, images , paths = loader ( r"archive",  "captions.txt" , "Images" )

images_preprocessed = images.map(lambda x: preprocesser(x))  # car images est  prefetch  non  tensor fixe . 
image_ref, caption,unique_words,unique_words_len  = Textpreprocesseur.preprocesser(captions)
tokens = tokenization(caption=caption)
embedder =  Embedder(input_dim=unique_words_len,out_dim=hidden_dim )

# Tests du  modèle .
my_model = Mymodel(cnn_model=cnn_model,LSTM_model=lstm_model, embedder=embedder)
sortie=my_model.call(images_preprocessed , tokens)

