import  tensorflow as tf 
from  tensorflow import keras 
from  tensorflow.keras.models import  Model  
import  yaml
import  sys 
"""
from  CNN_model import CNN_layer
from Embedding import Embedder
"""
try : 
     from utils.Tokenizer import tokenization   
except Exception :  
     sys.path.append("c:\\Users\\papch\\project1")  # à adapter peut-être selon  votre l'utilisateur. 
     from utils.Tokenizer import tokenization  

class Mymodel(Model) :  
    def __init__(self,  cnn_model : Model , LSTM_model : Model , embedder : Model,model_params_path: str="config\config.yml") :  
        super().__init__()
        self.cnn_model = cnn_model 
        
        self.model_params_path=model_params_path
        with open( self.model_params_path, 'r') as f :  
               params = yaml.safe_load(f)
               self.params = params.get("CNN", {})
               self.hidden_dim = int(self.params['hidden_dim'])
        self.embedder = embedder
        self.LSTM_model = LSTM_model
    
    def call(self, images , sequences) : 
        images =  next(iter(images))
        h_t , c_t = self.cnn_model(images)
        text_embedded = self.embedder(sequences)
        x_pred= self.LSTM_model(text_embedded, initial_state= [h_t,c_t])
        return x_pred
        

    
