import  tensorflow as tf 
from  tensorflow import keras 
from  tensorflow.keras.models import  Model , Sequential 
from  tensorflow.keras.layers import Dense
from tensorflow.keras.layers import deserialize as deserialize_layer
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

@tf.keras.utils.register_keras_serializable(package="Custom")
class Mymodel(Model) :  
    def __init__(self,  cnn_model : Model , LSTM_model : Model , embedder : Model,model_params_path: str="config\config.yml", dim_proj :int = 39, **kwargs) :  
        super().__init__(**kwargs)
        self.cnn_model = self._maybe_deserialize(cnn_model) 
        
        self.model_params_path=model_params_path
        with open( self.model_params_path, 'r') as f :  
               params = yaml.safe_load(f)
               self.params = params.get("MyModel", {})
               self.hidden_dim = int(self.params['hidden_dim'])
        self.embedder = self._maybe_deserialize(embedder)
        self.LSTM_model = self._maybe_deserialize(LSTM_model)
        self.dim_proj=dim_proj
        self.classifier= Sequential()
        if dim_proj > 512 : 
             self.classifier.add(Dense(512, activation= "relu"))
             self.classifier.add(Dense(dim_proj, activation = "softmax"))
        else: 
             self.classifier.add(Dense(dim_proj, activation = "softmax"))



    def _maybe_deserialize(self, arg) :  
          if isinstance(arg, dict) :  
               return  deserialize_layer(arg)
          return  arg 

         
    def get_config(self) :                       
         config =  super().get_config()
         config.update({
              "cnn_model" :  self.cnn_model,  
              "LSTM_model": self.LSTM_model,
              "embedder" :  self.embedder, 
              "model_params_path" :  self.model_params_path,
              "dim_proj" :  self.dim_proj
                

         })
         return config 
    def call(self, inputs) : 
        images , sequences= inputs[0], inputs[1]
        images =  images
        h_t , c_t = self.cnn_model(images)
        text_embedded = self.embedder(sequences)
        x_pred= self.LSTM_model(text_embedded, initial_state= [h_t,c_t])
        x_pred = self.classifier(x_pred)
        return x_pred
        

    
