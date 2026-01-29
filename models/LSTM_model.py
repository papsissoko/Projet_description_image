import  tensorflow as tf 
from tensorflow import  keras 
from tensorflow.keras.models import Sequential , Model
from  tensorflow.keras.layers import LSTM, Input
from utils.Tokenizer import tokenization
import  yaml  

class LSTM_model(Model) : 
    def __init__(self ,  model_params_path :  str ="config\config.yml") : 
        super().__init__()
        #self.sequences = tokenization()
        self.LSTM_model = Sequential()

        with open( model_params_path, 'r') as f :  
               params = yaml.safe_load(f)
               self.params = params.get("CNN", {})
        #self.nb_units = self.params['nb_filters'].split() 
        print(type(self.params['hidden_dim']))
        self.hidden_dim = self.params['hidden_dim']
        self.LSTM_model.add(LSTM(self.hidden_dim,return_sequences=False))
       
    def call( self,x, initial_state:None) :  
         return self.LSTM_model(x,initial_state=initial_state)
           
        

        
    
