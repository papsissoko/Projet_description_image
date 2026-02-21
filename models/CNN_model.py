import tensorflow as tf 
from  tensorflow import  keras 
from  tensorflow.keras.models import Sequential  
from  tensorflow.keras.layers import  Conv2D, BatchNormalization, MaxPooling2D , Dense , Flatten , Input
import  yaml

@tf.keras.utils.register_keras_serializable(package="Custom")
class CNN_layer(tf.keras.Model) :  
    def __init__(self,model_params_path :  str = "config/config.yml", **kwargs) :  
        super().__init__(**kwargs)
        self.model_params_path = model_params_path
        self.cnn_model = Sequential()
        with open( self.model_params_path, 'r') as f :  
               params = yaml.safe_load(f)
               self.parals_cnn = params.get("CNN", {})
               self.params = params.get("MyModel", {})

        nb_filters = self.parals_cnn['nb_filters'].split() 
        sizes = self.parals_cnn['filters_size'].split() 
        assert(len(nb_filters)==len(sizes)),"dans le fichier yaml  les deux doivent  avoir le même nombre d'élément"
        self.cnn_model.add(Input(shape= (256,256,3)))
        for i in range( len(nb_filters)) :  
            layer_cnn= Conv2D(int(nb_filters[i]) , int(sizes[i]) ,  padding = "valid")
            pool_layer = MaxPooling2D()
            batch_norm = BatchNormalization()
            self.cnn_model.add(layer_cnn)
            self.cnn_model.add(pool_layer)
            self.cnn_model.add(batch_norm)
        self.cnn_model.add(Flatten())
        self.cnn_model.add(Dense(int(self.params['hidden_dim'])*3))
        self.cnn_model.add(Dense(int(self.params['hidden_dim'])*2))
        self.h_t = Dense(int(self.params['hidden_dim']))
        self.c_t = Dense(int(self.params['hidden_dim']))
    

    def get_config(self) :  
         config = super().get_config()
         config.update({
              "model_params_path" :  self.model_params_path,  
              }
         )
         return  config 
    def call( self , x) :  
        x_feature_map = self.cnn_model(x)
        print("=="*50,x_feature_map )
        h_t = self.h_t(x_feature_map)
        c_t = self.c_t(x_feature_map)
        return h_t ,  c_t


if __name__ == "__main__" :  
     classs = CNN_layer()
     model = classs.construct_model()
     model.summary()