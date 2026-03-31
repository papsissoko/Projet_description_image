import  os 
#from loading.load_data import  loader

import tensorflow as tf 
from  tensorflow import  keras 
from  tensorflow.keras.models import  Sequential
from tensorflow.keras.layers import Resizing, Rescaling , InputLayer
from  tensorflow.keras import Model
import sys

class preprocesser(Model) :  
    def __init__(self,target_size : tuple[int,int] = (256,256) , batch_size : int=32 , is_rgb : bool = True ):
        super().__init__()
        self.target_size =  target_size
        self.pipeline = Sequential()
        self.pipeline.add(InputLayer(input_shape=(None,None,3)))
        self.resizer =Resizing(
            height =self.target_size[0], 
            width = self.target_size[1]
        )
        self.pipeline.add(self.resizer)
        if  is_rgb : 
            self.rescaler = Rescaling(scale = 1.0/255.0) 
            self.pipeline.add(self.rescaler)
    @ tf.function
    def  call(self,x) :
        x=  tf.cast(x, tf.float32)

        if tf.rank(x)==3: 
            x = tf.expand_dims(x, axis=0)
        x = tf.ensure_shape(x, [None,None, None, 3])
        
        return  self.pipeline(x)  



"""

if __name__ == "__main__" : 
    _ , images,paths= loader ( r"archive",  "captions.txt" , "Images" )
    image_preprocesser =preprocesser()
    image_0 = next(iter(images))
    image_preprocessed = image_preprocesser(image_0)    
    image_preprocessed_0 = image_preprocessed[0]
    plt.imshow(image_preprocessed_0.numpy())
    plt.show()
"""