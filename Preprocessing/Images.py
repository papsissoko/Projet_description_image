import  os 
from lodading.load_data import  loader
import matplotlib.pyplot as plt

import tensorflow as tf 
from  tensorflow import  keras 
from  tensorflow.keras.models import  Sequential
from tensorflow.keras.layers import Resizing, Rescaling 
from  tensorflow.keras import Model
import matplotlib.pyplot as plt 
import sys


class preprocesser(Model) :  
    def __init__(self,target_size : tuple = (256,256), batch_size : int=32 , is_rgb : bool = True ):
        super().__init__()
        self.target_size =  target_size
        self.pipeline = Sequential()
        self.resizer =Resizing(
            self.target_size[0], 
            self.target_size[1]
        )
        self.pipeline.add(self.resizer)
        if  is_rgb : 
            self.rescaler = Rescaling(scale = 1.0/255.0) 
            self.pipeline.add(self.rescaler)
    def  call(self,x) :
        x=  tf.cast(x, tf.float32)
        return  self.pipeline(x)  




if __name__ == "__main__" : 
    print("there")
    _ , images,paths= loader ( r"archive",  "captions.txt" , "Images" )
    image_preprocesser =preprocesser()
    image_0 = next(iter(images))
    image_preprocessed = image_preprocesser(image_0)    
    image_preprocessed_0 = image_preprocessed[0]
    print(f"l'image est de shape {image_preprocessed_0.shape} ,  et  de min  :  {tf.reduce_min(image_preprocessed_0)} ,  et de max :  {tf.reduce_max(image_preprocessed_0)}, et de moy {tf.reduce_mean(image_preprocessed_0)}")
    plt.imshow(image_preprocessed_0.numpy())
    plt.show()
