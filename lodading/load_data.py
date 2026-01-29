import  sys
import tensorflow as tf 
from  tensorflow import  keras 
import os

def loader( folder, caption_path, images_folder  ) :  
    print("here")
    captions =  os.path.join(folder,caption_path )
    images_path = os.path.join(folder, images_folder)
    print(images_path)
    images =  tf.keras.utils.image_dataset_from_directory(
        folder,  
        validation_split =0.3,  
        subset ="training", 
        seed =42,
        label_mode = None
        
    )
    try: 
        with open( captions , "r") as f :  
            file = f.read()
    except Exception as e : 
        print("="*50, f"erreur {e} lors de lkecture du  text file", "="*50 ) 

    return file ,  images ,  images.file_paths
    

if __name__ == "__main__" : 
    captions, images = loader ( r"archive",  "captions.txt" , "Images" )