import  sys
import tensorflow as tf 
import os
import yaml
import  numpy as np  

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
config_path = os.path.join(project_root, "config", "config.yml")

with open( config_path, 'r') as f :  
               params = yaml.safe_load(f)
               params = params.get("MyModel", {})
               batch_size =params['batch_size']

def loader( folder, caption_path, images_folder, val_split : int = 0.2  ) :  
    captions =  os.path.join(folder,caption_path )
    images_path = os.path.join(folder, images_folder)
    train_images =  tf.keras.utils.image_dataset_from_directory(
        folder,  
        validation_split =val_split,  
        batch_size=batch_size,
        subset ="training", 
        seed =42,
        label_mode = None,
        shuffle= False  
        
    )

    val_images =  tf.keras.utils.image_dataset_from_directory(
        folder,  
        validation_split =val_split,  
        batch_size=batch_size,
        subset ="validation", 
        seed =42,
        label_mode = None,
        shuffle= False 
        
    )
    try: 
        with open( captions , "r") as f :  
            file = f.read()
    except Exception as e : 
        print("="*50, f"erreur {e} lors de lkecture du  text file", "="*50 ) 
    
    paths_ = train_images.file_paths
    val_paths = val_images.file_paths
    paths=  [os.path.basename(path) for path  in  paths_]
    val_paths=  [os.path.basename(path) for path  in  val_paths]


    captions_list = file.strip().split('\n')[1:] 

    caption = []
    captions_paths = []
    i=0
    for cap  in  captions_list : 
          try :
                parts = cap.split(',',1)
                caption.append(parts[1])
                captions_paths.append(parts[0])
          except : 
               i+=1
    assert(len(caption)== len(captions_paths)), " Desynchronisation  entre la taille de l'image et de ses captions"
    
    if i>0: 
          print( f"{i} images ont été ignorées car les données sont mal formées")
          
    
    captions_dict = {x:y for x,y in zip(captions_paths,caption)}

    
    caption_train = [captions_dict[x] for x in captions_paths]
    #caption_val = [captions_dict[x] for x in val_paths]
    caption_val = []


    return caption_train,caption_val ,  train_images  , val_images
    

if __name__ == "__main__" : 
   import  matplotlib.pyplot as plt 
   y_train, y_val,  x_train,  x_val= loader ( r"archive",  "All_captions.txt" , "Images" )
   plt.imshow(next(iter(x_train))[0])
   plt.show()