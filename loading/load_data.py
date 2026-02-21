import  sys
import tensorflow as tf 
import os
import yaml


with open( "config/config.yml", 'r') as f :  
               params = yaml.safe_load(f)
               params = params.get("MyModel", {})
               batch_size =params['batch_size']

def loader( folder, caption_path, images_folder  ) :  
    captions =  os.path.join(folder,caption_path )
    images_path = os.path.join(folder, images_folder)
    images =  tf.keras.utils.image_dataset_from_directory(
        folder,  
        validation_split =None,  
        #batch_size=batch_size,
        #subset ="training", 
        seed =42,
        label_mode = None,
        shuffle= False  # il faut absolument le mettre à false car images.file_paths n'est pas mélangé. 
        
    )

    val_images =  tf.keras.utils.image_dataset_from_directory(
        folder,  
        validation_split =0.3,  
        batch_size=batch_size,
        subset ="validation", 
        seed =42,
        label_mode = None,
        shuffle= False  # il faut absolument le mettre à false car images.file_paths n'est pas mélangé. 
        
    )
    try: 
        with open( captions , "r") as f :  
            file = f.read()
    except Exception as e : 
        print("="*50, f"erreur {e} lors de lkecture du  text file", "="*50 ) 
    
    paths_ = images.file_paths
    val_paths = val_images.file_paths
    paths=  [path.split("\\")[-1] for path  in  paths_]
    val_paths=  [path.split("\\")[-1] for path  in  val_paths]


    captions_list = file.strip().split('\n')[1:] 
    captions_paths = [l.split(',', 1)[0] for l in captions_list]
    caption   = [l.split(',', 1)[1] for l in captions_list]
    
    captions_dict = {x:y for x,y in zip(captions_paths,caption)}


    caption_ordered = [captions_dict[x] for x in paths]
    caption_val = [captions_dict[x] for x in val_paths]


    return caption_ordered,caption_val ,  images , paths_
    

if __name__ == "__main__" : 
    captions, images = loader ( r"archive",  "captions.txt" , "Images" )