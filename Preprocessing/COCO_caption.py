import json
import csv
import os

def coco_to_csv(annotations_paths: list, txt_paths: list, output_path: str):

    """
    Cette fonction  a pour but de :
    Premièrement d'obtenir les captions du dataset COCO
    puis de les combiner avec les captions du dataset Flickr8k.   
    
    """
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["image", "caption"])  # en-tête unique
        
        #Traitement des fichiers JSON (COCO) 
        for annotations_path in annotations_paths:
            print(f"Traitement JSON : {annotations_path}")
            with open(annotations_path, "r") as f:
                data = json.load(f)
            
            id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}
            
            for annotation in data["annotations"]:
                image_id = annotation["image_id"]
                caption  = annotation["caption"]
                filename = id_to_filename[image_id]
                writer.writerow([filename, caption])
        
        #Traitement des fichiers TXT (Flickr8k style) 
        for txt_path in txt_paths:
            print(f"Traitement TXT : {txt_path}")
            with open(txt_path, "r", encoding="utf-8") as f:
                lines = f.read().strip().split('\n')[1:]  # skip en-tête
            
            for line in lines:
                parts = line.split(',', 1)
                if len(parts) == 2:
                    filename, caption = parts
                    writer.writerow([filename, caption])

    print(f"Fichier combiné créé : {output_path}")


if __name__ == "__main__":
    coco_to_csv(
        annotations_paths=[
            r"annotations_trainval2014\annotations\captions_train2014.json",
            r"annotations_trainval2014\annotations\captions_val2014.json",
        ],
        txt_paths=[
            r"archive\captions.txt",
        ],
        output_path=r"archive\All_captions.txt"
    )