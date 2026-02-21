import  torch  
from  transformers import  pipeline, VisionEncoderDecoderModel, ViTImageProcessor, GPT2Tokenizer
import sys 
import os 
import numpy as np
from PIL import Image

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(r"C:\Users\papch\project1")
if not project_root in  sys.path :  
    sys.path.append(project_root)
from loading.load_data import  loader

model_name = "nlpconnect/vit-gpt2-image-captioning"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
feature_extractor = ViTImageProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)
processor = ViTImageProcessor.from_pretrained(model_name)

device = 0 if torch.cuda.is_available() else -1
if device != -1:
    model.to(torch.float16)
_,_, images, paths = loader ( r"archive",  "captions.txt" , "Images" )

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
root_path= os.path.join(project_root,"project1")

def generate_captions_training( model : any = model, 
                      tokenizer: any= tokenizer, 
                      feature_extractor: any = feature_extractor,
                      device : int = device,
                      paths :list = paths,
                      root_path : str = root_path):
    paths= [os.path.join(root_path,path) for path in paths]
    for path in  paths :  
        images = Image.open(path)
        pixel_values =feature_extractor(images=images,return_tensors="pt").pixel_values
        print(type(model))
        output_ids = model.generate(pixel_values)
        preds =tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(preds)
        
    return model
generate_captions_training(root_path=root_path)
