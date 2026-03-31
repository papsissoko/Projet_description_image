import tensorflow as tf 
from  PIL import Image
import numpy as np 
from  Preprocessing.Images import  preprocesser
from models.Transformer.patcher_imp import test_embd, plot_patch,patchEmbedd
from models.Transformer.patch_encod import PatchEncoding
from models.Transformer.transformer_encoder import TransformerEncoderBlock
#from models.Transformer.classifier import Classifier
from models.Transformer.transformer_dec import Trasformer_dec
from models.Transformer.Positional_enc import Positional_enc
from models.CCN_LSTM.Embedding import Embedder
from models.Transformer.VIT import VIT


my_model = tf.keras.models.load_model("transformer.keras", compile=False)
image_test =np.array(Image.open(r"archive\Images\2599903773_0f724d8f63.jpg"))
image_processed = preprocesser(image_test)

text_input = ["<start>"] # On commence avec la balise de début
final_caption = []

pred = my_model.predict(image_processed)    
print(pred)