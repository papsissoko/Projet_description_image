#from  Preprocessing.Text import  Textpreprocesseur
#from loading.load_data import loader
from  tensorflow.keras.preprocessing.text  import  Tokenizer
from  tensorflow.keras.preprocessing.sequence import pad_sequences
from  tensorflow.keras.layers import  Embedding

import numpy  as np 

def tokenization(caption): 
        caption = ["<start> " + text.replace(".", "") + " <end>" for text in caption]
        # On retire < et > des filtres pour conserver les balises <start> et <end>
        tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
        tokenizer.fit_on_texts(caption)
        sequences = tokenizer.texts_to_sequences(caption)
        sequences_padded = pad_sequences(sequences,  padding = "post")
        max_len  = len(sequences_padded[0])  #on  a fait un padding donc les shapes de sortie doivent  les mêmes. 
        return  sequences_padded, tokenizer.word_index ,  max_len
