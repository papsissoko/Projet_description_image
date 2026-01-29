from  Preprocessing.Text import  Textpreprocesseur
from lodading.load_data import loader
from  tensorflow.keras.preprocessing.text  import  Tokenizer
from  tensorflow.keras.preprocessing.sequence import pad_sequences
from  tensorflow.keras.layers import  Embedding 

def tokenization(caption): 
        # On réututiliilisera cette fonction  dans le Transformer. 
        caption = ["<start> " +text.replace(".", "")+"<end>" for text in  caption]
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(caption)
        sequences = tokenizer.texts_to_sequences(caption)
        sequences_padded = pad_sequences(sequences,  padding = "post")

        return  sequences_padded
if __name__ == "__main__" :  
        sequences = tokenization()
        print(type(sequences),  sequences.shape)