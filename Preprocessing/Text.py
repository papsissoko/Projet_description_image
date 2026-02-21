import numpy  as np  
import  matplotlib.pyplot as plt 
from loading.load_data import loader

#caption, _ ,  _ = loader ( r"archive",  "captions.txt" , "Images" )
class Textpreprocesseur : 
    def __init__(self):
        pass
    
    def preprocesser ( self, x) : 
        """
        lignes = x.strip().split('\n')[1:] 
        image_ref = [l.split(',', 1)[0] for l in lignes]
        caption   = [l.split(',', 1)[1] for l in lignes]

        """ 
        words = "".join(x)
        all_words = words.split()
        self.unique_words = set(all_words)
        self.unique_words_len = len(self.unique_words)
        return self.unique_words,self.unique_words_len 
    
