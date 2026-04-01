import  pickle
from  Tokenizer  import   tokenization
from load_data import  loader
import  os 


def dump_tokens_pkl () : 
    
    captions,_,_,_  = loader ( r"archive",  "captions.txt" , "Images" )
    tokens, word_index, max_len = tokenization(caption=captions)
    tokens_val, word_index_val, max_len_val = tokenization(caption=captions)
    vocab_size = len(word_index) + 1
    os.makedirs("tokens", exist_ok=True)

    with open("tokens/tokenizer_data_train.pkl", "wb") as f: 
        pickle.dump({"word_index": word_index, "max_len": max_len, "tokens_padded": tokens},f)

    with open("tokens/tokenizer_data_test.pkl", "wb") as f: 
        pickle.dump({"word_index": word_index_val, "max_len": max_len_val, "tokens_padded": tokens_val},f)
    
    print("Données du tokenizer sauvegardées dans tokenizer_data.pkl")

    return  

if __name__ == "__main__" : 
    dump_tokens_pkl()

