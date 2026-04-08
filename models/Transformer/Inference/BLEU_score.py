import  warnings


def Bleu_score ( y_pred ,  y_true ,  n_gram : int = 2 ) :
    """
    Ceci est  une implémentation  de la fonction  BLEU_score  
    """  
    if n_gram > len(y_pred) or n_gram > len(y_true) :
        n_gram = min( len(y_pred), len(y_true))
        warnings.warn( f" Comme le n_gram  est supérieur au longueur de l'une des deux paramètres alors il  a été réduits  à {n_gram}")
    
    pred_unique_word = y_pred.split()
    true_unique_word = y_true.split()

    pred_unique_word_n_gram = [" ".join(x) for x in (pred_unique_word[i:i+n_gram] for i in  range(len(pred_unique_word)-n_gram))]

    true_unique_word_n_gram = [" ".join(x) for x in (true_unique_word[i:i+n_gram] for i in  range(len(true_unique_word)-n_gram))]
    
    # Précision  naïve =====>
    # si le modèle predit par exemple y=[ le chat le chat ...] et y_tre= [ le chat mange un souris ] gros BLEU score or on  en veut pas
    pred_formated = set(pred_unique_word_n_gram)
    true_formated = set(true_unique_word_n_gram)

    bleu_score = 0
    for word in   pred_formated :
            bleu_score += 1
    return bleu_score/len(pred_unique_word)


