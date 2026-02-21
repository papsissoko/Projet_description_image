Ce  petit projet est une impléméntation  d'une technique pour la description  des images. 



Prétraitrement des données :  

Dans un premier on  commence à  charger les données depuis la fonction  load_data qui  se trouve dans le fichier ./loading:  le code python  se trouvant  a pour objectif d'extraire les images et  les textes et prendrant  soin  également de bien faire correspondre les images   à  leurs à  leurs descriptions. 

Pour le prétraitrement  des images,  rien d'autre qu'un rescaling (il faut que les images aient la même taille pour le modèle) puis une normalisation  afin d'assurer que le gradient ne puisse diviger ou  ne jamais converger. 



Le Modèle d'IA utilisée est composée de :  

  CNN :  dont l'objectif est d'extraire des features dans les images et puis ensuite de le fournir ces features apprisent  pour initialiser la mémoire du LSTM. 

  LSTM : l'objectif du LSTM est de dire la phrase qui correspond à  une phrase, à partir de ce qui a été appris par le CNN. 

Ici  dans ce modèle le CNN joue le rôle de yeux alors que le LSTM joue en une sorte le cerveau et  de la bouche. Et la Tokenization  et l'embedding peuvent  être interprété comme les transmetteurs des signaux des yeux aux cerveaux. 