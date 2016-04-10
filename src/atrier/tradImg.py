# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 14:57:51 2016

@author: daphne
"""
"""
Qst : listes ou tableaux?

"""

# lecture des tags par ligne (pour 3 colonnes, ligne du haut : 2^0,2^1,2^2)
NB_ROBOTS = 100
NB_LIGNES = 3
NB_COLS = 3
SEUIL = 150

import cv2
import numpy as np

def valInit(nb_robots):
    nb_chiffres = len(str(nb_robots))-2
    min_puissance = 2**(nb_chiffres-1)
    plus_grande_2puissance = nb_chiffres-1 if min_puissance<nb_robots else nb_chiffres
    # TODO a modifier par le calcul
    NB_LIGNES = 3
    NB_COLS = 3


"""
identification du robot    
renvoie l'id lu a partir du tagId 
qui est la partie identification du robot

Contexte : tous les filtres ont été appliqués,
        l'image correspond a l'identifiant du robot sur le tag
"""
def readId(tagId):
    # le pas pour parcourir l'image 
    height = len(noir_blanc)
    width = len(noir_blanc[0])
    pas_lignes = height/NB_LIGNES
    pas_cols = width/NB_COLS
    
    # recupere un tableau avec juste des zeros et des 1
    tab = noir_blanc/255
    # identifiant: tableau aplati de la zone identifiant du tag en restreint (0:blanc,1:noir)
    identifiant = np.empty((NB_COLS*NB_LIGNES))
    
    # pour chaque ligne
    for i in range(0,pas_lignes*NB_LIGNES,pas_lignes):
        # pour chaque colonne
        i_id = i%NB_LIGNES
        for j in range(0,pas_cols*NB_COLS,pas_cols):
            # on calcule l'indice du tableau de l'ID
            j_id = j%NB_COLS
            ind_id = i_id*NB_LIGNES+j_id
            # on recupere le sous-tableau en calculant la moyenne de tous les points
            sous_tab = tab[i:i+pas_lignes,j:j+pas_cols]
            # on recupere le tableau restreint de l'identifiant
            val = round(sous_tab.mean())
            # on remplace les 1 par des 0 et inversement pour calculer
            # on inverse les valeurs pour accentuer les noirs (0)->1
            identifiant[ind_id] = int(-(val-1)) # on recupere l'integer correspondant
        # end for
    # end for
    ###
    # lecture de l'id
    # fct_2puissance_x(x):2**x
    #sum(map(fct_2puissance_x,list(np.where(img==1)))
    # on recupere la liste des indices contenant des 1 (ie des cases noires)
    indices = list(np.where(identifiant==1)[0])
    # on recupere la les puissances de 2 correspondantes
    g = lambda x:2**x
    indices = map(g,indices)
    # on recupere l'identifiant du robot
    # qui n'est autre que la somme des indices apres tranformation
    return sum(indices)

"""
determination de l'orientation du tag   
renvoie l'orientation du tag
enum?

Contexte : tous les filtres ont été appliqués,
        l'image correspond aux clignotants du robot sur le tag
"""
def readDirection(dirImg):
    pass



valInit(NB_ROBOTS)
# lecture de l'image
img = cv2.imread("tagCenter_noContour.png",0)

########## traitement de l'image
noir_blanc = cv2.equalizeHist(img)
# seuillage de l'image
ret,noir_blanc = cv2.threshold(noir_blanc,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
contImg = cv2.Canny(noir_blanc,100,200, L2gradient=True)
contours,hierarchie = cv2.findContours(contImg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# recuperation du tag dans l'image
######## TODO 
# tag complet?
# tag flat

ID = readId(img)
print("Robot numéro {}".format(ID))

# on affiche l'image
cv2.imshow("image",noir_blanc)
cv2.waitKey(0)