# PHS3903 - Projet de simulation
# Mini-devoir 1

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import interpolate
from scipy import stats

# Paramètres géométriques
R = 1.0 # Rayon de la sphère (m)

# Paramètres généraux de simulation
D_val = [3,6]  # Nombre de dimensions
Ntot_val = 100 * 2 ** (np.arange(1, 4, 1))  # Nombre de points par essai
Ness = 100 # Nombre d'essais par simulation

a = 1  # Dimension de la boîte cubique dans laquelle les points aléatoires seront générés

# Boucle sur le nombre de simulations
ND = len(D_val)
NNtot = len(Ntot_val)

V = np.zeros((ND, NNtot))  # Volumes calculés pour chaque série d'essais

for d in range(0, ND):
    D = D_val[d]  # Dimension
    Vtot = 1 # Volume du domaine
    Vth = 1 # Volume théorique

    for n in range(0, NNtot):
        Ntot = Ntot_val[n]  # Nombre de points

        Vind = np.zeros(Ness) # Volumes calculés pour chaque essai individuel

        for k in range(0,Ness): # Boucle sur les essais
            # Génération des nombres aléatoires (distribution uniforme)
            np.random.seed() # Initialise le générateur de nombres pseudo-aléatoires afin de ne pas toujours produire la même séquence à l'ouverture de Python...
            pts = a * np.random.uniform(low=-1, high=1, size=(Ntot, D)) # Coordonnées des points

            # Calcul du volume
            Nint = 1 # Nombre de points à l'intérieur
            Vind[k] = Nint / Ntot * Vtot # Volume calculé pour cet essai

        V[d, n] = np.mean(Vind) # Volume moyenné sur l'ensemble des essais

print(V)
