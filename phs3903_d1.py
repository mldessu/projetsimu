# PHS3903 - Projet de simulation
# Mini-devoir 1

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import interpolate
from scipy import stats
import math

# Paramètres géométriques
R = 1.0 # Rayon de la sphère (m)

# Paramètres généraux de simulation
D_val = [3,6]  # Nombre de dimensions
Ntot_val = 100 * 2 ** (np.arange(0, 5, 1))  # Nombre de points par essai
Ness = 100 # Nombre d'essais par simulation

a = 1  # Dimension de la boîte cubique dans laquelle les points aléatoires seront générés

# Boucle sur le nombre de simulations
ND = len(D_val)
NNtot = len(Ntot_val)

V = np.zeros((ND, NNtot))  # Volumes calculés pour chaque série d'essais
inc = np.zeros((ND, NNtot)) # Incertitudes pour chaque série d'essais

for d in range(0, ND):
    D = D_val[d]  # Dimension
    Vtot = ( a * 2 ) ** D # Volume du domaine
    Vth = (np.pi ** (D / 2)) / math.gamma(D / 2 + 1) # Volume théorique
    print(Vth)
    for n in range(0, NNtot):
        Ntot = Ntot_val[n]  # Nombre de points

        Vind = np.zeros(Ness) # Volumes calculés pour chaque essai individuel

        for k in range(0, Ness): # Boucle sur les essais
            # Génération des nombres aléatoires (distribution uniforme)
            np.random.seed() # Initialise le générateur de nombres pseudo-aléatoires afin de ne pas toujours produire la même séquence à l'ouverture de Python...
            pts = a * np.random.uniform(low=-a, high=a, size=(Ntot, D)) # Coordonnées des points
            
            # Calcul du volume
            distances = np.linalg.norm(pts, axis=1)
            Nint = np.sum(distances <= R) # Nombre de points à l'intérieur
            Vind[k] = Nint / Ntot * Vtot # Volume calculé pour cet essai

        V[d, n] = np.mean(Vind) # Volume moyenné sur l'ensemble des essais
        inc[d, n] = np.std(Vind) # Incertitude sur l'ensemble des essais (écart-type)
print(V)
print(inc)

inc_rel = inc / V * 100

print(inc_rel)
