import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import stats

# Paramètres physiques du problème
g = 9.81     # Champ gravitationnel (m²/s)
m = 1.000    # Masse du pendule (kg)
L = 1.000    # Longueur du câble (m)
beta = 0.1   # Constante d'amortissement (1/s)

# Conditions initiales
theta0 = np.pi/6        # Position initiale (rad)
omega0 = 5              # Vitesse inititale (rad/s)
dt0 = 2/omega0

# Boucle sur le nombre de simulations                                      
dt_val = [dt0, dt0/2, dt0/4, dt0/8, dt0/16]         # Vecteur des pas de temps pour chaque simulation
K = len(dt_val)                                     # Nombre de simulations
thetaf = np.zeros(K)                                # Vecteur des positions finales pour chaque simulation
tf_2 = 10                                           # Temps final (s)

for k in range(0,K):
# Paramètres spécifiques de la simulation
    dt = dt_val[k]               # Pas de temps de la simulation
    N = tf_2/dt                 # Nombre d'itérations (conseil : s'assurer que dt soit un multiple entier de tf)

# Initialisation
    t = np.arange(0, tf_2 + dt, dt)  # Vecteur des valeurs t_n
    theta = np.zeros(int(N + 1))  # Vecteur des valeurs theta_n
    theta[0] = theta0
    theta[1] = theta0 + (1 - (beta * dt) / 2) * omega0 * dt - (g / (2 * L) * (dt ** 2)) * np.sin(theta0) 


# Exécution
    for n in range(2, int(N + 1)):
        theta[n] = (4 * theta[n-1] - (2 - beta * dt) * theta[n-2] - ((2 * g) / L * (dt ** 2)) * np.sin(theta[n-1])) / (2 + beta * dt)

    thetaf[k] = theta[-1]  # Position au temps final tf
    plt.plot(t, theta, color='red')
    plt.xlabel('Temps (s)')
    plt.ylabel(r'$\theta$(t) (rad)')
    plt.grid()
    plt.show()

print(thetaf)

