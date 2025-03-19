import numpy as np
import matplotlib.pyplot as plt

epsilon = 0
a = 1 # Le paramètre de maille est donc l'unité de longueur
t = -2.5

kx = np.linspace(-np.pi,np.pi,1000)
ky = np.linspace(-np.pi,np.pi,1000)
kx, ky = np.meshgrid(kx, ky)

k = np.array([kx, ky])
a1 = a / 2 * np.array([3, np.sqrt(3)])
a2 = a / 2 * np.array([3, -np.sqrt(3)])

dot_k_a1 = np.tensordot(k, a1, axes=(0, 0))  
dot_k_a2 = np.tensordot(k, a2, axes=(0, 0))  
dot_k_a2_a1 = np.tensordot(k, a2 - a1, axes=(0, 0))

E_n = epsilon - t * np.sqrt(3 + 2 * np.cos(dot_k_a2_a1) + 2 * np.cos(dot_k_a1) + 2 * np.cos(dot_k_a2))
E_p = epsilon + t * np.sqrt(3 + 2 * np.cos(dot_k_a2_a1) + 2 * np.cos(dot_k_a1) + 2 * np.cos(dot_k_a2))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(kx, ky, E_n, cmap='viridis')
ax.plot_surface(kx, ky, E_p, cmap='viridis')
ax.set_xlabel(r'$\frac{k_x}{a_0}$')
ax.set_ylabel(r'$\frac{k_y}{a_0}$')
ax.set_zlabel(r'E($\mathbf{k}$) (eV)')

plt.pause(10)
plt.savefig("Diagramme_de_bande_2D1.png", dpi=400)
