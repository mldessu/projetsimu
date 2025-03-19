import time
import numpy as np
import matplotlib.pyplot as plt

# Équation différentielle: d^2 u/dx^2=g(x) sur x=(a,b)
# Conditions aux limites générales:
# x=a: c1*du/dx+c2*u+c3=0
# x=b: d1*du/dx+d2*u+d3=0

# Équation de transfert de chaleur d^2 T/dx^2=-S(x)/k sur x=(0,L)
# dans un mur d'isolation thermique
L = 0.3 #[m] ; Épaisseur du mur
k = 1     # k=1;#[W/(m*K)]; La conductivité thermique de la brique
h = 1     # h=1; #[W/(m^2*K)]; Coefficient de transfert thermique pour l'interface plane entre l'air et solide.
Ta = -10 #[oC]
Ti = 20

# Condition convective (de Robin) à x=0 (face externe du mur): -k*dT/dx=h(Ta-T)
c1 = -k
c2 = h
c3 = -h*Ta

# Condition de Neumann à x=L (face interne du mur): dT/dx=0 - flux net de chaleur est 0
d1 = 0
d2 = 1
d3 = -Ti

#(N+1) nœuds dans la maille
# Nmax=10000 pour 1G de mémoire

dx = 0.003
N = L / dx
x = np.linspace(0,L,int(N+1))

# Source volumique de chaleur q[W/m^3] d'épaisseur dL
# La source est intégrée dans la partie intérieure du mur
dL = 0.05 
q = 2000     # W/m^3;
S = q / (1+((x-L)/dL)**2) # Pourquoi c'est pas la même formule que dans le procédurier?

# Matrice pleine
A = np.diag(-2*np.ones(int(N+1)),0) + np.diag(np.ones(int(N)),-1) + np.diag(np.ones(int(N)),1)

A[0,0] = 2 * c2 * dx - 3 * c1
A[0,1] = 4 * c1
A[0,2] = -c1
A[int(N),int(N)] = 3 * d1 + 2 * d2 * dx
A[int(N),int(N-1)] = -4 * d1 
A[int(N),int(N-2)] = d1


b = -S / k * dx ** 2
b[0] = -2 * c3 * dx
b[int(N)] = -2 * d3 * dx

u = np.linalg.solve(A, b) # Option préférée

Tmax = u.max()

plt.plot(x,u)
plt.title('Température (°C)')
plt.xlabel('x [m]')    
plt.ylabel(r'$T_{eq}$(x) [$^o$C]')
plt.show()

