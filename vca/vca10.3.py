#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

# X, Y = np.meshgrid(np.linspace(-2*np.pi,2*np.pi, 1000), np.linspace(-2, 2, 100))
X, Y = np.meshgrid(np.linspace(-7,7, 100), np.linspace(-7, 7, 100))
Z = X+1j*Y

f = Z**3 + (-1 + 5j)*Z**2 + (-9 - 2j)*Z + 1 - 7j
g=(Z-(1 -2j))**2 * (Z-(-1 -1j))
# f *= 1/np.abs(f)
# f *= np.log(1+np.abs(f))/np.abs(f)
# lw = 0.2+2*np.abs(f)
lw=0.3
# breakpoint()
U = np.real(f)
V = np.imag(f)
M = np.hypot(U, V)


fig1, ax1 = plt.subplots()
# Q = ax1.quiver(X, Y, U, V, M, pivot='middle', units='width')
S = ax1.streamplot(
    X, Y, U, V,
    density=1,
    color='k',
    linewidth=lw,
    broken_streamlines=False,
    arrowsize=0.5
)
U = np.real(g)
V = np.imag(g)
T = ax1.streamplot(
    X, Y, U, V,
    density=1,
    color='r',
    linewidth=lw,
    broken_streamlines=False,
    arrowsize=0.5,
)
plt.show()
# qk = ax1.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',
#                    coordinates='figure')
