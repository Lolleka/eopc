#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

# X, Y = np.meshgrid(np.linspace(-2*np.pi,2*np.pi, 1000), np.linspace(-2, 2, 100))
X, Y = np.meshgrid(np.linspace(-5,5, 1000), np.linspace(-2, 2, 100))
Z = X+1j*Y

f = (1/Z)*(1/(np.sin(Z)**2))/20
# f *= 1/np.abs(f)
f *= np.log(1+np.abs(f))/np.abs(f)
lw = 0.2+2*np.abs(f)
# breakpoint()
U = np.real(f)
V = np.imag(f)
M = np.hypot(U, V)


fig1, ax1 = plt.subplots()
# Q = ax1.quiver(X, Y, U, V, M, pivot='middle', units='width')
S = ax1.streamplot(
    X, Y, U, V,
    density=2,
    color='k',
    linewidth=lw,
    broken_streamlines=False,
    arrowsize=0.5
)
plt.show()
# qk = ax1.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',
#                    coordinates='figure')
