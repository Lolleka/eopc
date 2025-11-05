#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

# X, Y = np.meshgrid(np.linspace(-2*np.pi,2*np.pi, 1000), np.linspace(-2, 2, 100))
C=0.0+0.0j
# C=2*np.pi+0.0j
A = 0.1
b = A*2
xlim = (C.real-b, C.real+b)
ylim = (C.imag-b, C.imag+b)

X, Y = np.meshgrid(np.linspace(xlim[0],xlim[1], 100), np.linspace(ylim[0],ylim[1], 100))
Z = X+1j*Y

def f(z):
    return 1/(z*np.sin(z))
def h(z):
    return np.conj(f(z))


# define small circle:
theta = np.linspace(0, 2*np.pi, 500)
dtheta = theta[1]-theta[0]
circle = A*np.exp(1j*theta)
tv = circle*1j/np.abs(circle)
nv = -1j*tv
h_c = h(circle+C)


dflow = np.empty(len(h_c))
for i in range(len(h_c)):
    dflow[i] = (h_c[i].real*nv[i].real + h_c[i].imag*nv[i].imag)*A*dtheta
flow = np.sum(dflow)
dwork = np.empty(len(h_c))
for i in range(len(h_c)):
    dwork[i] = (h_c[i].real*tv[i].real + h_c[i].imag*tv[i].imag)*A*dtheta
work = np.sum(dwork)

print(flow)
print(work)

F = f(Z)
H = np.conj(F)
lw=0.3
U_H = np.real(H)
V_H = np.imag(H)
M_H = np.hypot(U_H, V_H)

U_F = np.real(F)
V_F = np.imag(F)
M_F = np.hypot(U_F, V_F)


fig1, ax1 = plt.subplots()
# Q = ax1.quiver(X, Y, U, V, M, pivot='middle', units='width')
S_H = ax1.streamplot(
    X, Y, U_H, V_H,
    density=1,
    color='k',
    linewidth=lw,
    broken_streamlines=False,
    arrowsize=0.5,
)
# heatmap for H magnitude
ax1.pcolormesh(X, Y, np.log(M_H), cmap='RdBu_r')
# draw circle
circle_plt = plt.Circle((C.real, C.imag), A, color='red', fill=False)
ax1.add_patch(circle_plt)

# # quivers only on the circle
# circle = circle[0::10]
X = (circle+C).real
Y = (circle+C).imag
# F = f(circle+C)
# H = np.conj(F)
H = h_c
U_H = np.real(H)
V_H = np.imag(H)
M_H = np.hypot(U_H, V_H)
ax1.quiver(X, Y, U_H, V_H, M_H, pivot='tail', units='width', color='k')


ax1.set_xlim(xlim)
ax1.set_ylim(ylim)

plt.show()


