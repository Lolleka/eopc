#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax



def Pot(z):
    return (jnp.exp(z)+1)/(jnp.exp(z)-1)

def Hbar(z):
    _, dz = jax.jvp(Pot, (z,), (jnp.ones_like(z),))
    return jnp.conj(dz)

def Kbar(z):
    # return - np.conj(np.exp(z)/(np.exp(z)-1)**2)
    return - (np.exp(np.conj(z))/(np.exp(np.conj(z))-1)**2)
    # return -1/(jnp.exp(jnp.conj(z))) - 

def Abar(z):
    return -1/(np.conj(z)**2) - 1/(np.conj(z))

def plot_field(ax, X, Y, H, color = 'k', lw = 0.3):
    U_H = np.real(H)
    V_H = np.imag(H)
    M_H = np.hypot(U_H, V_H)

    ax.streamplot(
        X, Y, U_H, V_H,
        density=1,
        color=color,
        linewidth=lw,
        broken_streamlines=False,
        arrowsize=0.5
    )
    ax.pcolormesh(X, Y, np.log(M_H), cmap='RdBu_r')

def section_0():
    xlim = np.array((-1,1))*10
    ylim = np.array((-1,1))*5
    N = 1000
    M = 2000

    X1, Y1 = np.meshgrid(np.linspace(xlim[0],xlim[1], N).astype(float), np.linspace(ylim[0],ylim[1], M).astype(float))

    hbar =(Hbar(X1+1j*Y1))
    kbar = (Kbar(X1+1j*Y1))

    X2, Y2 = np.meshgrid(np.linspace(-0.1,0.1, N).astype(float), np.linspace(-0.1,0.1, M).astype(float))
    abar = (Abar(X2+1j*Y2))

    _, ax1 = plt.subplots()
    plot_field(ax1, X1, Y1, hbar, color='k', lw=0.3)
    # plot_field(ax1, X, Y, kbar, color='r', lw=0.3)
    # plot_field(ax1, X2, Y2, abar, color='r', lw=0.3)

    plt.show()

section_0()
