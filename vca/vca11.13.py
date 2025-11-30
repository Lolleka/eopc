#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax



# def Pot(z, v=0.1):
#     return (jnp.exp(z)+1)/(jnp.exp(z)-1) + v*jnp.real(jnp.conj(z))
def Pot(z):
    return (jnp.exp(z)+1)/(jnp.exp(z)-1)

def Hbar(z):
    _, dz = jax.jvp(Pot, (z,), (jnp.ones_like(z),))
    return jnp.conj(dz) + 1


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

    X2, Y2 = np.meshgrid(np.linspace(-0.1,0.1, N).astype(float), np.linspace(-0.1,0.1, M).astype(float))

    _, ax1 = plt.subplots()
    plot_field(ax1, X1, Y1, hbar, color='k', lw=0.3)

    plt.show()

section_0()
