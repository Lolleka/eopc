#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax


def invPot(w):
    return w + np.exp(w)

def s(z):
    return np.ones_like(z)

def Pot(z):
    return jnp.log(jnp.log(z)+z)

def g(z):
    # complex derivative Pot'(z) via JVP, works on scalars or arrays
    _, dz = jax.jvp(Pot, (z,), (jnp.ones_like(z),))
    return dz

def k(z):
    #return 1+np.exp(np.conj(z))
    return 1+np.exp(z)

def section_0():
    xlim = (-20,20)
    ylim = (-8,8)
    N = 2002
    M = 2000

    x, y = np.meshgrid(np.linspace(xlim[0],xlim[1], N).astype(float), np.linspace(ylim[0],ylim[1], M).astype(float))
    Z = x+1j*y

    f = invPot(Z)
    F = np.conj(f[:,1:-1]-f[:,0:-2])
    print(F.shape)
    X = x[:,0:-2]
    Y = y[:,0:-2]

    lw=0.3
    U_F = np.real(F)
    V_F = np.imag(F)
    M_F = np.hypot(U_F, V_F)

    _, ax1 = plt.subplots()
    ax1.streamplot(
        X, Y, U_F, V_F,
        density=1,
        color='k',
        linewidth=lw,
        broken_streamlines=False,
        arrowsize=0.5
    )
    ax1.pcolormesh(X, Y, np.log(M_F), cmap='RdBu_r')
    plt.show()

def section_1():
    xlim = (-20,20)
    ylim = (-8,8)
    N = 2002
    M = 2000

    X, Y = np.meshgrid(np.linspace(xlim[0],xlim[1], N).astype(float), np.linspace(ylim[0],ylim[1], M).astype(float))
    Z = X+1j*Y


    K = k(Z)

    lw=0.3
    U_K = np.real(K)
    V_K = np.imag(K)
    M_K = np.hypot(U_K, V_K)

    _, ax1 = plt.subplots()
    ax1.streamplot(
        X, Y, U_K, V_K,
        density=1,
        color='k',
        linewidth=lw,
        broken_streamlines=False,
        arrowsize=0.5
    )
    ax1.pcolormesh(X, Y, np.log(M_K), cmap='RdBu_r')
    plt.show()

def section_2():
    # here we manually plot just the streamlines for -pi < Psi < pi

    N = 100
    psi = np.linspace(-np.pi,np.pi,20)
    phi = np.linspace(-4, 1, N)
    xlim = (-20,20)
    ylim = (-8,8)
    Phi, Psi = np.meshgrid(phi, psi)

    W = Phi+1j*Psi
    f = invPot(W)
    _, ax1 = plt.subplots()
    for i in range(len(Psi)):
        ax1.plot(np.real(f[i,:]), np.imag(f[i,:]), color='k', linewidth=0.5)
    plt.show()


section_2()
