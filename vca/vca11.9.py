#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax


def lnsin(z: jnp.complex64) -> jnp.complex64:
    return jnp.log(jnp.sin(z))


def f(z):
    # complex derivative f'(z) via JVP, works on scalars or arrays
    _, dz = jax.jvp(lnsin, (z,), (jnp.ones_like(z),))
    return dz

def g(z):
    # complex derivative f'(z) via JVP, works on scalars or arrays
    _, dz = jax.jvp(f, (z,), (jnp.ones_like(z),))
    return dz


def h(z):
    return np.conj(g(z))

def section_0():
    xlim = (-8,8)
    ylim = (-4,4)

    X, Y = np.meshgrid(np.linspace(xlim[0],xlim[1], 100).astype(float), np.linspace(ylim[0],ylim[1], 100).astype(float))
    Z = X+1j*Y

    breakpoint()
    H = h(Z)
    lw=0.3
    U_H = np.real(H)
    V_H = np.imag(H)
    M_H = np.hypot(U_H, V_H)

    _, ax1 = plt.subplots()
    ax1.streamplot(
        X, Y, U_H, V_H,
        density=1,
        color='k',
        linewidth=lw,
        broken_streamlines=False,
        arrowsize=0.5,
    )
    ax1.pcolormesh(X, Y, np.log(M_H), cmap='RdBu_r')
    plt.show()

section_0()
