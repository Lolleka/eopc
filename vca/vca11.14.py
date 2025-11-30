#!/usr/bin/env python3

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def Hbar(z, v = 0):
    return 2/(np.conj(z)**2-1) + v


def plot_field(ax, X, Y, H, color = 'k', lw = 0.3):
    U_H = np.real(H)
    V_H = np.imag(H)
    M_H = np.hypot(U_H, V_H)

    return ax.streamplot(
        X, Y, U_H, V_H,
        density=1,
        color=color,
        linewidth=lw,
        broken_streamlines=False,
        arrowsize=0.5
    )
    # ax.pcolormesh(X, Y, np.log(M_H), cmap='RdBu_r')

def section_0():
    fig, ax = plt.subplots()
    xlim = np.array((-1, 1)) * 3
    ylim = np.array((-1, 1)) * 3
    N = 100
    M = 100

    X1, Y1 = np.meshgrid(
        np.linspace(xlim[0], xlim[1], N).astype(float),
        np.linspace(ylim[0], ylim[1], M).astype(float)
    )

    vs = np.linspace(0, 3, 50)

    def init():
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_aspect('equal')
        H0 = Hbar(X1 + 1j * Y1, v=vs[0])
        plot_field(ax, X1, Y1, H0, color='k', lw=0.3)
        return []

    def update(frame):
        ax.clear()
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_aspect('equal')
        H = Hbar(X1 + 1j * Y1, v=vs[frame])
        plot_field(ax, X1, Y1, H, color='k', lw=0.3)
        return []

    ani = FuncAnimation(
        fig,
        update,
        frames=len(vs),
        interval=100,
        init_func=init,
        blit=False
    )
    plt.show()


section_0()
