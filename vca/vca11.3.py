#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np


def f(z):
    return 1/(z*np.sin(z))
def h(z):
    return np.conj(f(z))

def section_1to4():
    C=2*np.pi+0.0j
    A = 0.1
    b = A*2
    xlim = (C.real-b, C.real+b)
    ylim = (C.imag-b, C.imag+b)

    X, Y = np.meshgrid(np.linspace(xlim[0],xlim[1], 100), np.linspace(ylim[0],ylim[1], 100))
    Z = X+1j*Y

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
    S_H = ax1.streamplot(
        X, Y, U_H, V_H,
        density=1,
        color='k',
        linewidth=lw,
        broken_streamlines=False,
        arrowsize=0.5,
    )
    ax1.pcolormesh(X, Y, np.log(M_H), cmap='RdBu_r')
    circle_plt = plt.Circle((C.real, C.imag), A, color='red', fill=False)
    ax1.add_patch(circle_plt)

    X = (circle+C).real
    Y = (circle+C).imag
    H = h_c
    U_H = np.real(H)
    V_H = np.imag(H)
    M_H = np.hypot(U_H, V_H)
    ax1.quiver(X, Y, U_H, V_H, M_H, pivot='tail', units='width', color='k')

    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)

    plt.show()

def section_5():
    R = np.concat((
        np.linspace(7+1j, 1+1j, 300),
        np.linspace(1+1j, 1-1j, 300),
        np.linspace(1-1j, 7-1j, 300),
        np.linspace(7-1j, 7+1j, 300)))
    xlim = (-1,9)
    ylim = (-2,2)

    X, Y = np.meshgrid(np.linspace(xlim[0],xlim[1], 100), np.linspace(ylim[0],ylim[1], 100))
    Z = X+1j*Y

    tv = np.diff(np.concat((R, (R[0],))))
    tv = tv/np.abs(tv)
    nv = -1j*tv
    h_r = h(R)

    dflow = np.empty(len(h_r))
    for i in range(len(h_r)):
        ds = np.abs(R[(i+1)%len(R)]-R[i])
        dflow[i] = (h_r[i].real*nv[i].real + h_r[i].imag*nv[i].imag)*ds
    # remove nans
    flow = np.sum(dflow[~np.isnan(dflow)])

    dwork = np.empty(len(h_r))
    for i in range(len(h_r)):
        ds = np.abs(R[(i+1)%len(R)]-R[i])
        dflow[i] = (h_r[i].real*nv[i].real + h_r[i].imag*nv[i].imag)*ds
        dwork[i] = (h_r[i].real*tv[i].real + h_r[i].imag*tv[i].imag)*ds
    work = np.sum(dwork[~np.isnan(dwork)])

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
    S_H = ax1.streamplot(
        X, Y, U_H, V_H,
        density=1,
        color='k',
        linewidth=lw,
        broken_streamlines=False,
        arrowsize=0.5,
    )
    ax1.pcolormesh(X, Y, np.log(M_H), cmap='RdBu_r')
    ax1.add_patch(plt.Rectangle((1,-1), 6,2, color='blue', fill=False))

    X = R.real
    Y = R.imag
    H = h_r
    U_H = np.real(H)
    V_H = np.imag(H)
    M_H = np.hypot(U_H, V_H)
    ax1.quiver(X, Y, U_H, V_H, M_H, pivot='tail', units='width', color='k')

    # U_H = np.real(tv)
    # V_H = np.imag(tv)
    # M_H = np.hypot(U_H, V_H)
    # ax1.quiver(X, Y, U_H, V_H, M_H, pivot='tail', units='width', color='k')

    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)

    plt.show()

section_5()
