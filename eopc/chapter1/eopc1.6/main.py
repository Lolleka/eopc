#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def make_gom(n: int):
    # position = 0
    # scale - 1
    m = np.random.normal(0, 1, (n, n))
    return m+m.T

def make_goe(n: int, m: int):
    return np.array([make_gom(n) for _ in range(m)])

def calc_ev_split(goe):
    n = goe[0].shape[0]
    eigenvalues = np.stack(list(map(lambda x: np.sort(np.linalg.eigvals(x)), goe)))
    return np.array(list(map(lambda x: x[n//2]-x[n//2-1], eigenvalues)))
    
def wigner_surmise(x):
    return (np.pi*x/2)*np.exp(-np.pi*x**2/4)

def section_a():

# calculate eigenvalues of each gom
    N = 2
    M = 5000
    goe = make_goe(N, M)
    # build histogram of eigenvalues differences (at n = N/2)
    plt.figure(1)
    ev_split = [calc_ev_split(make_goe(N, M)) for N in (2,4,10)]
    for i in range(len(ev_split)):
        plt.hist(ev_split[i]/np.average(ev_split[i]), bins=100)
    plt.show()

def section_e():
    x = np.linspace(0, 6, 1000)

    M = 5000
    plt.figure(1)
    for N in (2,4,10):
        # build histogram of eigenvalues differences (at n = N/2)
        ev_split = calc_ev_split(make_goe(N, M))
        plt.hist(ev_split/np.average(ev_split), bins=100, density=True)
    plt.plot(x, wigner_surmise(x))
    plt.show()

def section_f():
    x = np.linspace(0, 6, 1000)
    M = 20000
    for N in (2,4,10,20):
        plt.figure()
        # build histogram of eigenvalues differences (at n = N/2)
        goe = make_goe(N, M)
        goe = goe / np.abs(goe)
        # print(goe); breakpoint()
        ev_split = calc_ev_split(goe)
        plt.hist(ev_split/np.average(ev_split), bins=N*20, density=True)
        plt.plot(x, wigner_surmise(x))
    plt.show()

if __name__ == '__main__':
    # section_a()
    # section_e()
    section_f()

