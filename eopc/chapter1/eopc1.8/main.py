#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def BirthdayCoincidences(K, C, D = 365):
    bdays = np.array([np.sort(np.random.randint(D, size=(K,))) for _ in range(C)])
    same_bdays = np.sum(np.array(list(map(lambda c: 0 in c, np.diff(bdays, axis=1)))))
    return same_bdays/C

def section_a():
    C = 20
    D = 365
    K_max = 100
    K = np.linspace(1, K_max, K_max).astype(int)
    P = np.zeros((len(K), ))
    N = 100
    for _ in range(N):
        P += np.array(list(map(lambda K: BirthdayCoincidences(K, C, D), K)))
    P /= N
    P_approx = 1-(1-1/D)**(K*(K-1)/2)
    P_approx2 = 1-np.exp(-K**2/(2*D))
    plt.plot(K, P)
    plt.plot(K, P_approx)
    plt.plot(K, P_approx2)
    plt.show()

def section_b():
    D = 2**32
    K = np.ceil(np.sqrt(2*D*np.log(2))).astype(int)
    print(K)
    N = 1000
    C = 0
    L = np.ceil(K**(1/3)).astype(int)
    print(L)
    for _ in range(N):
        C += 1 if (np.min(np.diff(np.sort((np.random.randint(D, size=(K,)))))) == 0) else 0
    print(C/N)
    
    
    

if __name__ == '__main__':
    # section_a()
    section_b()
