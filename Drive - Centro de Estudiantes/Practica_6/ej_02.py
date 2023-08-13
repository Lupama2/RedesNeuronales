#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 29-05-21
File: ej_2.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description: 
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def probabilitiesState(h, T):
    exp_p = np.exp(h/T)
    exp_n = np.exp(-h/T)
    prob_p = exp_p / (exp_p+exp_n)
    prob_n = 1 - prob_p
    return [prob_p, prob_n]

N = 4000
p = 40
Ts = np.arange(2, 0, -0.1)
m_mean = []
m_std = []

patters = np.random.choice([-1.0, 1.0], size=(p,N))

J = (patters.T @ patters)/N
np.fill_diagonal(J, 0)

for T in Ts:
    print(T)
    idx = np.arange(N)

    m_log = []

    for mu in range(p):
        print("{}/{}".format(mu,p), end='\r')

        xi = np.copy(patters[mu])
        S = np.copy(patters[mu])
        for _ in range(10):
            np.random.shuffle(idx)
            # No se si calcular el h una vez antes de recorrer todo o calcularlo de nuevo para cada componente.
            # h = J @ S
            for i in idx:
                h_i = J[i] @ S

                prop_p, _ = probabilitiesState(h_i, T)

                coin = np.random.rand()
                if coin <= prop_p:
                    S[i] = 1.0
                else:
                    S[i] = -1.0
        
        m = (xi * S).mean()
        m_log.append(m)
    
    m_log = np.array(m_log)
    m_mean.append(m_log.mean())
    m_std.append(m_log.std())

m_mean = np.array(m_mean)
m_std = np.array(m_std)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.fill_between(Ts, m_mean+m_std, m_mean-m_std, alpha=0.5)
ax.scatter(Ts, m_mean)
ax.set_xlabel("T")
ax.set_ylabel(r"$\overline{m}$")
plt.savefig("2.pdf", bbox_inches='tight')
plt.show()

