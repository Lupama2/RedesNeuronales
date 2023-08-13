#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 23-05-21
File: ej_01.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description: 
"""

import os
import numpy as np
import matplotlib.pyplot as plt



# p = 60
# N = 500

Ns = [500, 1000, 2000, 4000]
alphas = [0.12, 0.14, 0.16, 0.18]

# fig = plt.figure()
# gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)


# fig11 = plt.figure(figsize=(8, 8), constrained_layout=False)
# outer_grid = fig11.add_gridspec(4, 4, wspace=0, hspace=0)

fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(10, 6))


for i, N in enumerate(Ns):
    for j, alpha in enumerate(alphas):
        p = int(alpha*N)

        patters = np.random.choice([-1, 1], size=(N, p)).astype(float)

        J = (patters @ patters.T)/N
        np.fill_diagonal(J, 0)

        m_log = []

        for mu in range(p):
            xi = np.copy(patters[: ,mu:mu+1])
            S = np.copy(patters[: ,mu:mu+1])
            
            S_new = J @ S
            S_new[S_new >= 0] = 1.0
            S_new[S_new < 0] = -1.0
            
            n_iter = 1

            while(not (S == S_new).all()):
                print(N, " ", p, " ", n_iter, " ", (S_new==S).sum())
                S = np.copy(S_new)
                # S_new = np.sign(J @ S)

                S_new = J @ S
                S_new[S_new >= 0] = 1.0
                S_new[S_new < 0] = -1.0

                # print(np.where( S != S_new ))
                print(S[np.where( S != S_new )])
                print(S_new[np.where( S != S_new )])

                # import ipdb; ipdb.set_trace(context=15)  # XXX BREAKPOINT

                n_iter += 1

                assert(len(np.unique(S_new)) == 2)
            
            m = (xi * S).mean()
            m_log.append(m)

        m_log = np.array(m_log)

        axs[i,j].hist(m_log)

plt.show()

            
