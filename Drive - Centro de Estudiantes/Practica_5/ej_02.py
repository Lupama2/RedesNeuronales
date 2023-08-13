#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 11-05-21
File: ej_02.py
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

SAVE_PATH = os.path.join(os.getcwd(), "Figuras")
os.makedirs(SAVE_PATH, exist_ok=True)

def gaussianNeighborFunc(i, i_, sigma):
    """
    Gaussian Neighborhood function for exercise 2. Takes the index, the winner index and the sigma parameter in the exponencial
    """
    return np.exp(-(((i-i_)**2)/(2*(sigma**2))))


def generateInput():
    """
    Generate a 2D input with r in [0.9,1.1] and theta in [0,pi] with uniform distribution
    """
    valid = False
    while not valid:
        x = np.random.uniform(-1.1,1.1)
        y = np.random.uniform(0,1.1)

        r = np.linalg.norm([x,y])
        if 0.9 <= r <= 1.1:
            valid = True
    return x, y

def getWinnerIndex(W, xi):
    """
    Get index of the closest column of W to xi, using norm2 as distance
    """
    return np.argmin(np.linalg.norm(W.T-xi, axis=1))

def getDW(i_, W, xi, sigma):
    dW = (xi - W.T)

    # import ipdb; ipdb.set_trace(context=15)  # XXX BREAKPOINT
    for col in range(len(dW)):
        Lambda = gaussianNeighborFunc(col, i_, sigma)
        dW[col] *= Lambda
    # import ipdb; ipdb.set_trace(context=15)  # XXX BREAKPOINT
    return dW.T
    

def ej2_fit(lr=1e-2, epochs=1000, sigma=1):

    # Inicializo los pesos en una linea entre los puntos (-1, 0.5) y (1, 0.5)
    W = np.mgrid[-1:1:10j, 0.5:0.6:0.1].reshape(2,-1)
    w_log = [np.copy(W)]

    for epoch in range(epochs):
        xi = generateInput()
        i_ = getWinnerIndex(W, xi)

        dW = getDW(i_, W, xi, sigma)
        W += (lr*dW)

        w_log.append(np.copy(W))

    
    return np.array(w_log)

def auxPlotFunc(x_):
    res = []
    for x in x_:
        if -0.9 <= x and x <= 0.9:
            res.append((0.9**2 - x**2)**0.5)
        else:
            res.append(0.0)
    return np.array(res)
        


def ej2_plots(sigma):

    input_zone_x = np.linspace(-1.1, 1.1, 10000)
    input_zone_ymax = (1.1**2 - input_zone_x**2)**0.5
    input_zone_ymin = auxPlotFunc(input_zone_x)

    W10 = ej2_fit(lr=0.1, sigma=sigma, epochs=10)
    W100 = ej2_fit(lr=0.1, sigma=sigma, epochs=100)
    W1000 = ej2_fit(lr=0.1, sigma=sigma, epochs=1000)
    W10000 = ej2_fit(lr=0.1, sigma=sigma, epochs=10000)

    fig = plt.figure(figsize =(7.8,4.5))
    ax = fig.add_subplot(111)
    ax.set_title(r"$\sigma=${}".format(sigma))

    ax.plot(W10000[0][0], W10000[0][1], 'o--', label=r'$Inicial$')
    
    ax.plot(W10[-1][0], W10[-1][1], 'o--', label=r'$10$')
    ax.plot(W100[-1][0], W100[-1][1], 'o--', label=r'$100$')
    # ax.plot(W1000[-1][0], W1000[-1][1], 'o--', label=r'$1000$')
    ax.plot(W10000[-1][0], W10000[-1][1], 'o--', label=r'$10000$')


    ax.fill_between(input_zone_x, input_zone_ymin, input_zone_ymax, color='C0', alpha=0.2)

    ax.set_xlabel(r"$\xi_{1}$")
    ax.set_ylabel(r"$\xi_{2}$")
    

    ax.set_xlim(-1.1,1.1)
    ax.set_ylim(0,1.1)

    ax.legend(loc='best')
    plt.tight_layout()

    fig_save_path = os.path.join(SAVE_PATH, "ej2_{:.2f}.pdf".format(sigma))
    plt.savefig(fig_save_path, bbox_inches="tight")


    plt.show()


if __name__ == "__main__":
    
    ej2_plots(sigma=0.01)
    ej2_plots(sigma=0.5)
    ej2_plots(sigma=5)