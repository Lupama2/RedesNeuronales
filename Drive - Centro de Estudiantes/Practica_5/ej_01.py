#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 10-05-21
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
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.set()

SAVE_PATH = os.path.join(os.getcwd(), "Figuras")
os.makedirs(SAVE_PATH, exist_ok=True)

Sigma = np.array([
    [2,1,1,1],
    [1,2,1,1],
    [1,1,2,1],
    [1,1,1,2]
])

sqrtSigma = np.array([
    [1.309, 0.309, 0.309, 0.309],
    [0.309, 1.309, 0.309, 0.309],
    [0.309, 0.309, 1.309, 0.309],
    [0.309, 0.309, 0.309, 1.309]
])

eValues, eVectors = np.linalg.eig(Sigma)

np.random.multivariate_normal([0,0,0,0], Sigma)


def ej1_fit(lr=1e-3, epochs=1000):

    w_log = []

    w = np.random.uniform(-0.01, 0.01, 4)                   # Inicializacion de los pesos
    # xi = np.random.multivariate_normal([0,0,0,0], Sigma)    # Entrada con la dist. del enunciado

    # w_log.append(w)

    for epoch in range(epochs):
        xi = np.random.multivariate_normal([0,0,0,0], Sigma)

        V = w.T @ xi

        dw = lr * V * (xi - V*w)
        w += dw

        w_log.append(np.copy(w))
    
    return np.array(w_log)

def plots_ej1():

    w = ej1_fit(epochs=5000)
    while w[-1][0] < 0:
        w = ej1_fit(epochs=5000)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(w[:,0], label=r'$\omega_{1}$')
    ax.plot(w[:,1], label=r'$\omega_{2}$')
    ax.plot(w[:,2], label=r'$\omega_{3}$')
    ax.plot(w[:,3], label=r'$\omega_{4}$')
    
    ax.set_xlabel("Epocas", fontsize=15)
    ax.set_ylabel(r"$\omega_{i}$", fontsize=15)
    ax.legend(loc='best')
    plt.tight_layout()
    
    fig_save_path = os.path.join(SAVE_PATH, "ej1_omegas.pdf")
    plt.savefig(fig_save_path, bbox_inches="tight")

    plt.show()

    #---------------------------#
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.plot(np.linalg.norm(w, axis=1))
    # ax.plot(np.linalg.norm(w, axis=1), label=r'$|\vec{\omega}|$')
    
    ax.set_xlabel("Epocas", fontsize=15)
    ax.set_ylabel(r"$|\vec{\omega}|$", fontsize=15)
    plt.tight_layout()
    
    fig_save_path = os.path.join(SAVE_PATH, "ej1_Norma.pdf")
    plt.savefig(fig_save_path, bbox_inches="tight")

    plt.show()

    #---------------------------#
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')

    # ax.plot(w[:,0], w[:,1], np.zeros_like(w[:,2]), linewidth=0.5, label=r'$\omega_{1}\omega_{2}$')
    # ax.plot(w[:,0], np.zeros_like(w[:,1]), w[:,2], linewidth=0.5, label=r'$\omega_{1}\omega_{3}$')
    # ax.plot(np.zeros_like(w[:,0]), w[:,1], w[:,2], linewidth=0.5, label=r'$\omega_{2}\omega_{3}$')

    # ax.plot([w[:,0][-1], w[:,0][-1]], [w[:,1][-1], w[:,1][-1]], [0, w[:,2][-1]], '--k')
    # ax.plot([w[:,0][-1], w[:,0][-1]], [0, w[:,1][-1]], [w[:,2][-1], w[:,2][-1]], '--k')
    # ax.plot([0, w[:,0][-1]], [w[:,1][-1], w[:,1][-1]], [w[:,2][-1], w[:,2][-1]], '--k')

    # ax.set_xlabel(r'$\omega_{1}$', fontsize=15)
    # ax.set_ylabel(r'$\omega_{2}$', fontsize=15)
    # ax.set_zlabel(r'$\omega_{3}$', fontsize=15)

    # ax.plot(w[:,0], w[:,1], w[:,2], label=r'$\omega_{1}\omega_{2}\omega_{3}$')
    # ax.view_init(elev=14, azim=24)
    # # ax.legend(loc='lower center', ncol=4, fancybox=True, shadow=False)
    # ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0), ncol=4, fancybox=True, shadow=False)
    # fig.tight_layout()

    # fig_save_path = os.path.join(SAVE_PATH, "ej1_3D.png")
    # plt.savefig(fig_save_path, bbox_inches="tight")

    # plt.show()





if __name__ == "__main__":
    
    plots_ej1()