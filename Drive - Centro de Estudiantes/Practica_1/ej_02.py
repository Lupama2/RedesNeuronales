#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 18-03-21
File: ej_02.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description: 
"""

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()

T = 20.0 + 273.15     # K : Kelvin
k = 8.6e-5          # ev/K

z = 1 # Segun Alavaro las valencias son todas 1, pero creo que del Cloro no, pero fue

class Ion(object):
    def __init__(self, name, C_in=None,C_out=None, P=None, z=1):
        self.name = name
        self.C_in = C_in        # Concentracion adentro
        self.C_out = C_out      # Concentracion afuera
        self.P = P              # Permeabilidad?
        self.z = z              # Valencia
    def goldman(self,V):
        xi = z * V / (k * T)
        num = self.C_out - self.C_in * np.exp(xi)
        den = 1 - np.exp(xi)
        return self.P * xi * num / den
    def nernst(self):
        return k * T * np.log(self.C_in / self.C_out)

if __name__ == "__main__":
    
    K = Ion(r"$K^{+}$", C_in=430, C_out=20, P=1)
    Na = Ion(r"$Na^{+}$", C_in=50, C_out=40, P=0.03)
    Cl = Ion(r"$Cl^{-}$", C_in=65, C_out=550, P=0.1)

    V = np.linspace(-0.1, 0.1, 1000)

    for ion in [K, Na, Cl]:
        plt.plot(V, ion.goldman(V), label=ion.name)

    plt.ylabel(rf"$\propto$ Corriente")
    plt.xlabel("Potencial de membrana (V)")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show() 