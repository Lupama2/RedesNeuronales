#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 01-03-21
File: ej_01.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description: 
"""

import numpy as np
import matplotlib.pyplot as plt


T = 20.0 + 273.15     # K : Kelvin
k = 8.6e-5          # ev/K

def nernst(concentraciones):
  return k * T * np.log(concentraciones[1] / concentraciones[0])

if __name__ == "__main__":

    K = [430, 20]
    Na = [50, 440]
    Cl = [65, 550]

    for ion, conc in zip( ("K", "Na", "Cl") , (K, Na, Cl) ):
        print("{}: {}".format(ion, nernst(conc))) 
