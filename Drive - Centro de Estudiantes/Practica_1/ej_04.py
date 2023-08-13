#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 18-03-21
File: ej_04.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description: 
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import find_peaks
import seaborn as sns
sns.set()

# Constantes
V_Na = 50       # mV
V_K = -77       # mV
V_l = -54.4     # mV
g_Na = 120      # mS/cm²
g_K = 36        # mS/cm²
g_l = 0.3       # mS/cm²
C = 1           # Capacitancia de la membrana \mu F / cm²

def a_m(V):
    aux = 0.1*V + 4.0 
    return aux / (1 - np.exp(-aux) )

def b_m(V):
    return 4 * np.exp(-(V+65.0) / 18.0)

def a_h(V):
    return 0.07 * np.exp(-(V+65.0) / 20.0)

def b_h(V):
    return 1.0 / (1.0 + np.exp(-0.1*(V+35.0)) )

def a_n(V):
    aux = 0.1 * (V + 55)
    return 0.1 * aux / (1 - np.exp(-aux) )

def b_n(V):
    return 0.125 * np.exp(-0.0125*(V+65.0))

def tau(a, b):
    return 1.0/(a+b)

def x_inf(a, b):
    return a/(a+b)

def Hogdkin_Huxley(z, t, I=0, m_inf=False, hn_cte=False):
    V, m, h, n = z

    # dmdt
    a, b = a_m(V), b_m(V)
    if m_inf or hn_cte:
        dmdt = 0    # Aca la derivada no importa, pero si pongo None da mal no se xq
        m = x_inf(a,b)
    else:
        dmdt = (x_inf(a,b) - m) / tau(a,b)
    
    # dhdt
    a, b = a_h(V), b_h(V)
    dhdt = (x_inf(a,b) - h) / tau(a,b)
    
    # dndt
    if hn_cte:
        dndt = -dhdt
        # n = 0.8 - h
    else:
        a, b = a_n(V), b_n(V)
        dndt = (x_inf(a,b) - n) / tau(a,b)
    
    # dVdt
    I_Na = g_Na * np.power(m,3) * h * (V - V_Na)
    I_K = g_K * np.power(n,4) * (V - V_K)
    I_L = g_l * (V - V_l)
    dVdt = I - I_Na - I_K - I_L
    dVdt /= C

    return [dVdt, dmdt, dhdt, dndt]

def simulacion(n_iter = 2000, m_inf=False, hn_cte=False):
    # z_0 = [-70, 0.1, 0.5, 0.3]  # De donde saco esto Alvaro?
    z_0 = [-70, x_inf(a_m(-70), b_m(-70)), 0.5, 0.3]  # De donde saco esto Alvaro?
    
    t = np.linspace(0, 200, n_iter)
    V = np.zeros_like(t)
    m = np.zeros_like(t)
    h = np.zeros_like(t)
    n = np.zeros_like(t)
    V[0], m[0], h[0], n[0] = z_0

    I_ext = np.zeros_like(t)
    I_ext[n_iter // 4: 3 * n_iter // 4] = 10    # Cuando encajamos corriente

    for i in range(1, n_iter):
        t_span = [t[i - 1], t[i]]
        z = odeint(Hogdkin_Huxley, z_0, t_span, args=(I_ext[i], m_inf, hn_cte))
        z_0 = z[1]
        V[i], m[i], h[i], n[i] = z_0
        # import ipdb; ipdb.set_trace(context=15)  # XXX BREAKPOINT
    
    if m_inf or hn_cte:
        m = x_inf(a_m(V), b_m(V))
    
    # import ipdb; ipdb.set_trace(context=15)  # XXX BREAKPOINT

    I_Na = g_Na * m**3 * h * (V - V_Na)
    I_K = g_K * n**4 * (V - V_K)
    I_L = g_l * (V - V_l)

    plotear(t, V, I_ext, I_Na, I_K, I_L, m, h, n)

def barridoI(n_iter = 2000, t_max=200, I_iter=10, m_inf=False, hn_cte=False):
    z_0 = [-70, x_inf(a_m(-70), b_m(-70)), 0.5, 0.3]  # De donde saco esto Alvaro?

    t = np.linspace(0, t_max, n_iter)
    I_max_list = np.linspace(4, 12, I_iter)
    I_max_list = np.concatenate([I_max_list, I_max_list[:-1][::-1]])

    V_log = []
    m_log = []
    h_log = []
    n_log = []

    for I_max in I_max_list:
        print("I: {}".format(I_max))
        V = np.zeros_like(t)
        m = np.zeros_like(t)
        h = np.zeros_like(t)
        n = np.zeros_like(t)
        V[0], m[0], h[0], n[0] = z_0

        z = odeint(Hogdkin_Huxley, z_0, t, args=(I_max,))

        V = z[:,0]
        m = z[:,1]
        h = z[:,2]
        n = z[:,3]

        V_log.append(V)
        m_log.append(m)
        h_log.append(h)
        n_log.append(n)

        # Para la siguiente iteracion dejo como CI el estado final de esta iteracino
        z_0 = z[-1]
    
    f_list = []
    for V in V_log:
        times, _ = find_peaks(V, height=-20)
        f = 0
        if len(times) > 1:
            isi = (times[1: ] - times[: -1]) * t_max / n_iter
            f = 1 / np.mean(isi) * 1e3  # El tiempo en segundos
        f_list.append(f)
    
    plt.plot(I_max_list, f_list)
    u = np.diff(I_max_list)
    v = np.diff(f_list)
    x = np.array(I_max_list[: -1]) + u/2
    y = np.array(f_list[: -1]) + v/2
    norm = np.sqrt(u**2 + v**2) 
    plt.quiver(x, y, u / norm, v / norm, angles="xy", alpha=0.5)
    plt.ylabel("Tasa de disparo (Hz)")
    plt.xlabel("Corriente (μA cm$^{-2}$)")
    plt.tight_layout()
    plt.show()




    


def plotear(t, V, I_ext, I_Na, I_K, I_L, m, h, n):
    plt.plot(t, V, label="Membrana")
    plt.ylabel("Tensión de membrana (mV)")
    plt.xlabel("Tiempo (ms)")
    # sns.despine(trim=True)
    plt.legend(loc='best')
    plt.show()
    # plt.close()

    plt.plot(t, I_ext, label="Externa")
    plt.ylabel(r"Corriente externa (μA cm$^{-2}$)")
    plt.xlabel("Tiempo (ms)")
    # sns.despine(trim=True)
    plt.legend(loc='best')
    plt.show()
    # plt.close()

    plt.plot(t, I_Na, label="Na")
    plt.plot(t, I_K, label="K")
    plt.plot(t, I_L, label="L")
    plt.ylabel(r"Corriente (μA cm$^{-2}$)")
    plt.xlabel("Tiempo (ms)")
    # sns.despine(trim=True)
    plt.legend(loc='best')
    plt.show()
    # plt.close()

    plt.plot(V, I_Na, label="Na")
    plt.plot(V, I_K, label="K")
    plt.plot(V, I_L, label="L")
    plt.ylabel(r"Corriente (μA cm$^{-2}$)")
    plt.xlabel("Tensión (mV)")
    # sns.despine(trim=True)
    plt.legend(loc='best')
    plt.show()
    # plt.close()

    plt.plot(t, m, label="m")
    plt.plot(t, h, label="h")
    plt.plot(t, n, label="n")
    plt.plot(t, n+h, label="n+h")
    plt.ylabel("Variables de compuerta")
    plt.xlabel("Tiempo (ms)")
    # sns.despine(trim=True)
    plt.legend(loc='best')
    plt.show()
    # plt.close()

    plt.plot(V, m, label="m")
    plt.plot(V, h, label="h")
    plt.plot(V, n, label="n")
    plt.ylabel("Variables de compuerta")
    plt.xlabel("Tensión (mV)")
    # sns.despine(trim=True)
    plt.legend(loc='best')
    plt.show()
    # plt.close()


if __name__ == "__main__":
    # simulacion()
    # simulacion(m_inf=True)
    # simulacion(hn_cte=True)

    barridoI()