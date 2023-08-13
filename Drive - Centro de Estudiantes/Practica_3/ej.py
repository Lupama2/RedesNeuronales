#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 23-03-21
File: ej.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description: 
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import hist
import seaborn as sns
sns.set(font_scale=1.5)


# Creamos carpeta para guardar los archivos
SAVE_PATH = os.path.join("Informe", "Figuras")
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# Abrimos los archivos
spikes = np.loadtxt("spikes.dat")
t, stimulus = np.loadtxt("stimulus.dat", unpack=True)

# En el estimulo tenemos un tiempo mas respecto a los spikes, lo tiramos.
t = t[:-1]
stimulus = stimulus[:-1]

############################
# 1
############################

# Vamos a calcular todos los tiempos entre dos spikes
isis = np.array([])     # Guardamos los tiempos entre spikes
t_spikes = []           # Guardamos los tiempos de los spikes

# Recorremos las pruebas y nos quedamos con las diferencias entre las
# posiciones que hay un 1 
for i in range(len(spikes)):
    isis = np.concatenate((isis, np.diff(t[spikes[i] > 0])))
    t_spikes.append(t[spikes[i] > 0])

t_spikes = np.array(t_spikes, dtype=object)

# Histograma de tiempos entre spikes
fig = plt.figure(figsize=(9,6))
hist(isis, bins='freedman', density=True)
plt.xlim(0, 50)
plt.xlabel("ISI [ms]")
plt.ylabel(r"$P(\tau)$")
plt.tight_layout()
fig_name = os.path.join(SAVE_PATH, "1_Distr_ISI.pdf")
plt.savefig(fig_name)
# plt.close()
plt.show()


fig, axs = plt.subplots(2,1,figsize=(9,6), gridspec_kw={'height_ratios': [1, 1.8]})
# Grafico del estimulo
axs[0].plot(t, stimulus)
axs[0].set_ylabel("S [dB]")
axs[0].set_xlim(-1, 1001)
axs[0].xaxis.set_visible(False)
# Rasterplot
plt.eventplot(t_spikes)
axs[1].set_xlabel("Tiempo [ms]")
axs[1].set_ylabel("Prueba")
axs[1].set_xlim(-1, 1001)
axs[1].set_ylim(0, 120)
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0.01)
fig_name = os.path.join(SAVE_PATH, "1_Rasterplot.pdf")
plt.savefig(fig_name)
# plt.close()
plt.show()

# Calculamos el CV
CV = isis.std() / isis.mean()

print("Coeficiente de variacion = {}".format(CV))
print("Intervalo inter-spikes promedio = {} ms".format(isis.mean()))
print("Tasa de disparo promedio = {} Hz".format(1.0 / isis.mean() * 1e3))

############################
# 2
############################

# Sacamos la cantidad de spikes total en cada experimento 
count = spikes.sum(axis=1)

plt.figure(figsize=(9,6))
hist(count, bins='freedman', density=True)
plt.xlabel(r"$N$")
plt.ylabel(r"$P(N)$")
plt.tight_layout()
fig_name = os.path.join(SAVE_PATH, "2_PN.pdf")
plt.savefig(fig_name)
plt.show()
# plt.close()

# Calculamos el factor de Fano
Fano = count.var() / count.mean()
print("Factor de Fano: {}".format(Fano))

############################
# 3
############################

rate = np.mean(spikes, axis=0) * 1e4
print("Tasa de disparo proemdio {} Hz".format(rate.mean()))

# fig, ax = plt.subplots()
plt.figure(figsize=(9,6))
plt.plot(t[1:], rate[1:], lw=1)
plt.ylabel(r"$r(t)$ [Hz]")
plt.xlabel("t [ms]")
plt.tight_layout()
# fig_name = os.path.join(SAVE_PATH, "3_rt_{}.pdf".format(binwidth))
fig_name = os.path.join(SAVE_PATH, "3_rt.pdf")
plt.savefig(fig_name)
# plt.close()
plt.show()

############################
# 4
############################

cant_spikes = spikes.sum(axis=0)
total = cant_spikes.sum()

filtro = []
tau_log = []

# Barremos tau de 0 a 100 ms
for tau in range(1001):
    suma = 0
    
    for i in range(len(cant_spikes)):    # Recorremos todos los spikes
        t_previo = i - tau
        if(t_previo >= 0):
            # Cant spikes * estimulo hace un tiempo tau
            suma += cant_spikes[i] * stimulus[t_previo]  
    
    filtro.append(suma / total)
    tau_log.append(tau / 10)

filtro = np.array(filtro)
tau_log = np.array(tau_log)

plt.figure(figsize=(9,6))
plt.plot(tau_log, filtro)
plt.xlabel(r"$\tau$ [ms]")
plt.ylabel(r"Spike-triggered average $C(\tau)$ [dB]")
plt.tight_layout()
fig_name = os.path.join(SAVE_PATH, "4_filtro.pdf")
plt.savefig(fig_name)
plt.show()
# plt.close()
        