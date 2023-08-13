#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 18-03-21
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
from numpy.core.fromnumeric import mean
from scipy.integrate import odeint
from scipy.signal import find_peaks
import seaborn as sns
sns.set()

# Creamos carpeta para guardar los archivos
SAVE_PATH = os.path.join("Figuras", "ej_01")
if not os.path.exists(SAVE_PATH):
    os.makedirs(os.path.join(SAVE_PATH,"Excitatorio"))
    os.makedirs(os.path.join(SAVE_PATH,"Inhibitorio"))

# Constantes
V_Na = 50       # mV
V_K = -77       # mV
V_l = -54.4     # mV
g_Na = 120      # mS/cm²
g_K = 36        # mS/cm²
g_l = 0.3       # mS/cm²
C = 1           # Capacitancia de la membrana \mu F / cm²

# Funciones auxiliares para el modelo de Hogdkin Huxley
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

def s_inf(V):
    return 0.5 * (1 + np.tanh(0.2*V))

# Modelo Hogdkin Huxley, es una adaptacion que permite algunas aproximaciones que no se usan en este ejercicio
def Hogdkin_Huxley(z, t, g_syn, V_pre, I_ext=0, V_syn=0, tauS=3, m_inf=False, hn_cte=False):
    V, m, h, n, s = z

    # dsdt
    dsdt = (s_inf(V_pre) - s) / tauS

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
    I_syn = -g_syn * s * (V - V_syn)

    dVdt = I_ext + I_syn - I_Na - I_K - I_L
    dVdt /= C

    return [dVdt, dmdt, dhdt, dndt, dsdt]

def simulacion(n_iter=5000, t_max=5000, V_syn=0, I_ext=1e1, tauS=3):

    g_syn_list = np.linspace(0, 2, 21)              # Rango de g_syn que queremos estudiar

    f_log = []
    shift_log = []

    for g_syn in g_syn_list:
        
        # Se puede jugar mucho con las condiciones iniciales, pero me parecio mas razonable usar las funciones x_inf
        # ya que son valores que pueden alcanzarse en algun momento
        z_0_1 = [-10, x_inf(a_m(-10), b_m(-10)), x_inf(a_h(-10), b_h(-10)), x_inf(a_n(-10), b_n(-10)), s_inf(-10)]
        z_0_2 = [-30, x_inf(a_m(-30), b_m(-30)), x_inf(a_h(-30), b_h(-30)), x_inf(a_n(-30), b_n(-30)), s_inf(-30)]
        
        print("g_syn: {:.2f}".format(g_syn), end=" ")

        t = np.linspace(0, t_max, n_iter+1)
        
        V_1, V_2 = np.zeros_like(t), np.zeros_like(t)   # Aca vamos a ir guardando los valores de la simulacion
        m_1, m_2 = np.zeros_like(t), np.zeros_like(t)
        h_1, h_2 = np.zeros_like(t), np.zeros_like(t)
        n_1, n_2 = np.zeros_like(t), np.zeros_like(t)
        s_1, s_2 = np.zeros_like(t), np.zeros_like(t)
        
        V_1[0], m_1[0], h_1[0], n_1[0], s_1[0] = z_0_1      # El primer valor en la condicion inicial
        V_2[0], m_2[0], h_2[0], n_2[0], s_2[0] = z_0_2

        # Cuando a la funcion odeint le tenemos que pasar argumentos, lo que tenemos que hacer es ir integrando para cada t nuevo
        # tomando como condicion inicial el resultado de la iteracion anterior
        for i in range(1,n_iter+1):
            t_span = [t[i - 1], t[i]]

            # Hacemos una iteracion
            z_1 = odeint(Hogdkin_Huxley, z_0_1, t_span, args=(g_syn, z_0_2[0], I_ext, V_syn, tauS,))
            z_2 = odeint(Hogdkin_Huxley, z_0_2, t_span, args=(g_syn, z_0_1[0], I_ext, V_syn, tauS,))
            
            # El resultado son las CI de la siguiente iteracion
            z_0_1 = z_1[1]
            z_0_2 = z_2[1]

            V_1[i], m_1[i], h_1[i], n_1[i], s_1[i] = z_0_1
            V_2[i], m_2[i], h_2[i], n_2[i], s_2[i] = z_0_2
        
        # Ploteo el resultado y lo guardo, son muchas imagenes asi que lo dejo comentado
        plotear(t, V_1, V_2, g_syn, V_syn)

        # Calculamos la tasa de disparo y el desfasaje
        # Para que esto este bien, la resolucion temporal tiene que ser bastante guanaca
        f, shift = tasaDeDisparoYDesfasaje(t, V_1, V_2, t_max/n_iter, V_syn)

        f_log.append(f)
        shift_log.append(shift)
        
    
    # Guardo tasa de disparo y shifteo
    f_log = np.array(f_log)
    shift_log = np.array(shift_log)

    # Cambiamos el nombre del archivo segun estemos en el caso excitatorio o inhibitorio
    if V_syn == 0:
        file_name = os.path.join(SAVE_PATH, "Excitatorio.npz")
    else:
        file_name = os.path.join(SAVE_PATH, "Inhibitorio.npz")
    
    np.savez(file_name, g_syn=g_syn_list, f=f_log, shift=shift_log)

# Funcion que nos calcula la tasa de disparo y el desfasaje
def tasaDeDisparoYDesfasaje(t, V_1, V_2, deltaT, V_syn):
    # Calculamos los picos, capaz es mejor hacerlo con los cruces en cero, pero paja aprender ahora
    peaks_1, _ = find_peaks(V_1, height=-10)
    peaks_2, _ = find_peaks(V_2, height=-10)

    # Nos quedamos con los ultimos, porque son los que seguro pertenecen al regimen transitorio
    peaks_1 = peaks_1[-20:]
    peaks_2 = peaks_2[-20:]

    # Calculamos la diferencia de tiempos entre picos
    T1 = (t[peaks_1[1:]] - t[peaks_1[:-1]])
    T2 = (t[peaks_2[1:]] - t[peaks_2[:-1]])

    # Promediamos y sacamos la frecuencia en Hz
    T = np.concatenate((T1, T2))
    T = T.mean()         
    f = 1.0 / (T * 1e-3)  # A segundos

    # Sacamos el desfasaje calculando la fraccion respecto al periodo
    T_diff = abs((t[peaks_1] - t[peaks_2]).mean())      # Capaz el abs no hace falta
    shift = ((T_diff % T) / T) * 2 * np.pi    

    print("f={:.2f}, s={:.2f}".format(f,shift))

    return [f, shift]

# Arma graficos con un nivel de obsesividad importante, pero le falta para que me termine de gustar
def plotsBarridos():

    file_exc = os.path.join(SAVE_PATH, "Excitatorio.npz")
    file_inh = os.path.join(SAVE_PATH, "Inhibitorio.npz")

    data_exc = np.load(file_exc)
    data_inh = np.load(file_inh)

    plt.figure()
    ax = plt.subplot(111)
    ax.plot(data_exc['g_syn'], data_exc['f'], '--.', label="Excitatorio")
    ax.plot(data_inh['g_syn'], data_inh['f'], '--.', label="Inhibitorio")
    ax.set_ylabel("Frecuencia [Hz]")
    ax.set_xlabel(r"$g_{syn}$ [$\frac{mS}{cm^{2}}$]")
    plt.legend(loc='best')
    plt.tight_layout()
    file_name = os.path.join(SAVE_PATH,"Tasa_de_Disparo.pdf")
    plt.savefig(file_name, format='pdf')
    # plt.show()
    plt.close()

    plt.figure()
    ax = plt.subplot(111)
    ax.plot(data_exc['g_syn'], data_exc['shift'], '--o', label="Excitatorio")
    ax.plot(data_inh['g_syn'], data_inh['shift'], '--o', label="Inhibitorio")
    ax.set_ylabel("Desfasaje [rad]")
    ax.set_xlabel(r"$g_{syn}$ [$\frac{mS}{cm^{2}}$]")
    ax.set_yticks([0, 0.25*np.pi, 0.5*np.pi, 0.75*np.pi, np.pi])
    ax.set_yticklabels(["$0$", r"$\frac{1}{4}\pi$",
                     r"$\frac{1}{2}\pi$", r"$\frac{3}{4}\pi$", r"$\pi$"])
    plt.legend(loc='best')
    plt.tight_layout()
    file_name = os.path.join(SAVE_PATH,"Desfasaje.pdf")
    plt.savefig(file_name, format='pdf')
    # plt.show()
    plt.close()


# Arma graficos con un nivel de obsesividad importante, pero le falta para que me termine de gustar
def plotear(t, V_1, V_2, g_syn, V_syn):

    if V_syn == 0:
        file_name = os.path.join(SAVE_PATH, "Excitatorio", "{:.2f}.pdf".format(g_syn))
        file_path = os.path.join(SAVE_PATH, "Excitatorio_g=1.npz")
    else:
        file_name = os.path.join(SAVE_PATH, "Inhibitorio", "{:.2f}.pdf".format(g_syn))
        file_path = os.path.join(SAVE_PATH, "Inhibitorio_g=1.npz")
    
    # Guardo los datos para g_syn = 1 asi despues hago un grafico lindo
    if(g_syn == 1):
        np.savez(file_path, t=t, V_1=V_1, V_2=V_2)

    plt.plot(t, V_1, label="Neurona 1")
    plt.plot(t, V_2, label="Neurona 2")
    plt.ylabel("V [mV]")
    plt.xlabel("Tiempo [ms]")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, fancybox=True, shadow=False)
    plt.tight_layout()
    plt.savefig(file_name, format='pdf')
    plt.close()
    # plt.show()

    plt.plot(t[-len(V_1)//10:], V_1[-len(V_1)//10:], label="Neurona 1")
    plt.plot(t[-len(V_2)//10:], V_2[-len(V_2)//10:], label="Neurona 2")
    plt.ylabel("V [mV]")
    plt.xlabel("Tiempo [ms]")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, fancybox=True, shadow=False)
    plt.tight_layout()
    plt.savefig(file_name + "aux.pdf", format='pdf')
    plt.close()
    # plt.show()

# Arma graficos con un nivel de obsesividad importante, pero le falta para que me termine de gustar
def ploteoVoltaje():

    file_exc = os.path.join(SAVE_PATH, "Excitatorio_g=1.npz")
    file_inh = os.path.join(SAVE_PATH, "Inhibitorio_g=1.npz")

    data_exc = np.load(file_exc)
    data_inh = np.load(file_inh)

    # Final Excitatorio
    plt.plot(data_exc['t'][-10001:], data_exc['V_1'][-10001:], label="Neurona 1")
    plt.plot(data_exc['t'][-10001:], data_exc['V_2'][-10001:], label="Neurona 2")
    plt.ylabel("V [mV]")
    plt.xlabel("Tiempo [ms]")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, fancybox=True, shadow=False)
    plt.tight_layout()
    plt.savefig("Excitatorio_final.pdf", format='pdf')
    # plt.close()
    plt.show()

    # Principio Excitatorio
    plt.plot(data_exc['t'][:10001], data_exc['V_1'][:10001], label="Neurona 1")
    plt.plot(data_exc['t'][:10001], data_exc['V_2'][:10001], label="Neurona 2")
    plt.ylabel("V [mV]")
    plt.xlabel("Tiempo [ms]")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, fancybox=True, shadow=False)
    plt.tight_layout()
    plt.savefig("Excitatorio_inicio.pdf", format='pdf')
    # plt.close()
    plt.show()

    # Final Inhibitorio
    plt.plot(data_inh['t'][-10001:], data_inh['V_1'][-10001:], label="Neurona 1")
    plt.plot(data_inh['t'][-10001:], data_inh['V_2'][-10001:], label="Neurona 2")
    plt.ylabel("V [mV]")
    plt.xlabel("Tiempo [ms]")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, fancybox=True, shadow=False)
    plt.tight_layout()
    plt.savefig("Inhibitorio_final.pdf", format='pdf')
    # plt.close()
    plt.show()

    # Principio Inhibitorio
    plt.plot(data_inh['t'][:10001], data_inh['V_1'][:10001], label="Neurona 1")
    plt.plot(data_inh['t'][:10001], data_inh['V_2'][:10001], label="Neurona 2")
    plt.ylabel("V [mV]")
    plt.xlabel("Tiempo [ms]")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, fancybox=True, shadow=False)
    plt.tight_layout()
    plt.savefig("Inhibitorio_inicio.pdf", format='pdf')
    # plt.close()
    plt.show()

if __name__ == "__main__":
    simulacion(n_iter=250000, t_max=2500, I_ext=1e1)
    simulacion(n_iter=250000, t_max=2500, I_ext=1e1, V_syn=-80)
    plotsBarridos()
    ploteoVoltaje()



# f(x)=a+b*tanh(c*(x-x0))
# f(x)=1*(1+tanh(c*(x-x0)))
# a*(1+tanh(c*(x-x0)))