#Import libraries
import numpy as np

#Import functions
from functions import m_inf, h_inf, n_inf, s_inf, tau_m, tau_h, tau_n, tau_s, I_syn

#Import parameters
from parameters import C_hat, g_K_adim, g_Na_adim, g_L_adim, V_K, V_Na, V_L

def derivada(t, y, I_ext, g_syn, V_syn):
    '''

    C_hat = C / g_hat : [ms = mili segundos]
    I_ext : [muA/cm2]
    V: [mV]
    
    Derivada
    y[0]: V1
    y[1]: m1
    y[2]: h1
    y[3]: n1
    y[4]: s1
    y[5]: V2
    y[6]: m2
    y[7]: h2
    y[8]: n2
    y[9]: s2
    
    '''

    #Def derivative vector
    dydt = np.empty(10)
    N_eq = 5 #ec. por neurona

    #Asigno variables
    V1 = y[0]; m1 = y[1]; h1 = y[2]; n1 = y[3]; s1 = y[4]
    V2 = y[5]; m2 = y[6]; h2 = y[7]; n2 = y[8]; s2 = y[9]

    #Eq of charge conservation
    dydt[0] = (1/C_hat) *(I_ext + I_syn(V1, s1, g_syn, V_syn) - g_K_adim*n1**4*(V1 - V_K) - g_Na_adim*m1**3*h1*(V1 - V_Na) - g_L_adim*(V1 - V_L))

    dydt[0 + N_eq] = (1/C_hat) *(I_ext + I_syn(V2, s2, g_syn, V_syn) - g_K_adim*n2**4*(V2 - V_K) - g_Na_adim*m2**3*h2*(V2 - V_Na) - g_L_adim*(V2 - V_L))

    #Eq's m, h, n
    dydt[1] = (m_inf(V1) - m1)/tau_m(V1)
    dydt[2] = (h_inf(V1) - h1)/tau_h(V1)
    dydt[3] = (n_inf(V1) - n1)/tau_n(V1)
    dydt[4] = (s_inf(V2) - s1)/tau_s(V1)

    dydt[1 + N_eq] = (m_inf(V2) - m2)/tau_m(V2)
    dydt[2 + N_eq] = (h_inf(V2) - h2)/tau_h(V2)
    dydt[3 + N_eq] = (n_inf(V2) - n2)/tau_n(V2)
    dydt[4 + N_eq] = (s_inf(V1) - s2)/tau_s(V2)

    return dydt