#Import libraries
import numpy as np

##########################################################
#                   x_inf
##########################################################

def m_inf(V):

    a_m = 0.1*(V + 40)/(1 - np.exp(-(V + 40)/10))
    b_m = 4*np.exp(-(V + 65)/18)

    return a_m/(a_m + b_m)

def h_inf(V):
        
    a_h = 0.07*np.exp(-(V + 65)/20)
    b_h = 1/(1 + np.exp(-(V + 35)/10))

    return a_h/(a_h + b_h)

def n_inf(V):
    
    a_n = 0.01*(V + 55)/(1 - np.exp(-(V + 55)/10))
    b_n = 0.125*np.exp(-(V + 65)/80)

    return a_n/(a_n + b_n)

def s_inf(V):
    return 0.5*(1 + np.tanh(V/5))


##########################################################
#                   tau_x
##########################################################

def tau_m(V):
        
    a_m = 0.1*(V + 40)/(1 - np.exp(-(V + 40)/10))
    b_m = 4*np.exp(-(V + 65)/18)

    return 1/(a_m + b_m)

def tau_h(V):

    a_h = 0.07*np.exp(-(V + 65)/20)
    b_h = 1/(1 + np.exp(-(V + 35)/10))

    return 1/(a_h + b_h)

def tau_n(V):
    
    a_n = 0.01*(V + 55)/(1 - np.exp(-(V + 55)/10))
    b_n = 0.125*np.exp(-(V + 65)/80)

    return 1/(a_n + b_n)

def tau_s(V):
    return 3

##########################################################
#                   I_syn
##########################################################

def I_syn(V, s, g_syn, V_syn):
    
    return - g_syn*s*(V - V_syn)

