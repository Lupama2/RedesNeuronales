# # Práctica 5
# 
# date: 30/10/2023  
# File: Chehade_practica5.ipynb
# Author : Pablo Naim Chehade   
# Email: pablo.chehade.villalba@gmail.com  
# GitHub: https://github.com/Lupama2  

# ## Ejercicio 1
#Import libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#Parameters
w_ini_max = 0.01 #valor máximo del peso inicial

#Def correlation matrix
C = np.array([[2, 1, 1, 1],
              [1, 2, 1, 1],
              [1, 1, 2, 1],
              [1, 1, 1, 2]])

C_12 = np.array([[1.309, 0.309, 0.309, 0.309],
                [0.309, 1.309, 0.309, 0.309],
                [0.309, 0.309, 1.309, 0.309],
                [0.309, 0.309, 0.309, 1.309]])

#Calculo los autovalores y autovectores de C
eigvals, eigvecs = np.linalg.eig(C)
#Calculo el autovector con el mayor autovalor
eigvec_max = eigvecs[:, np.argmax(eigvals)]
#Imprimo
print('El autovector con el mayor autovalor es: ', eigvec_max)

def ejemplo():
    #Genero 4 nros con distribución normal
    z = np.random.randn(4)
    #Multiplico por C_12
    x = np.dot(C_12, z)
    return x

def inicializacion_pesos():
    #Genero 4 nros con distribución uniforme
    w = np.random.uniform(-w_ini_max, w_ini_max, 4)
    return w
    

def actualizacion_pesos(w, eta):
    #Genero un ejemplo
    xi = ejemplo()
    #Calculo la salida
    V = np.dot(w, xi)
    #Calculo delta_w
    delta_w = eta*V*(xi - V*w)
    return w + delta_w

def train(N_train, eta):
    #Genero la matriz de pesos
    w_matrix = np.zeros([N_train, 4])
    #Genero los pesos iniciales
    w = inicializacion_pesos()
    w_matrix[0] = w
    #Entreno N_train pasos
    for i in range(1,N_train):
        w = actualizacion_pesos(w, eta)
        w_matrix[i] = w
    return w_matrix

#Entreno una red
N_train = 5000

#Promedio w_matrix y delta_w para cada índice en una caja de ancho M con np.convolve
def plt_w(M, eta):
    w_matrix = train(N_train, eta)
    #Calculo los delta_w
    delta_w = w_matrix[1:] - w_matrix[:-1]

    w_matrix_mean = np.empty([N_train - M + 1, 4])
    delta_w_mean = np.empty([N_train - M, 4])
    for i in range(4):
        w_matrix_mean[:, i] = np.convolve(w_matrix[:, i], np.ones(M)/M, mode='valid')
        delta_w_mean[:, i] = np.convolve(delta_w[:, i], np.ones(M)/M, mode='valid')

    # Grafico los pasos en el tiempo y en otra gráfica, los cambios delta w_j entre pasos
    fig, ax = plt.subplots(3, 1, figsize=(9.5, 5.5), sharex=True)
    fig.subplots_adjust(hspace=0.1)

    ax[0].plot(np.abs(w_matrix_mean))
    # ax[0].set_xlabel('Pasos')
    ax[0].set_ylabel('$|w_j|$')
    ax[0].labels = ['1', '2', '3', '4']
    #Grafico las labels arriba fuera del gráfico
    ax[0].legend(ax[0].labels, loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=4)
    ax[0].set_ylim(0, 1)
    # ax[0].set_xscale("log")
    ax[0].grid()

    ax[1].plot(delta_w_mean)
    # ax[1].set_xlabel('Pasos')
    ax[1].set_ylabel('$\Delta w_j$')
    ax[1].labels = ['1', '2', '3', '4']
    # ax[1].legend(ax[1].labels)
    ax[1].grid()

    ax[2].plot(np.abs(np.abs(w_matrix_mean) - np.abs(eigvec_max)))
    ax[2].set_xlabel('Pasos')
    ax[2].set_ylabel('Error')
    ax[2].labels = ['1', '2', '3', '4']
    #Grafico las labels arriba fuera del gráfico
    # ax[2].legend(ax[2].labels)
    ax[2].set_yscale("log")
    ax[2].grid()

    plt.show()

#Hago gráficos para el informe
plt_w(1, 0.001)
plt.savefig('Informe/ej1_fig1.png', bbox_inches='tight', dpi=300)

plt_w(200, 0.001)
plt.savefig('Informe/ej1_fig2.png', bbox_inches='tight', dpi=300)

plt_w(1, 0.01)
plt.savefig('Informe/ej1_fig3.png', bbox_inches='tight', dpi=300)

# ## Ejercicio 2
#Def parámetros globales
r1 = 0.9
r2 = 1.1
N = 2
M = 10
w_ini_max = 0.1

#Def seed
np.random.seed(0)

#Genero pares de nros aleatorios (x,y) con distribución uniforme sobre un círculo
def generar_datos(n):
    r_array = np.empty([n,2])
    contador = 0
    while(contador < n):
        #Genero nros aleatorios con distribución uniforme entre 0 y 1
        x = (np.random.rand()*2 - 1)*r2
        y = (np.random.rand()*2 - 1)*r2
        #Me fijo si pertenece al anillo
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y,x)
        if r1 < r < r2 and 0 < theta < np.pi:
            r_array[contador] = np.array([x,y])
            contador += 1
    return r_array

def Vecindad(i,i_star,sigma):
    return np.exp(-((i-i_star)**2)/(2*sigma**2))

def algoritmo_de_Kohonen(w, x, sigma, eta):
    #Calculo la neurona ganadora
    i_star = np.argmin(np.linalg.norm(w-x,axis=1))
    #Actualizo los pesos
    delta_w = eta*Vecindad(np.arange(M),i_star,sigma).reshape(M,1)*(x-w)
    w += delta_w
    return w

def entrenamiento_Kohonen(N_train, sigma, eta):
    #Inicializo pesos de la red
    w = np.random.rand(M,N)*w_ini_max
    w_matriz = np.empty([N_train,M,N])
    #Genero los datos
    r_array = generar_datos(N_train)
    #Entreno la red
    for i in range(N_train):
        w = algoritmo_de_Kohonen(w, r_array[i], sigma, eta)
        #Guardo el valor
        w_matriz[i] = w
    return w_matriz

def plt_entrenamiento_Kohonen(N_train, sigma, eta):
    #Redef seed
    np.random.seed(0)
    w_matriz = entrenamiento_Kohonen(N_train, sigma, eta)

    #Graph los pesos en el tiempo
    fig, ax = plt.subplots(figsize = (5,4))
    color_vec = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]

    #Grafico los pesos
    for i in range(M):
        ax.plot(w_matriz[0:,i,0],w_matriz[0:,i,1],'o-',color=color_vec[i], alpha = 0.1)
    for i in range(M):
        #Pinto de otro color el último punto
        ax.plot(w_matriz[-1,i,0],w_matriz[-1,i,1],'o-',color="k", alpha = 1)
        ax.text(w_matriz[-1,i,0],w_matriz[-1,i,1],str(i),color="k",alpha=1)

    #Grafico un anillo entre r1 y r2 con theta entre 0 y pi
    theta = np.linspace(0,np.pi,100)
    ax.plot(r1*np.cos(theta),r1*np.sin(theta),'k')
    ax.plot(r2*np.cos(theta),r2*np.sin(theta),'k')
    ax.plot([-r2,-r1],[0,0],'k'); ax.plot([r1,r2],[0,0],'k')
    ax.set_xlim(-1.2,1.2)
    ax.set_ylim(-0.1,1.2)

    #Quito los tickslabels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    #Agrego arriba a la izquierda el valor de N_train
    ax.text(-1.1,1.05,'$\mathrm{N_{train}}$ = '+str(N_train),color="k",alpha=1)

    plt.show()
    return

#Guardo gráficos particulares y los guardo
# plt_entrenamiento_Kohonen(N_train, sigma, eta)

#Figura 1
sigma = 1
eta = 0.1

plt_entrenamiento_Kohonen(10, sigma, eta)
plt.savefig('Informe/ej2_fig1_1.png', bbox_inches='tight', dpi=300)

#Descomentar las siguientes líneas para hacer la variación de N_train, eta y sigma
# plt_entrenamiento_Kohonen(200, sigma, eta)
# plt.savefig('Informe/ej2_fig1_2.png', bbox_inches='tight', dpi=300)

# plt_entrenamiento_Kohonen(750, sigma, eta)
# plt.savefig('Informe/ej2_fig1_3.png', bbox_inches='tight', dpi=300)

# plt_entrenamiento_Kohonen(1750, sigma, eta)
# plt.savefig('Informe/ej2_fig1_4.png', bbox_inches='tight', dpi=300)

# #Figura 2
# sigma = 2
# eta = 0.1

# plt_entrenamiento_Kohonen(10, sigma, eta)
# plt.savefig('Informe/ej2_fig2_1.png', bbox_inches='tight', dpi=300)

# plt_entrenamiento_Kohonen(200, sigma, eta)
# plt.savefig('Informe/ej2_fig2_2.png', bbox_inches='tight', dpi=300)

# plt_entrenamiento_Kohonen(750, sigma, eta)
# plt.savefig('Informe/ej2_fig2_3.png', bbox_inches='tight', dpi=300)

# plt_entrenamiento_Kohonen(50000, sigma, eta)
# plt.savefig('Informe/ej2_fig2_4.png', bbox_inches='tight', dpi=300)

# #Figura 3
# sigma = 1
# eta = 0.01

# plt_entrenamiento_Kohonen(10, sigma, eta)
# plt.savefig('Informe/ej2_fig3_1.png', bbox_inches='tight', dpi=300)

# plt_entrenamiento_Kohonen(200, sigma, eta)
# plt.savefig('Informe/ej2_fig3_2.png', bbox_inches='tight', dpi=300)

# plt_entrenamiento_Kohonen(750, sigma, eta)
# plt.savefig('Informe/ej2_fig3_3.png', bbox_inches='tight', dpi=300)

# plt_entrenamiento_Kohonen(6000, sigma, eta)
# plt.savefig('Informe/ej2_fig3_4.png', bbox_inches='tight', dpi=300)




