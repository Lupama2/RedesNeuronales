#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 22-04-21
File: ej1_1erArquitectura.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description: 
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

SAVE_PATH = os.path.join(os.getcwd(), "Figuras", "ej1_1erArquitectura")
os.makedirs(SAVE_PATH, exist_ok=True)

def tanh(x_):
    return np.tanh(x_)

def gradTanh(x_):
    return 1 - np.tanh(x_)**2

def MSE(score, y_true):
    return (score-y_true)**2

def gradMSE(score, y_true):
    return 2*(score-y_true)

def predict(score):
    """El score es continuo, pero la prediccion es 1 o -1.
    Ponemos +-0.9 como threshold ya que usamos tanh como activacion"""
    if score >= 0.9:
        return 1
    elif score <= -0.9:
        return -1
    else:
        return score
        
def accuracy(score, y_true):
    "Devuelve 100% o 0% segun si le pego o no a la salida correcta"
    y_predict = predict(score)
    return (y_predict == y_true)*100.0

def fit(x_train, y_train, lr=1.0, epochs=10000, initialization='normal', verbose=False):

    # np.random.seed(seed)
    t_convergencia = -1

    dim = 2     # Dimension del problema
    clases = 1  # Cantidad de clases 
    nn = 2      # Numero de neuronas

    loss = np.array([])
    acc  = np.array([])

    # Primero, inicializo las componentes de las W1 y W1 (que tambien tienen el bias)
    if initialization == 'uniform':
        [[b1_1,b1_2],[w1_11,w1_12],[w1_21,w1_22]] = np.random.uniform(-1,1, size=(dim+1,nn))
        [b2_1, w2_1, w2_2] = np.random.uniform(-1,1, size=(nn+1,clases))
    elif initialization == 'normal':
        [[b1_1,b1_2],[w1_11,w1_12],[w1_21,w1_22]] = (1/np.sqrt(nn+1))*np.random.normal(0,1, size=(dim+1,nn))
        [b2_1, w2_1, w2_2] = (1/np.sqrt(clases+1))*np.random.normal(0,1, size=(nn+1,clases))
    else:
        raise("Invalid weight initialization: {}".format(initialization))

    for epoch in range(epochs):
        log_loss = 0.0
        log_acc = 0.0

        db1_1  = 0.0
        db1_2  = 0.0
        dw1_11 = 0.0
        dw1_12 = 0.0
        dw1_21 = 0.0
        dw1_22 = 0.0

        db2_1 = 0.0
        dw2_1 = 0.0
        dw2_2 = 0.0

        tic = time.time()

        for x, y in zip(x_train, y_train):

            # ----- Forward ----- #
            # Suma ponderada de la primer capa
            y1_1 = b1_1 + x[0]*w1_11 + x[1]*w1_21
            y1_2 = b1_2 + x[0]*w1_12 + x[1]*w1_22

            # Funcion de activacion de la primer capa
            s1_1 = tanh(y1_1)
            s1_2 = tanh(y1_2)

            # Suma ponderada de la segunda capa
            y2 = b2_1 + w2_1*s1_1 + w2_2*s1_2

            # Funcion de activacion de la segunda capa
            s2 = tanh(y2)

            # ----- Calculamos loss y accuracy ----- #
            log_loss += MSE(s2, y)
            log_acc  += accuracy(s2, y)
            # La regularizacion es para ñoños
            
            # ----- Backpropagation -----#
            # Voy a ir pisando la variable grad a medida que la use y ya no la necesite
            
            # Gradiente de la funcion de loss
            grad = gradMSE(s2, y)

            # Gradiente local de la activacion de la segunda capa (la de salida)
            grad2 = gradTanh(y2)

            # Actualizamos gradiente
            grad = grad * grad2

            # Aca ya podemos calcular los dW2
            db2_1 += (grad)
            dw2_1 += (grad * s1_1)
            dw2_2 += (grad * s1_2)

            # Seguimos mandando el gradiente para atras
            # Ahora el gradiente tiene 2 componentes
            grad = [grad*w2_1, grad*w2_2]

            # Calculamos gradiente local de la activacion de la primer capa
            grad1 = [gradTanh(y1_1), gradTanh(y1_2)]

            # Actualizamos el gradiente
            grad = [grad[0]*grad1[0], grad[1]*grad1[1]]

            # Y con esto, ya se pueden obtener todos los dW1
            db1_1  += (grad[0])
            db1_2  += (grad[1])
            dw1_11 += (grad[0] * x[0])
            dw1_12 += (grad[1] * x[0])
            dw1_21 += (grad[0] * x[1])
            dw1_22 += (grad[1] * x[1])

        # Actualizamos los pesos.
        # Divido por 4 porque en realidad esto ya tiene los dW correspondientes al bacht de 4
        b1_1  -= (lr * db1_1 * 0.25)
        b1_2  -= (lr * db1_2 * 0.25)
        w1_11 -= (lr * dw1_11 * 0.25)
        w1_12 -= (lr * dw1_12 * 0.25)
        w1_21 -= (lr * dw1_21 * 0.25)
        w1_22 -= (lr * dw1_22 * 0.25)

        b2_1 -= (lr * db2_1 * 0.25)
        w2_1 -= (lr * dw2_1 * 0.25)
        w2_2 -= (lr * dw2_2 * 0.25)
        
        tac = time.time()

        # Appendeamos los datos
        loss = np.append(loss, log_loss * 0.25)
        acc  = np.append(acc , log_acc * 0.25)

        # Voy a printear printeamos la evolucion
        if verbose == True:
            print("{}/{}".format(epoch,epochs), end=' ')
            print("{}s".format(int(tac-tic)), end=' ')
            print("loss: {:.5f}".format(loss[-1]), end=' ')
            print("acc: {:.2f}".format(acc[-1]), end='\n')

        # Como tiempo de convergencia, vamos a tomar la primer epoca en donde
        # la loss haya llegado a 0.1
        if((loss[-1] <= 0.1) and (t_convergencia==-1)):
            t_convergencia = epoch

    # Para el tiempo de convergencia, tambien vamos a pedir que la red haya llegado a 100 de acc
    if not (acc[-100:] == 100).all():
        t_convergencia = -1
    
    return [t_convergencia, loss, acc]


def diezCorridas():

    x_train = np.array([[-1,-1],[-1,1],[1,-1],[1,1]]).astype(np.float)
    y_train = np.array([1,-1,-1,1]).astype(np.float)

    figLoss = plt.figure()
    axLoss = figLoss.add_subplot(111)
    figAcc = plt.figure()
    axAcc = figAcc.add_subplot(111)

    for _ in range(10):

        t, loss, acc = fit(x_train, y_train, lr=1e-1,initialization='normal', epochs=2000, verbose=False)

        axLoss.plot(loss)
        axAcc.plot(acc)

    axLoss.set_xlabel("Epoca", fontsize=15)
    axLoss.set_ylabel("MSE", fontsize=15)
    figLoss.tight_layout()

    axAcc.set_xlabel("Epoca", fontsize=15)
    axAcc.set_ylabel("Accuracy", fontsize=15)
    figAcc.tight_layout()

    figLoss.savefig(os.path.join(SAVE_PATH, "Loss.pdf"), format='pdf')
    figAcc.savefig(os.path.join(SAVE_PATH, "Acc.pdf"), format='pdf')
    plt.show()

def meanTdeConvergencia():
    
    x_train = np.array([[-1,-1],[-1,1],[1,-1],[1,1]]).astype(np.float)
    y_train = np.array([1,-1,-1,1]).astype(np.float)

    t_convergencia = []
    count = 0

    while(len(t_convergencia) < 1000):

        t, _, _ = fit(x_train, y_train, lr=1e-1,initialization='normal', epochs=2000, verbose=False)

        if t != -1:
            t_convergencia.append(t)
        
        count += 1
        print(len(t_convergencia))
    
    t_convergencia = np.array(t_convergencia)
    t_medio = t_convergencia.mean()
    
    print("Tiempo medio: ", t_medio)
    print("# de corridas: ", count)
    return t_medio




if __name__ == "__main__":
    # diezCorridas()

    meanTdeConvergencia()