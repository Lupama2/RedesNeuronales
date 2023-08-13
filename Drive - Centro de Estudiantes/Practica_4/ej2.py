#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 25-04-21
File: ej2.py
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
import itertools
import seaborn as sns
sns.set()

SAVE_PATH = os.path.join(os.getcwd(), "Figuras", "ej2")
os.makedirs(SAVE_PATH, exist_ok=True)


def lossMSExor(scores, y_true):
    return ((scores-y_true)**2).mean()

def gradientMSExor(scores, y_true):
    return 2*(scores-y_true)/len(y_true)

def acc_XOR(scores, y_true):
    S = np.copy(scores)
    S[S > 0.9] = 1
    S[S < -0.9] = -1
    return (S == y_true).mean() * 100

def Tanh( x_):
    return np.tanh(x_)

def gradTanh(x_):
    return 1 - np.tanh(x_)**2

def fit(x_train, y_train, N2, size_bacht=32, lr=1e-1, landa=1e-2, epochs=200, initialization='normal', loss_f=lossMSExor, grad_f=gradientMSExor,
                    act_1=Tanh, grad_1=gradTanh, act_2=Tanh, grad_2=gradTanh, accuracy=acc_XOR, verbose=True):
    
    n, dim = x_train.shape
    clases = 1
    nn = N2

    idx = np.arange(n)

    loss = np.array([])
    acc  = np.array([])

    if initialization == 'uniform':
        W1 = np.random.uniform(-1,1, size=(dim+1, nn))
        W2 = np.random.uniform(-1,1, size=(nn+1, clases))
    elif initialization == 'normal':
        W1 = (1/np.sqrt(nn+1))*np.random.normal(0,1, size=(dim+1,nn))
        W2 = (1/np.sqrt(clases+1))*np.random.normal(0,1, size=(nn+1,clases))
    else:
        raise("Invalid weight initialization: {}".format(initialization))


    x_train = np.hstack((np.ones((len(x_train),1)), x_train))

    n_bacht = int(len(x_train)/size_bacht)

    for e in range(epochs):
        log_loss = 0
        log_acc  = 0

        np.random.shuffle(idx)  # Mezclo los indices

        tic = time.time()

        for i in range(n_bacht):

            bIdx = idx[size_bacht*i: size_bacht*(i+1)]

            x_bacht = x_train[bIdx]
            y_bacht = y_train[bIdx]

            # Capa 1
            Y1 = np.dot(x_bacht, W1)
            S1 = act_1( Y1 )

            # Capa 2
            S1 = np.hstack((np.ones((len(S1),1)), S1))
            Y2 = np.dot(S1, W2)
            S2 = act_2( Y2 )

            # Regularizacion
            reg1 = np.sum(W1 * W1)
            reg2 = np.sum(W2 * W2)
            reg  = 0.5 * landa * (reg1 + reg2)

            log_loss += loss_f(S2, y_bacht) + reg
            log_acc  += accuracy(S2, y_bacht)


            # Ahora arranca el backpropagation
            grad = grad_f(S2, y_bacht)  # Este gradiente ya tiene hecho el promedio
            grad2 = grad_2( Y2 )

            grad = grad * grad2

            # Capa 2
            dW2 = np.dot(S1.T, grad)    # El grad ya tiene el promedio en bachts

            grad = np.dot(grad, W2.T)
            grad = grad[:, 1:]  # saco la colunmas correspondiente al bias

            # Capa 1
            grad1 = grad_1( Y1 )

            grad = grad * grad1

            dW1 = np.dot(x_bacht.T, grad) # El grad ya tiene el promedio en bachts

            # Actualizo las W
            W1 -= (lr * (dW1 + landa*W1))
            W2 -= (lr * (dW2 + landa*W2))
        

        loss = np.append(loss, log_loss/n_bacht)
        acc  = np.append(acc , log_acc/n_bacht)

        tac = time.time()

        if verbose == True:
            print("{}/{}".format(e,epochs), end=' ')
            print("{}s".format(int(tac-tic)), end=' ')
            print("loss: {:.5f}".format(loss[-1]), end=' ')
            print("acc: {:.2f}".format(acc[-1]), end='\n')

    print("Precision final con los datos de entrenamiento: ", acc[-1])

    return [loss, acc]

def varioNprima(epochs = 10000):

    N = 5
    N_primas = [1, 3, 5, 7, 9, 11]
    cant_ejemplos = 2**N

    x_train = np.array([x for x in itertools.product([-1, 1], repeat=N)])
    y_train = np.prod(x_train, axis=1).reshape(cant_ejemplos, 1)

    figLoss = plt.figure()
    axLoss = figLoss.add_subplot(111)
    figAcc = plt.figure()
    axAcc = figAcc.add_subplot(111)

    for N_prima in N_primas[::-1]:

        loss, acc = fit(x_train, y_train, N_prima, lr=1e-1, landa=0, initialization='normal', epochs=epochs, verbose=False)

        axLoss.plot(loss, label='N\'={}'.format(N_prima))
        axAcc.plot(acc, label='N\'={}'.format(N_prima))

    axLoss.set_xlabel("Epoca", fontsize=15)
    axLoss.set_ylabel("MSE", fontsize=15)
    axLoss.legend(loc='best')
    figLoss.tight_layout()

    axAcc.set_xlabel("Epoca", fontsize=15)
    axAcc.set_ylabel("Accuracy", fontsize=15)
    axAcc.legend(loc='center right')
    figAcc.tight_layout()

    figLoss.savefig(os.path.join(SAVE_PATH, "Loss.pdf"), format='pdf')
    figAcc.savefig(os.path.join(SAVE_PATH, "Acc.pdf"), format='pdf')
    plt.show()


if __name__ == "__main__":
    varioNprima(5000)