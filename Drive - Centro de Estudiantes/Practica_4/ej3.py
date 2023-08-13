#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 22-04-21
File: ej3.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description: 
"""


import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras import losses, metrics, optimizers

SAVE_PATH = os.path.join(os.getcwd(), "Figuras", "ej3")
os.makedirs(SAVE_PATH, exist_ok=True)

def createAndFitModel(nData=100,testSize=100, valSize=100, lr=1e-3, rf=0, epochs=200, batch_size=1):

    # Datos
    x_train = np.random.uniform(0, 1, nData).reshape((nData, 1))
    y_train = 4 * x_train * (1 - x_train)

    x_val = np.random.uniform(0, 1, nData).reshape((nData, 1))
    y_val = 4 * x_val * (1 - x_val)
    
    x_test = np.random.uniform(0, 1, nData).reshape((nData, 1))
    y_test = 4 * x_test * (1 - x_test)

    # Arquitectura de la red
    inputs  = Input(shape=(x_train.shape[1], ), name="Input")
    layer_1 = Dense(5, activation='sigmoid')(inputs)
    concat  = Concatenate()([inputs, layer_1])
    outputs = Dense(1, activation='linear')(concat)

    # Creamos el modelo
    model = keras.models.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=optimizers.SGD(learning_rate=lr),
        loss=losses.MeanSquaredError(name='loss'),
        # metrics=[metrics.MeanSquaredError(name='acc_MSE')]
    )

    model.summary()

    # Entreno
    hist = model.fit(x_train,
                    y_train,
                    validation_data=(x_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=2)

    # Calculo la loss y Accuracy para los datos de test
    test_loss = model.evaluate(x_test, y_test)

    # Guardo los datos
    # np.save(os.path.join(SAVE_PATH, '{}.npy'.format(description)), hist.history)

    return [hist.history['loss'], hist.history['val_loss'], test_loss]


def varioEjemplos():

    colors = ['b', 'g', 'r', 'm']

    Ns = [5, 10, 100]

    figLoss = plt.figure()
    axLoss = figLoss.add_subplot(111)

    for i in range(len(Ns)):
        
        loss, val_loss, test_loss = createAndFitModel(nData=Ns[i], lr=1e-5, epochs=1000)

        axLoss.plot(loss, c=colors[i], label='N={}'.format(Ns[i]))
        axLoss.plot(val_loss, c=colors[i], linestyle='--', label='N={}-val'.format(Ns[i]))

    axLoss.set_xlabel("Epoca", fontsize=15)
    axLoss.set_ylabel("MSE", fontsize=15)
    axLoss.set_yscale("log", base=10)
    axLoss.legend(loc='best')
    figLoss.tight_layout()

    figLoss.savefig(os.path.join(SAVE_PATH, "Loss.pdf"), format='pdf')
    plt.show()

def mapeoLogistico(x):
    return 4 * x * (1 - x)

def graficosEvolucion(nData=1000000,testSize=500, valSize=500, lr=1e-3, rf=1e-5, epochs=100, batch_size=128):

    # Datos
    x_train = np.linspace(0, 1, nData).reshape((nData, 1))
    # x_train = np.random.uniform(0, 1, nData).reshape((nData, 1))
    y_train = 4 * x_train * (1 - x_train)

    x_val = np.random.uniform(0, 1, valSize).reshape((valSize, 1))
    y_val = 4 * x_val * (1 - x_val)
    
    x_test = np.random.uniform(0, 1, testSize).reshape((testSize, 1))
    y_test = 4 * x_test * (1 - x_test)

    # Arquitectura de la red
    inputs  = Input(shape=(x_train.shape[1], ), name="Input")
    layer_1 = Dense(5, activation='sigmoid')(inputs)
    concat  = Concatenate()([inputs, layer_1])
    outputs = Dense(1, activation='linear')(concat)

    # Creamos el modelo
    model = keras.models.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=optimizers.SGD(learning_rate=lr),
        loss=losses.MeanSquaredError(name='loss')
    )

    model.summary()

    # Entreno
    hist = model.fit(x_train,
                    y_train,
                    validation_data=(x_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=2)
    
    test_loss = model.evaluate(x_test, y_test)

    plt.plot(hist.history['loss'], label="Loss Training")
    plt.plot(hist.history['val_loss'], label="Loss Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, 'Largo_Loss.pdf'), format="pdf")
    plt.show()

    np.random.seed(10)
    x_0 = np.random.rand()

    exact = np.array([x_0])
    wNN = np.array([x_0])

    # Hago la primer iteracion con la ec. exacta y con la red entrenada para comparar
    x_t = mapeoLogistico(exact[-1])
    x_t_nn = model.predict([wNN[-1]])

    exact = np.append(exact, x_t)
    wNN = np.append(wNN, x_t_nn)

    for i in range(100):
        # Y a partir de aca hago la evolucion con la ecuacion exacta
        x_t = mapeoLogistico(exact[-1])
        x_t_nn = mapeoLogistico(wNN[-1])

        exact = np.append(exact, x_t)
        wNN = np.append(wNN, x_t_nn)
    
    plt.plot(exact, label="Exacta")
    plt.plot(wNN, label="Con RN")
    plt.xlabel("Iteracioines")
    plt.ylabel(r"$x(t)$")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.11), ncol=2)
    plt.savefig(os.path.join(SAVE_PATH, 'Largo_Evolucion.pdf'),format="pdf")
    plt.show()




if __name__ == "__main__":
    varioEjemplos()

    # graficosEvolucion()