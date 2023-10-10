"""
File: xor-model-1.py
Author: German Mato
Email: matog@cab.cnea.gov.ar
Description: Codificacion del XOR modelo 1 - version tensorflow 2 - modified accuracy
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

seed=3                           # for reproducibility 
np.random.seed(seed)
tf.random.set_seed(seed)

# Network architecture
hidden_dim=2                            # Numero de unidades ocultas
model=tf.keras.Sequential(
    [
         tf.keras.layers.Input(shape=(2,)),
         tf.keras.layers.Dense(hidden_dim, activation="tanh"),
         tf.keras.layers.Dense(1, activation="tanh"),
    ]
)


# Data Input

ntrain=4
x_train=np.zeros((ntrain,2), dtype=np.float32)
y_train=np.zeros((ntrain,1), dtype=np.float32)


x_train[0,0]=1
x_train[0,1]=1
y_train[0]=1

x_train[1,0]=-1
x_train[1,1]=1
y_train[1]=0

x_train[2,0]=1
x_train[2,1]=-1
y_train[2]=0

x_train[3,0]=-1
x_train[3,1]=-1
y_train[3]=1

print(x_train.shape)
print(y_train.shape)

x_test=np.copy(x_train)
y_test=np.copy(y_train)
print(x_test.shape)
print(y_test.shape)

# accuracy compatible with tensorflow v1
from tensorflow.python.keras import backend as K
def v1_accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

# Model 
opti=tf.keras.optimizers.SGD(lr=0.01)

model.compile(optimizer=opti,
              loss='MSE', metrics=[v1_accuracy])

history=model.fit(x=x_train, y=y_train,
                epochs=500,
                batch_size=4,
                shuffle=False,
                validation_data=(x_test, y_test), verbose=True)
#
tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=False, show_layer_names=True, rankdir='TB')
print(model.summary())
encoded_log = model.predict(x_test, verbose=True)
print(encoded_log.shape)


#####################################################################
# Output files
fout=open("xor-out.dat","wb")
ftrain=open("xor-train.dat","wb")
ftest=open("xor-test.dat","wb")
#
np.savetxt(ftrain,np.c_[x_train,y_train],delimiter=" ")
np.savetxt(ftest,np.c_[x_test, y_test],delimiter=" ")
np.savetxt(fout,np.c_[x_test, encoded_log],delimiter=" ")

W_Input_Hidden = model.layers[0].get_weights()[0]
W_Output_Hidden = model.layers[1].get_weights()[0]
B_Input_Hidden = model.layers[0].get_weights()[1]
B_Output_Hidden = model.layers[1].get_weights()[1]
#print(summary)
print('INPUT-HIDDEN LAYER WEIGHTS:')
print(W_Input_Hidden)
print('HIDDEN-OUTPUT LAYER WEIGHTS:')
print(W_Output_Hidden)

print('INPUT-HIDDEN LAYER BIAS:')
print(B_Input_Hidden)
print('HIDDEN-OUTPUT LAYER BIAS:')
print(B_Output_Hidden)

# "Loss"
plt.plot(np.sqrt(history.history['loss']))
plt.plot(np.sqrt(history.history['val_loss']))
plt.plot(history.history['v1_accuracy'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation','v1_accuracy'], loc='upper left')
plt.show()
