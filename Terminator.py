##################Imports section###################
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')

from sklearn.model_selection import train_test_split

################ Load Data section ##################
def load_MNIST_Data():
    train_img, test_img, train_lbl, test_lbl = train_test_split(mnist.data, mnist.target, test_size=1/7.0, random_state=0)
    return train_img, train_lbl, test_img, test_lbl

################# Neural sectionz####################


class Neural_Network(object):
    def __init__(self):
        self.inputSize = 784
        self.outputSize = 10
        self.hiddenSize = 128

        #inicializo el W
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (100, 128)
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (100, 128)
        
    def forward(self, X):
        self.z = np.dot(X, self.W1)
        self.z2 = self.relu(self.z)             # activation function
        self.z3 = np.dot(self.z2,self.W2) 
        output = self.relu(self.z3)             # final activation function
        print(output.shape)
        return self.z 

    def relu(self,x):
        return np.maximum(x, 0, x)

    def cross_entropy(predictions, targets, epsilon=1e-12):
        """
        Computes cross entropy between targets (encoded as one-hot vectors)
        and predictions. 
        Input: predictions (N, k) ndarray
               targets (N, k) ndarray        
        Returns: scalar
        """
        predictions = np.clip(predictions, epsilon, 1. - epsilon)
        N = predictions.shape[0]
        ce = -np.sum(np.sum(targets*np.log(predictions+1e-9)))/N
        return ce

    

"""
predictions = np.array([[0.25,0.25,0.25,0.25],
                        [0.01,0.01,0.01,0.96]])
targets = np.array([[0,0,0,1],
                   [0,0,0,1]])
ans = 0.71355817782  #Correct answer
x = cross_entropy(predictions, targets)
print(np.isclose(x,ans))
"""


def Train():
    data = load_MNIST_Data()
    train_X = data[0]       #imagenes de entrenamiento (60000)
    train_Y = data[1]       #Labeld de entrenamiento (60000)

    test_X = data[2]        #Imagenes de prueba (10000)
    test_Y = data[3]        #Labels de prueba (10000)

    X = train_X[:100]       #Por el momento se toman las 100 primeras imagenes, debe ser aleatorio
    Y = train_Y[:100]

    Y_vectorizado = np.zeros((len(X), len(X[0])))     #Creacion de labels vectorizados
    for i in range(len(Y)):                     
        Y_vectorizado[i][int(Y[i])] = 1            #Se pone 1.0 en la posicion del vector

    NN = Neural_Network()
    output = NN.forward(X)

    #No funciona aun
    #ce = NN.cross_entropy(output, Y_vectorizado)
    #print(ce)

    
Train()
