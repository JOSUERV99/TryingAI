"""
Perceptron Simple

Neurona biologica:
Partes:
    #Dendritas: se encargan de recibir informacion de
    otra neurona(entrada)
    #Nucleo: donde surgen las transformaciones de la
    entrada (procesamiento, activacion)
    #Axon: transmite la informacion procesada por el nucleo 
    hacia otra neurona (salida)
""" 
""""
Neurona Artificial:
    Partes:
    Conexiones de entrada (dendritas)
        x1, x2, x3, ... xn
    Unidad, Nodo: 
        Mecanismo de la red neuronal
            v_x = [x1, x2, x3] # vector de entradas
            v_w = [w1, w2, w3] # vector de pesos
            z = [v_x] * [v_w] # producto punto de vectores

        Funcion de activacion:
            salida = f(z) = f( [v_x]*[v_w] ) > u
            *** algunas funciones de activacion
                sigmoide e [0,1], relu, tanh 
                
    Salida
        y -> resultado de la salida de la funcion de activacion
        dado el valor de la suma ponderada de las entradas dadas

    Aprendizaje
        1. xj entrada en posicion j
        2. wj peso en posicion j
        3. y salida de la neurona
        4. Y salida esperada
        5. a constante tal que 0 < a < 1 (learning rate)
"""


"""
        Regla de actualizacion de pesos
            wj' = wj + a*(Y-y)*xj  #de forma analoga con a = 1
            Despues de cada iteracion al obtener y si esta difiere de Y se repite el proceso

            1. xi entrada en posicion i
            2. wi peso en posicion i
            3. yi salida para cada iteracion
            4. Dm = {(x1,y1),...,(xm,ym)} periodo de aprendizaje de m iteraciones
            Para actualizar los pesos
                Por cada (x,y) en Dm se pasa a la regla de actualizacion

        Es linealmente separable si:
               si existe un valor positivo v  
               y un vector de peso w tal que:
               yi * ( (w,xi) + u ) > v para todos los i

               El numero de errores esta limitado a:
               (2R/v)^2        
"""

import random as r
import numpy as np
from math import e

def weightedSum(inputs):
    return sum ([inputs[x]*weights[x] for x in range(len(weights))])

def sigmoide(x):
    return 1/(1 + e**(-x))

def tanh(x):
    return 2/(1 + e**(-2*x)) - 1

train_set = [ [[0, 0], 0],  [[0, 1], 1], [[1, 0], 1], [[1, 1], 1]] # or function

inputs = []
n_inputs = 2
weights = []
bias = 1
umbral = 0.5
alpha = 0.1

inputs = [ float(input(f'x_{x} :')) for x in range(n_inputs) ] + [bias]
weights = [r.random() for _ in range(len(inputs)) ]

def calculateOutput(weightedSum, fn=sigmoide):
    if fn(weightedSum) > umbral:
        return 1
    else:
        return 0

z = weightedSum(inputs)
outputs = [ weights[x]*inp for x, inp in enumerate(inputs)]

def training(inputs, weights, outputs, train_set, alpha=1, iterations=10000):
    for _ in range(iterations):
        for x, train_row in enumerate(train_set):
            for j, wj in enumerate(weights):
                #[[0, 0], 0] Primer valor es la lista de entradas, y el otro valor es la salida esperada
                expectedResult = train_row[1]
                # actualizacion de los pesos  wj' = wj + a*(Y-y)*xj
                weights[j] = wj + alpha*(expectedResult - outputs[j])*inputs[j]

def score(outputs, train_set):
    score = 0.0
    for i in range(len(outputs)):
        score += train_set[i][1] - outputs[i]
    print("Score: ", score/len(outputs))

def predict(test_input):
    predictions = [ sum([weights[x]*inp for inp in test_input[x]]) for x in range(len(weights))]
    print(predictions)

training(inputs, weights, outputs, train_set)
score(outputs, train_set)

print(outputs)
print('\n', train_set)
print(weights)

predict([[1, 1, bias], [1, 0, bias], [0, 1, bias], [0, 0, bias]])