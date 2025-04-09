import numpy as np
from src.lib.feature_map import Polynomial_Regression as PR
from src.lib.feature_map import Combinations_Replacement as CR
from src.lib.feature_map import Length, Min, Max

#cria a design matrix, sendo ela nossa nova matriz de treino
def Design_Matrix(input_matrix, phi_degree):
    d_y = Length(input_matrix) #range of examples
    d_x = Length(input_matrix[0]) #range of variables

    design_matrix = np.zeros((d_y, CR(d_x, phi_degree) + 1))

    for y in range(d_y): #pra cada exemplo, expande polinomialmente ele
        design_matrix[y] = PR(input_matrix[y], d_x, phi_degree)

    return design_matrix

#calcula a loss pra poder mostrar quanto mais ou menos que tá baixando
def Loss(labels, phi, weights):
    N = Length(phi) 
    loss = 0

    for i in range(N):
        #calculando o dot product primeiro; 
        dot_product = np.dot(weights, phi[i])
        #calculando o desvio do valor esperado;
        error = labels[i] - dot_product
        #calculando o vetor de desvio a ser somado;
        error = pow(error,2)
        loss += error

    return loss

#calcula o vetor gradiente dos pesos;
#rf = rigorosidade da regularização
def Gradient_Weights(labels, phi, weights, rf):
    N = Length(phi) #número de pontos = n° linhas
    dimensions = Length(phi[0]) #dimensões = n° colunas

    gradient = np.zeros((dimensions))

    #calculando gradiente da loss;
    for i in range(N):
        dot_product = np.dot(weights, phi[i])
        error = dot_product - labels[i]
        tmp = error * phi[i]
        gradient += tmp

    gradient = 2 * gradient / N

    #regularização
    gradient[1:] = gradient[1:] + 2 * rf * weights[1:]

    return gradient    

#como o objetivo não é trabalhar com uma quantidade gigantesca de
#dados, não é necessário implementar a versão estocástica, bastando
#a vanilla; 
def Gradient_Descent(labels, phi, n1, rf, epochs, t_size):
    losses = np.zeros((epochs))
    weights = np.random.rand(t_size)

    for i in range(epochs):
        gradient_w = Gradient_Weights(labels, phi, weights, rf)
        weights -= n1 * gradient_w
        losses[i] = Loss(labels, phi, weights)
        
    return weights, losses

