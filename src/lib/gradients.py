import numpy as np
from src.lib.feature_map import Polynomial_Regression as PR
from src.lib.feature_map import Combinations_Replacement as CR
from src.lib.feature_map import Length
import pdb

#cria a design matrix, sendo ela nossa nova matriz de treino
def Design_Matrix(input_matrix, phi_degree):
    d_y = Length(input_matrix) #range of examples
    d_x = Length(input_matrix[0]) #range of variables

    design_matrix = np.zeros((d_y, CR(d_x, phi_degree) + 1))

    for y in range(d_y): #pra cada exemplo, expande polinomialmente ele
        design_matrix[y] = PR(input_matrix[y], d_x, phi_degree)#tirei um copy

    return design_matrix


#calcula o vetor gradiente dos pesos;
#rf = rigorosidade da regularização
#já recebo a variança ao quadrado;
def Gradient_Weights(labels, phi, variance, weights, rf):
    N = Length(phi) #número de exemplos = n° linhas
    dimensions = Length(phi[0]) #dimensões = n° colunas
    pdb.set_trace()
    gradient = np.zeros((dimensions))

    #calculando gradiente da loss;
    for i in range(N):
        #calculando o dot product primeiro; 
        dot_product = np.dot(weights, phi[i])
        #calculando o desvio do valor esperado;
        error = labels[i] - dot_product
        #calculando o vetor de desvio a ser somado;
        tmp = error * phi[i]
        gradient += tmp

    #multiplicando pelo fator com a variança e somando com o
    #da regularização;

    # factor = -1 / variance
    # gradient = gradient * factor
    gradient = -(2/N) * gradient #MUDEI AQUI DE 'DIMENSIONS' PARA 'N'

    gradient = gradient + (2 * rf * weights)

    return gradient    


#gradiente da variança (já vem squared);
def Gradient_Variance(labels, phi, variance, weights):
    N = Length(phi)
    dimensions = Length(phi[0])

    gradient = 0 #aqui é só um número, p/calcular fácil

    #calcula soma de r^2;
    for i in range(N):
        dot_product = np.dot(weights, phi[i])
        error = labels[i] - dot_product
        #botando o erro ao quadrado 
        error = error * error
        #somando à variável de armazenamento
        gradient += error

    #multiplicando pelo escalar com a variânça
    

    return gradient #unidimensional (1 só parâmetro)


#como o objetivo não é trabalhar com uma quantidade gigantesca de
#dados, não é necessário implementar a versão estocástica, bastando
#a vanilla; agora que parei pra pensar, a fórmula com inversão talvez 
#fosse melhor aqui pra esse contexto :p
def Gradient_Descent(labels, phi, variance, weights, n0, n1, rf):
    gradient_w = Gradient_Weights(labels, phi, variance, weights, rf)
    #talvez aqui seja melhor limitar até onde calcula a variança?
    #ou separar ambos;
    gradient_v = Gradient_Variance(labels, phi, variance, weights)

    variance -= (n0 * gradient_v)

    for i in range(Length(weights)):
        weights[i] -= (n1 * gradient_w[i]) 

    #a variança não pode ficar excessivamente baixa
    if (variance < 1e-6):
        variance = 1e-6

    return weights, variance


