import numpy as np
from src.lib.feature_map import Polynomial_Regression as PR
from src.lib.feature_map import Combinations_Replacement as CR

#NOTA: é possível facilmente melhorar (e muito) a performance do código 
#usando as bibliotecas do numpy de otimização de matrizes; eu não fiz 
#isso por querer dar uma revisadinha mesmo, e também pra aprender =)


#cria a design matrix, sendo ela nossa nova matriz de treino
def Design_Matrix(input_matrix, phi_degree):
    d_y = len(input_matrix)
    d_x = len(input_matrix[0])

    design_matrix = np.zeros((d_y, CR(d_x, phi_degree) + 1))

    for y in range(d_y):
        design_matrix[y] = PR(input_matrix[y], d_x, phi_degree).copy()

    return design_matrix


#calcula o vetor gradiente dos pesos;
#rf = rigorosidade da regularização
#já recebo a variança ao quadrado;
def Gradient_Weights(labels, phi, variance, weights, rf):
    N = len(phi) #casos teste = n° linhas
    dimensions = len(phi[0]) #dimensões = n° colunas
    gradient = np.zeros((dimensions))

    medium_losses = np.zeros((N))

    #calcula 'r', aka medium loss;
    for i in range(N):
        aux = 0
        for j in range(dimensions):
            aux += weights[j] * phi[i][j]
        medium_losses[i] = labels[i] - aux

    #usa os r's obtidos para calcular cada parcial em relação
    #a cada peso do vetor 'weights';
    for positions in range(dimensions):
        sum = 0
        for i in range(N):
            sum += medium_losses[i] * phi[i][positions]

        sum = sum / (variance * variance * N)
        sum = sum + (weights[positions] * rf)

        gradient[positions] = sum

    return gradient    


#gradiente da variança (já vem squared);
def Gradient_Variance(labels, phi, variance, weights):
    N = len(phi)
    dimensions = len(phi[0])

    const = N / (2 * variance) #constante da variança

    medium_losses = 0 #aqui é só um número, p/calcular fácil

    #calcula soma de r^2;
    for i in range(N):
        aux = 0
        for j in range(dimensions):
            aux += weights[j] * phi[i][j]
        medium_losses += (labels[i] - aux) * (labels[i] - aux)

    medium_losses = (-1) * medium_losses / (2 * variance * variance * N)

    gradient = medium_losses + const

    return gradient #unidimensional (1 só parâmetro)


#como o objetivo não é trabalhar com uma quantidade gigantesca de
#dados, não é necessário implementar a versão estocástica, bastando
#a vanilla;
def Gradient_Descent(labels, phi, variance, weights, n0, n1, rf):
    gradient_w = Gradient_Weights(labels, phi, variance, weights, rf)
    gradient_v = Gradient_Variance(labels, phi, variance, weights)

    variance -= (n0 * gradient_v)

    for i in range(len(weights)):
        weights[i] -= (n1 * gradient_w[i]) 

    variance = clamp(variance) #não pode ser negativo ou 0

    return weights, variance
