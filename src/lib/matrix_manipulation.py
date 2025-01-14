import numpy as np
from src.lib.feature_map import Polynomial_Regression as PR
from src.lib.feature_map import Combinations_Replacement as CR


#cria a design matrix, sendo ela nossa nova matriz de treino
def Design_Matrix(input_matrix, phi_degree):
    d_y = len(input_matrix)
    d_x = len(input_matrix[0])

    design_matrix = np.zeros((d_y, CR(d_x, phi_degree) + 1))

    print (f"{design_matrix}")

    for y in range(d_y):
        design_matrix[y] = PR(input_matrix[y], d_x, phi_degree).copy()

    return design_matrix


#calcula o vetor gradiente dos pesos;
#lf = learning factor

#NOTA: é possível facilmente melhorar a performance do código usando as
#bibliotecas do numpy de otimização de matrizes; eu não fiz isso por 
#querer dar uma revisadinha mesmo, e também pra aprender =)
def Gradient_Weights(labels, phi, variance, weights, lf):
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
        sum = sum + (lf * weights[positions])

        gradient[positions] = sum

    return gradient    

# #gradiente da variança;
# def Gradient_Variance()

# #atualiza pesos e variança no modelo ("learning");
# def Update_Model()

# #sub-rotina para treinar o modelo várias vezes;
# def Training()
