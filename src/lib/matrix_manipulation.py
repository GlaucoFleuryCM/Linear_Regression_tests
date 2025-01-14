import numpy as np
from lib.feature_map import Polynomial_Regression as PR
from lib.feature_map import Combinations_Replacement as CR

#cria a design matrix, sendo ela nossa nova matriz de treino
def Design_Matrix(input_matrix, phi_degree):
    d_x = len(input_matrix[0])
    d_y = input_matrix.size() / d_x

    design_matrix = np.zeros(CR(d_x, phi_degree), d_y)

    for y in range(d_y):
        design_matrix[y] = PR(input_matrix[y], d_x, phi_degree).copy

    return design_matrix

#calcula o vetor gradiente dos pesos;
def Gradient_Weights()

#gradiente da variança;
def Gradient_Variance()

#atualiza pesos e variança no modelo ("learning");
def Update_Model()

#sub-rotina para treinar o modelo várias vezes;
def Training()
