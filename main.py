from src.lib import matrix_manipulation as mm
import numpy as np

matrix = [[1, 2, 3, 4], [2, 2, 2, 2], [4, 2, 5, 2]]
label = [1, 2, 3]
weight = [1, 1, 1, 1, 1]

matrix = np.array(matrix)
label = np.array(label)
weight = np.array(weight)

design = mm.Design_Matrix(matrix, 1)
design = np.array(design)
print (f"{design}")

print (mm.Gradient_Weights(label, design, 0.1, weight, 0.1))
# print (mm.Design_Matrix(matrix, 1))


# from src.lib import feature_map

# content = input("digite os valores, separando por espaço: ")
# variaveis = list(map(float, content.split()))#python é um brainrot msm nn é possível
# print (variaveis)
# print ("tamanho do vetor: ", len(variaveis))
# num2 = int (input ("Grau da Expansão: "))
# print ("resultado: ", feature_map.polynomial_regression (variaveis, len(variaveis), num2))

