from src.lib import feature_map

content = input("digite os valores, separando por espaço: ")
variaveis = list(map(float, content.split()))#python é um brainrot msm nn é possível
print (variaveis)
print ("tamanho do vetor: ", len(variaveis))
num2 = int (input ("Grau da Expansão: "))
print ("resultado: ", feature_map.polynomial_regression (variaveis, len(variaveis), num2))

