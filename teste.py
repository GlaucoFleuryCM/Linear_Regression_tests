from src.lib import gradients as g
from src.lib import feature_map as fm


data = [1,2,3]
labels=[1,4,6]
weights=[1,1]
lamb = 1
variance = 1

data = g.Design_Matrix(data, 1)

print(f'tabela de dado: {data}')

print (f'resultado: {g.Gradient_Weights(labels, data, variance, weights, lamb)}')

    # #calcula 'r', aka medium loss;
    # for i in range(N):
    #     aux = 0
    #     for j in range(dimensions):
    #         aux += weights[j] * phi[i][j]
    #     medium_losses[i] = labels[i] - aux

    # #usa os r's obtidos para calcular cada parcial em relação
    # #a cada peso do vetor 'weights';
    # for positions in range(dimensions):
    #     sum = 0
    #     for i in range(N):
    #         sum += medium_losses[i] * phi[i][positions]

    #     sum = sum / (variance * variance * N)
    #     sum = sum + (weights[positions] * rf)

    #     gradient[positions] = sum