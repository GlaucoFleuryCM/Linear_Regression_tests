from src.lib import control_line, gradients, plotting, feature_map
import numpy as np

#recebendo especificações da manipulação;
image, dominium, data = control_line.Interface1()
reg, lf, degree, trials = control_line.Interface2()

#ajustando dados para serem utilizados pelo computador
#(transformando tudo em número e lista)
indexes = data.columns
image_name = indexes[image-1]
dominium_name = []
image = list(data[image_name])
num_dados = len(data)
tamanho = len(dominium)
if (tamanho == 2):
    dominium_name.append(indexes[int(dominium[0]) - 1])
    dominium1 = list(data[dominium_name[0]])
    dominium_name.append(indexes[int(dominium[1]) - 1])
    dominium2 = list(data[dominium_name[1]])
    dominium_f = np.column_stack((dominium1, dominium2))
else:
    dominium_name = indexes[dominium[0] - 1]
    dominium_f = np.array(data[dominium_name])

#inicializando a design matrix;
phi = gradients.Design_Matrix(dominium_f, degree)
#calculando tamanho do vetor de pesos;
w_size = feature_map.Combinations_Replacement(tamanho, degree) + 1

#treinando o modelo;
weights, losses, trials = gradients.Gradient_Descent(image, phi, lf, reg, trials, w_size)

#plotando tudo  
if (tamanho == 2):
    plotting.Graph_3D(degree, weights, image, dominium1, dominium2,
                       image_name, dominium_name[0], dominium_name[1])
    plotting.Graph_Loss(losses, trials)
else:
    plotting.Graph_2D(degree, weights, dominium_f,
                       image, dominium_name, image_name)
    plotting.Graph_Loss(losses, trials)   












