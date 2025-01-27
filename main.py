from src.lib import control_line, gradients, plotting, feature_map
import numpy as np
import time 

#recebendo especificações da manipulação;
t1 = time.time()
image, dominium, data = control_line.Interface1()
reg, lf1, lf2, degree = control_line.Interface2()

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

#inicializando variáveis
weights = np.random.rand(feature_map.Combinations_Replacement(tamanho, degree) + 1)
variance = np.random.rand(1)

print(weights)

#matriz phi funcionando normalmente
phi = gradients.Design_Matrix(dominium_f, degree)
t2 = time.time()

#errando e aprendendo 1000 vezes à lá Rock Lee;
for i in range(500): 
    weights, variance = gradients.Gradient_Descent(image, phi, variance, weights,
                                                    lf1, lf2, reg)
t3 = time.time()
#plotando tudo
if (tamanho == 2):
    plotting.Graph_3D(degree, weights, image[0:30], dominium1[0:30], dominium2[0:30],
                       image_name, dominium_name[0], dominium_name[1])
else:
    plotting.Graph_2D(degree, weights, dominium_f[0:30],
                       image[0:30], dominium_name, image_name)   
t4 = time.time()

print(f'tempo de treino: {t3 - t2}')
print(f'tempo pra plottar a parada: {t4 - t3}')











