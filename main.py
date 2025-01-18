from src.lib import control_line, gradients, plotting
import pandas as pd
import numpy as np

#recebendo especificações da manipulação;
image, dominium, data = control_line.Interface1()
#reg, lf1, lf2, degree = control_line.Interface2()

#ajustando dados para serem utilizados pelo computador
#(transformando tudo em número e lista)
indexes = data.columns
image_name = indexes[image-1]
dominium_name = []
image = list(data[image_name])
num_dados = len(data)
if (len(dominium) == 2):
    dominium_name.append(indexes[int(dominium[0]) - 1])
    dominium1 = list(data[dominium_name[0]])
    dominium_name.append(indexes[int(dominium[1]) - 1])
    dominium2 = list(data[dominium_name[1]])
    dominium_f = np.column_stack((dominium1, dominium2))
else:
    dominium_name = indexes[dominium[0] - 1]
    dominium_f = np.array(data[dominium_name])

#inicializando variáveis
weights = np.random.rand(num_dados)
variance = np.random.rand(1)

phi = gradients.Design_Matrix(dominium_f, 2)

print (phi)









