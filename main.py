from src.lib import control_line, gradients, plotting
import pandas as pd

#recebendo especificações da manipulação;
image, dominium, data = control_line.Interface1()

#ajustando dados para serem utilizados pelo computador
indexes = data.columns
image_name = indexes[image-1]
dominium_name = []
if (len(dominium) == 2):
    dominium_name.append(indexes[int(dominium[0]) - 1])
    dominium1 = list(data[dominium_name[0]])
    dominium_name.append(indexes[int(dominium[1]) - 1])
    dominium2 = list(data[dominium_name[1]])
else:
    dominium_name = indexes[dominium[0] - 1]
    dominium1 = list(data[dominium_name])
image = list(data[image_name])
num_dados = 







