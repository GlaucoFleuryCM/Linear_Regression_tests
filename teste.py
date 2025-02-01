from src.lib import control_line, gradients, plotting, feature_map
import numpy as np
import pdb

#PRA MUDAR GRAU DE PHI: mudar 'phi', 'weights', e 'graph-2D'

data = [1, 2, 3]
data = np.array(data)

labels = [1, 4, 9]
labels = np.array(labels)

phi = gradients.Design_Matrix(data, 2)

weights = [0.1, 0.1, 0.1]
weights = np.array(weights)
variance = 1e-3

losses = np.zeros((1000))
for i in range(1000):
    weights, variance, loss = gradients.Gradient_Descent(
                                    labels, phi, variance,
                                    weights, 1e-2, 1e-2, 0
                                )
    losses[i] = loss
    
plotting.Graph_2D(2,weights,data,labels,'x','y') #passei phi degree como 1   
plotting.Graph_Loss(losses,1000)