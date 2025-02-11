from src.lib import control_line, gradients, plotting, feature_map
import numpy as np
import matplotlib.pyplot as plt
import pdb

#PRA MUDAR GRAU DE PHI: mudar 'phi', 'weights', e 'graph-2D'
w_size = 7

#data = [0,1,2,3,4]
data = [1,3,5,7,9,11]
data = np.array(data)

#labels = [0,1,2,3,4]
labels = [1,4,2,8,7,14]
labels = np.array(labels)

phi = gradients.Design_Matrix(data, 6)

weights = np.random.rand(w_size)
weights = np.array(weights)

regularization = 0
lr = 0.01
trials = 100000

#weights = gradients.Analitical_MLE(labels, phi)
weights, losses, trials = gradients.Gradient_Descent(labels, phi, lr, regularization, trials, w_size)

print(weights)

plotting.Graph_2D(6,weights,data,labels,'x','y')  
plotting.Graph_Loss(losses,trials)