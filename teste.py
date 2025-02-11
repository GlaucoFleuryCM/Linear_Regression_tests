from src.lib import gradients, plotting
import numpy as np

#PRA MUDAR GRAU DE PHI: mudar 'phi', 'w_size', e 'graph-2D'
w_size = 3

data = [1,2,3,4,5,6,7]
data = np.array(data)

labels = [9,4,1,0,1,4,9]
labels = np.array(labels)

phi = gradients.Design_Matrix(data, 2)

weights = np.random.rand(w_size)
weights = np.array(weights)

regularization = 0
lr = 0.00005
trials = 1000000

weights, losses, trials = gradients.Gradient_Descent(labels, phi, lr, regularization, trials, w_size)

print(f'peso dos polinômios, de X^0 a X^N: {weights}')

plotting.Graph_2D(2,weights,data,labels,'x','y')  
plotting.Graph_Loss(losses,trials)