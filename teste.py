from src.lib import control_line, gradients, plotting, feature_map
import numpy as np

data = [1, 2, 4, 8, 3]
data = np.array(data)

labels = [0, 3, 4, 7, 5]
labels = np.array(labels)

phi = gradients.Design_Matrix(data, 2)

weights = [0.5, 0.5, 0.5]
weights = np.array(weights)
variance = 1e-2

for i in range(100):
    weights, variance = gradients.Gradient_Descent(
                                    labels, phi, variance,
                                    weights, 1e-4, 1e-4, 0 
                                )
    
plotting.Graph_2D(4,weights,data,labels,'x','y')    