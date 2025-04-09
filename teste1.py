from src.lib import gradients, plotting
import numpy as np

#REGRESSÃO LINEAR BASIQUINHA 2D

#degree of the exponent
degree = 1

#dados no eixo X (feature base para previsao)
data = [1,2,3,4,5,6,7,8,9,10,11,12]
data = np.array(data)

#labels: o que desejamos prever
labels = [1,3,2.4,6,5.5,7,7.6,8.9,8.1,12,10.9,11.4]
labels = np.array(labels)

#matriz de Design: comporta todos os exemplos, com
#as features expandidas 
phi = gradients.Design_Matrix(data, degree)

#inicializando theta como aleatório
weights = np.random.rand(degree + 1)
weights = np.array(weights)

#hyperparameters
regularization = 0
lr = 0.005
epochs = 1000

#rebendo dados do gradient descent: theta atualizado, losses
weights, losses = gradients.Gradient_Descent(labels, \
phi, lr, regularization, epochs, degree + 1)

#plotando os gráficos
plotting.Graph_2D(degree,weights,data,labels,'x','y')  
plotting.Graph_Loss(losses,epochs)