from src.lib import gradients, plotting
import numpy as np

#OVER_FITTING E UNDER_FITTING

#degree of the exponent
degree = 7

#dados no eixo X (feature base para previsao)
data = [0.1, 0.15, 0.3, 0.45, 0.7, 0.85]
data = np.array(data)

#labels: o que desejamos prever
labels = [0.1, 0.2, 0.24, 0.19, 0.7, 0.65]
labels = np.array(labels)

#matriz de Design: comporta todos os exemplos, com
#as features expandidas 
phi = gradients.Design_Matrix(data, degree)

#inicializando theta como aleatório
weights = np.random.rand(degree + 1)
weights = np.array(weights)

#hyperparameters
regularization = 0 #REGULARIZAR: mudar esse valor para 0.12
lr = 0.12
epochs = 1000000

#rebendo dados do gradient descent: theta atualizado, losses
weights, losses = gradients.Gradient_Descent(labels, \
phi, lr, regularization, epochs, degree + 1)

#plotando os gráficos
plotting.Graph_2D(degree,weights,data,labels,'x','y')  
plotting.Graph_Loss(losses,epochs)