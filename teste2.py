from src.lib import gradients, plotting
import numpy as np

#REGRESSÃO LINEAR BASIQUINHA 3D

#degree of the exponent
degree = 1

#feature 1
data1 = [1,2,3,4,5,6,7,8,9,10,11,12]
data1 = np.array(data1)

#feature 2
data2 = [1,2,3,4,5,6,7,8,9,10,11,12]
data2 = np.array(data2)

#juntando as 2 features em uma matriz
data = np.column_stack((data1, data2))

#labels: o que desejamos prever
labels = [1,3,2.4,6,5.5,7,7.6,8.9,8.1,12,10.9,11.4]
labels = np.array(labels)

#matriz de Design: comporta todos os exemplos, com
#as features expandidas 
phi = gradients.Design_Matrix(data, degree)

#inicializando theta como aleatório
weights = np.random.rand(3)#o tamanho muda pra acomodar 1,x1,x2
weights = np.array(weights)

#hyperparameters
regularization = 0
lr = 0.005
epochs = 10000

#rebendo dados do gradient descent: theta atualizado, losses
weights, losses = gradients.Gradient_Descent(labels, \
phi, lr, regularization, epochs, 3)

#plotando os gráficos
plotting.Graph_3D(degree,weights,labels,data1,data2,'z','x','y')  
plotting.Graph_Loss(losses,epochs)