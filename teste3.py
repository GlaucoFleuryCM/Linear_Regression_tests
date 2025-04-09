from src.lib import gradients, plotting
import numpy as np

#CURVE_FITTING EM UMA PARABOLA: expansão polinomial de grau 2

#degree of the exponent
degree = 2

#dados no eixo X (feature base para previsao)
data = [1,2,3,4,5,6,7]
data = np.array(data)

#labels: o que desejamos prever
labels = [9,4,1,0,1,4,9]
labels = np.array(labels)

#matriz de Design: comporta todos os exemplos, com
#as features expandidas 
phi = gradients.Design_Matrix(data, degree)

#inicializando theta como aleatório
weights = np.random.rand(degree + 1)
weights = np.array(weights)

#hyperparameters
regularization = 0
lr = 0.0005
#APAGUE ESSE COMENTÁRIO DEPOIS NICOLAS!
#aqui embaixo, tira um 0 de epoch e rode: vc mostra pra eles q tá
#no processo de convergir, mas falta treinar mais; dps add ele e 
#mostra o curve fitting perfeito, e a loss tendendo a 0
epochs = 100000

#rebendo dados do gradient descent: theta atualizado, losses
weights, losses = gradients.Gradient_Descent(labels, \
phi, lr, regularization, epochs, degree + 1)

#plotando os gráficos
plotting.Graph_2D(degree,weights,data,labels,'x','y')  
plotting.Graph_Loss(losses,epochs)