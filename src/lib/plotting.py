import matplotlib.pyplot as mlp #my little ponny kkk;
from mpl_toolkits import mplot3d #precisa, pro eixo Z
from src.lib.feature_map import Polynomial_Regression, Combinations_Replacement
import numpy as np

#pra gráficos 3D, calcula o valor de f(x,y), considerando se tratar de
#uma expansão polinomial;
def Function_3D(exp, weights, phi_degree, X, Y):
    num = Combinations_Replacement(2, phi_degree) 
    Z = 0
    for i in range(num):
        exp_Y = exp[num] / 2
        exp_X = phi_degree - exp_Y

        Z += weights[i] * pow(X, exp_X) * pow(Y, exp_Y)
    
    return Z    


#plotta o gráfico 3D do modelo;
def Graph_3D(phi_degree, weights, z_points, x_points, y_points, z_name, x_name, y_name): 
    x = [1,2] #gambiarra
    exp = Polynomial_Regression(x, 2, phi_degree)
    
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)

    Z = Function_3D(exp, weights, phi_degree, X, Y)

    fig = mlp.figure()
    ax = fig.add_subplot(111, projection = '3d')

    #dá pra pôr label aqui?
    ax.plot_surface(X, Y, Z, cmap='viridis')

    #mudar? maybe
    ax.scatter(x_points, y_points, z_points, color='red', label='data', marker='o')

    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_zlabel(z_name)

    mlp.legend()
    mlp.show()


#cálculo de f(X) = Y
def Function_2D(weights, phi_degree, X):
    Z = 0
    phi_degree = int(phi_degree)
    for i in range(phi_degree):
        Z += weights[i] * X
    
    return Z


#gráfico 2D;
def Graph_2D(phi_degree, weights, x_points, y_points, x_name, y_name):

    X = np.linspace(-100,100, 1000)

    Y = Function_2D(weights, phi_degree, X)

    #pra que mexer em fig size?
    fig = mlp.figure(figsize = (10,8))
    ax = fig.add_subplot(111)

    ax.plot(X, Y, label='Fitted function', color='blue')

    ax.scatter(x_points, y_points, color='red', label='data', marker='o')

    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)

    ax.legend()
    mlp.show()
    

