import pandas as pd
import glob 
import os

#objetivo: trazer as especificações da base de dados pro python;
def Interface1():
    #buscando e selecionando os .csv necessários;
    print ("os dados serão buscados na sua pasta de Downloads")
    user = input("Username: ")
    try:
        os.chdir(f'/home/{user}/Downloads') 
    except:
        print('username incorreto')
        exit()  
    print()
    print("Escolha um pelo número:")
    files = glob.glob('*.csv')
    i = 0
    for file in files:
        i += 1
        print(f'{i} = {file}')
    print()
    chosen = input("csv escolhido: ")
    if (int(chosen) > i):
        print('nah')
        exit()
    data = pd.read_csv(rf'/home/{user}/Downloads/{files[int(chosen)-1]}')
    #usuário pode escolher domínio e imagem do gráfico que for formado;
    print()
    num = int(input('Quantas dimensões você quer usar de Domínio? '))
    if(num>2 or num<1):
        print("invalido")
        exit()
    print()    
    print('Escolha abaixo suas dimensões, dentre as possíveis: ')
    options = list(data.columns)
    i = 0
    for names in options:
        i+=1
        print (f'{i}:{names}')
    print()
    dominium = []
    for j in range(num):
        aux = int(input(f'dimensão{j+1}: '))
        dominium.append(aux)
    print()
    image = int(input('Qual delas você quer usar de imagem? '))

    return(image, dominium, data)

#recebe detalhes da implementação;
def Interface2():
    print("Me forneça:")
    reg = int(input("fator de regularização: "))
    lf1 = int(input("learning factor da variança: "))
    lf2 = int(input("learning factor dos pesos: "))
    degree = int(input("grau da expansão polinomial: "))

    return(reg, lf1, lf2, degree)

def Two_Dimensions(image, dominium, data):
    indexes = data.columns
    image_name = indexes[image-1]
    dominium_name = []
    image = list(data[image_name])
    num_dados = len(data)

    dominium_name.append(indexes[int(dominium[0]) - 1])
    dominium1 = list(data[dominium_name[0]])
    dominium_name.append(indexes[int(dominium[1]) - 1])
    dominium2 = list(data[dominium_name[1]])
    
def One_Dimension(image, dominium, data):    
    indexes = data.columns
    image_name = indexes[image-1]
    dominium_name = []
    image = list(data[image_name])
    num_dados = len(data)

    dominium_name = indexes[dominium[0] - 1]
    dominium1 = list(data[dominium_name])