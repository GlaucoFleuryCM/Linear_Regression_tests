import pandas as pd
import glob 
import os
#OBJETIVO: providenciar dataset, especificações de modelagem, e 
#terminar com uma plotagem (caso seja 2D/3D) e também com uma
#capacidade de predição maneira

print ("os dados serão buscados na sua pasta de Donwloads")
user = input("Username: ")
try:
    os.chdir(f'/home/{user}/Downloads') 
except:
    print('username incorreto')
    exit()  
print()
#tem que digitar o númerozinho
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
#quer grafar oq cm oq? entre 1 e 2 opções
print()
num = input('Quantas dimensões você quer usar de Domínio?')
print('Escolha abaixo suas dimensões, dentre as possíveis:')
options = list(data.columns)
i = 0
for names in options:
    i+=1
    print (f'{i}:{names}')
print()






