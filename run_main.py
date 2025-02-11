import kagglehub
from os import walk, chdir, getlogin
from sys import platform
import subprocess
import pdb

#script pra te mostrar um exemplo do que a regressão criada é capaz de fazer;
#pegarei um dataset que eu catei da net, e settarei tudo certinho, de tal modo
#que você possa visualizar de forma bem lindinha o data-fit; o csv estará
#em /Downloads (remova depois manualmente), e o projeto deve estar na sua home; 

#checar se é um sistema LINUX
if platform != "linux" and platform != "linux2":
    print('Você não está utilizando um sistema linux')
    quit()
    
#checando se, na pasta de Downloads o arquivo já está baixado;
#se não estiver, baixa; do contrário, deixa quieto mesmo;

#pegando o username
user = getlogin()

#settando o diretório em que vou procurar o csv;
chdir(f'/home/{user}/Downloads') 

#procurando o file
flag = False
for paths, directory, files in walk('.'):
    for file in files:
        if (file == 'crop_yield_data.csv'):
            flag = True

#baixando o treco no seu computador;
if (flag == False):            
    kagglehub.dataset_download("govindaramsriram/crop-yield-of-a-farm")  

#agora que o data-set tá aí, vamos rodar a main e botar pra quebrar;

#buscando o diretório em que o código está pra settar as parada tudo
#ASSUMO que tu deu git clone do projeto na sua home; 
#Não é necessário o 'if __name__ == "__main__"'; eu quero executar TUDO
#do programa que eu estou importanto mesmo;
chdir(f'/home/{user}/BayesProject')
inputs = [str(user), '2', '1', '1',
           '2', '0', '0.005', '10', '100000']
#'fingindo' que é um comando que eu dei no terminal;
inputs = "\n".join(inputs) + "\n"
subprocess.run(
    ["python3", "main.py"], 
    input = inputs.encode('utf-8')
    )
