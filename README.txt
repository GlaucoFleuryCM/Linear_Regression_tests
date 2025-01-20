Projeto de Implementação de Regressão linear
========================================================
 Implementação de um modelo de regressão linear que busca
 um vetor paramétrico (via Maximum Likelihood Estimation)
 que se adeque à previsão de dados futuros, para um data set
 providenciado pelo usuário.

Como usar e o que esperar?
---------------------------------
 O programa requer a instalação de algumas livrarias*:
 -matplotlib
 -numpy
 -pandas
 *as demais livrarias (os + glob) já são por padrão instaladas
 junto do python em sua máquina
 
 Após clonar o repositório em seu computador local, acesse 
 a pasta contendo-o e digite em seu terminal:

 $ python3 main.py
 
 O programa irá perguntar algumas coisas, as quais são bem
 diretas para entender. Será buscada em sua pasta de Downloads
 um arquivo no formato .csv, e você deve escolher quais dimensões
 deseja usar como domínio (MÁX = 2) e quantas como imagem (MÁX =
 1). Após isso, você deve escolher alguns fatores importantes 
 para o programa (learning factors e constante de regularização).
 A partir disso, um gráfico (2D ou 3D, dependendo do número de
 domínios escolhidos) aparecerá, exibindo a curva preditiva do
 modelo e alguns dados do data set para comparação.

 Velocidade: o programa é lento, porém não otimizá-lo foi uma 
 escolha deliberada, já que não é muito necessário (criado por
 diversão) e também considerando o fato de que eu queria testar 
 algumas propriedades matriciais; a otimização seria muito 
 simples, bastando trocar as multiplicações de matrizes nos 
 gradientes (loops) por funções do numpy.

Teoria e Matemática
-----------------------
 Discutidos no pdf.

