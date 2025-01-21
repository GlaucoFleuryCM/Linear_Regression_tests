Projeto de Implementação de Regressão linear
========================================================
 Implementação de um modelo de regressão linear que busca
 um vetor paramétrico (via Maximum Likelihood Estimation)
 que se adeque à previsão de dados futuros, para um data set
 providenciado pelo usuário.


Como usar e o que esperar
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
 para o programa. A partir disso, um gráfico (2D ou 3D, 
 dependendo do número de domínios escolhidos) aparecerá,
 exibindo a curva preditiva do modelo e alguns dados do 
 data set para comparação.


Parâmetros requisitados
--------------------------
 -'Username' = o nome de usuário em sua máquina
 -'csv escolhido' + dimensões = digite os números desejados
 -'fator de regularização' = o quanto você quer penalizar O
 overfitting do modelo / estimular a generalização
 -'l.f da variança' + 'l.f dos pesos' = o quanto você deseja
 que o modelo mude com base no gradiente 
 -'grau da expansão polinomial' = qual o grau do polinômio
 que você deseja calcular para se adequar ao modelo?

 Recomenda-se os seguintes valores:
 -l.f variança => [1e-6, 1e-4]
 -l.f pesos => [1e-4, 1e-3]
 -fator da regularização => [1e-1, 1e+1]
 no fundo, será necessário testá-los múltiplas vezes
 para achar os parâmetros que melhor se adequam ao
 contexto.

Teoria e Matemática
-----------------------
 Discutidos no pdf presente em 'relatorio_teste'

