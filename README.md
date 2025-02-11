# Projeto de Implementação de Regressão linear

 Implementação de um modelo de regressão linear que busca
 um vetor paramétrico (via Maximum Likelihood Estimation)
 que se adeque à previsão de dados futuros, para um data set
 providenciado pelo usuário.

## Como usar e o que esperar

 O programa requer a instalação de algumas livrarias*:
 - matplotlib
 - numpy
 - pandas

 *as demais livrarias (os + glob) já são por padrão instaladas
 junto do python em sua máquina
 
 Tenha em mente as seguintes limitações:
 - só serão aceitas plottagens 2D ou 3D
 - só arquivos .csv são aceitos
 - datasets com valores muito altos sofrerão overflow

 Após clonar o repositório em seu computador local, acesse 
 a pasta contendo-o e digite em seu terminal:

    $ python3 main.py

 daí você deve informar o que for requisitado.
 A partir disso, um gráfico (2D ou 3D, 
 dependendo da dimensionalidade escolhida do domínio) aparecerá,
 exibindo a curva preditiva do modelo.

 Se quiser um exemplo prático da capacidade do modelo, escreva:

    $ python3 teste.py

## Parâmetros requisitados

 - 'Username' = o nome de usuário em sua máquina
 - 'csv escolhido' + dimensões = digite os números desejados
 - 'fator de regularização' = o quanto você quer penalizar O
 overfitting do modelo / estimular a generalização
 - 'l.f dos pesos' = o quanto você deseja
 que o modelo mude com base no gradiente 
 - 'grau da expansão polinomial' = qual o grau do polinômio
 que você deseja calcular para se adequar ao modelo?

 Recomenda-se os seguintes valores:
 - l.f pesos => [1e-12, 1e-1]
 - fator da regularização => [1e-5, 1e-1]
 no fundo, será necessário testá-los múltiplas vezes
 para achar os parâmetros que melhor se adequam ao
 contexto. Tenha em mente: quanto maior a dimensão de
 de seus dados, menor terá que ser o learning factor.

## Teoria e Matemática da Implementação

 Discutidos no pdf presente em 'relatorio_teste'

