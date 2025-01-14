from math import comb
#propósito: produzir combinações de polinômios X1, X2, .. Xn para poder 
#efetuar o feature mapping e utilizar polynomial regression pra curve fitting;

#utiizar várias vezes para fazer a expansão polinomial;
def combination(polynomials, number, combinations, end, start, size_comb, i, n):
    #casos bases da recursão
    if (i == size_comb):
        combinations[n] = number
        n += 1 #O N TEMM QUE ESTAR*** FORA DA RECURSÃOOOOOOO!!!
        print (n)
        return 

    if (start > end):
        return

    print (f"polinomio = {polynomials[start]}; number = {number}")
    number = number * polynomials[start]

    combination(polynomials, number, combinations, end, start, size_comb, i + 1, n)
    combination(polynomials, number, combinations, end, start + 1, size_comb, i, n)

#para calcular o tamanho do vetor de fmapping;
def combinations_replacement (n, r):
    i = 1
    result = 0

    while i <= r:
        result += comb(n + i - 1, i)
        i += 1

    return result

#assumo que a ordem dos polinômios não importe (não vejo o por que);
def polynomial_regression(polynomials, size_pol, size_comb):
    if (len(polynomials) == 0): 
        return 0
    
    size = combinations_replacement (size_pol, size_comb)

    print (f"tamanho do vetor pra mapping: {size}")

    mapping = [0] * (size + 1) 
    mapping[0] = 1
    n = 1

    index = 1
    while index <= size_comb:
        number = 1
        combination (polynomials, number, mapping, size_pol - 1, 0, index, 0, n)#talvez tenha que ajustar 'size_pol'
        index += 1

    return mapping

    
                                        