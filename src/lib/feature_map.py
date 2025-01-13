#propósito: produzir combinações de polinômios X1, X2, .. Xn para poder 
#efetuar o feature mapping e utilizar polynomial regression pra curve fitting;

#utiizar várias vezes para fazer a expansão polinomial;
def combination(polynomials, vec, combinations, size_pol, size_comb):
    #vec[0] guarda o valor, vec[1] guarda o n° de elementos para formá-lo;
    #bases recursivas;
    if (vec[1] == size_comb):
        combinations.append(vec[0])
        return      
    if (size_pol + vec[1] < size_comb):#CHECAR
        return
    
    size_pol -= 1
    combination(polynomials, vec, combinations, size_pol, size_comb)
    vec[1] += 1
    vec[0] = vec[0] * polynomials[size_pol + 1]
    combination(polynomials, vec, combinations, size_pol, size_comb)

#assumo que a ordem dos polinômios não importe (não vejo o por quê);
def polynomial_regression(polynomials, size_pol, size_comb):
    if (len(polynomials) == 0): 
        return 0
    
    mapping = []
    mapping.append(1)

    for index in range(0,size_comb-1):
        list = []
        for variable in polynomials:
            list = [variable] * index
        vec = []
        combinations = []
        combination (list, vec, combinations, size_pol, size_comb)
        mapping.append(combinations)

    return combinations




    
                                        