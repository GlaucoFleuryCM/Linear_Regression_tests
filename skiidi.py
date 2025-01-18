from src.lib.gradients import Design_Matrix
from src.lib.feature_map import Polynomial_Regression

vec = [1,2,3,4]

vector = [
    [[1,2], [2,4]],
    [[2,2], [1,1]]
]

print (Polynomial_Regression(vec, 4, 2))