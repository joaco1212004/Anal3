from math import cos
from math import pi
import numpy as np

def calcular_an_triangular (t, armonicos): # recibe el coeficiente de bn sin su sumatoria, la cantidad de armonicos que desea calcular y en que valor de t desea calcular la serie
    val_actuaL_serie_bn = 0
    for i in range(1, armonicos + 1): # calcula la sumatoria
        val_actuaL_serie_bn += an_triangular(t, i)
    
    return val_actuaL_serie_bn

def an_triangular(t, n):
    return 4 * ((((-1)**n)-1)/((pi**2)*(n**2))) * (cos(2*pi*n*t))
