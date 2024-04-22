from math import pi
from math import sin

def calcular_bn_tren_de_pulsos (t, armonicos): # recibe el coeficiente de bn sin su sumatoria, la cantidad de armonicos que desea calcular y en que valor de t desea calcular la serie
    val_actuaL_serie_bn = 0
    for i in range(1, armonicos + 1): # calcula la sumatoria
        val_actuaL_serie_bn += bn_tren_de_pulsos(t, i)
    
    return val_actuaL_serie_bn

def bn_tren_de_pulsos (t, n): 
    return (2/(n*pi)) * (1 - (-1)**n) * (sin(2*pi*n*t)) # coeficiente bn que en este caso es todo f(t)

def calcular_bn_diente_sierra (t, armonicos): # recibe el coeficiente de bn sin su sumatoria, la cantidad de armonicos que desea calcular y en que valor de t desea calcular la serie
    val_actuaL_serie_bn = 0
    for i in range(1, armonicos + 1): # calcula la sumatoria
        val_actuaL_serie_bn += bn_diente_de_sierra(t, i)
    
    return val_actuaL_serie_bn

def bn_diente_de_sierra (t, n):
    return (((-1)*((-1)**n))/(pi*n)) * (sin(2*pi*n*t))