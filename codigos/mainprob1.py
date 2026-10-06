from problema1_1 import (
    simular_muestra,
    graficar_aceptacion_rechazo
)

from problema1_2 import (
    graficar_maxima_verosimilitud,
    estimar_muchas_muestras,
    calcular_metricas
)

import matplotlib.pyplot as plt
import numpy as np

# PARTE 1

n = 10

muestra, aceptados_x, aceptados_y, rechazados_x, rechazados_y = \
    simular_muestra(n)


print("===================================")
print("MUESTRA")
print("===================================")

print(muestra)


print("\n===================================")
print("ACEPTADOS")
print("===================================")

print(aceptados_x)


print("\n===================================")
print("RECHAZADOS")
print("===================================")

print(rechazados_x)



# GRÁFICA ACEPTACIÓN-RECHAZO

graficar_aceptacion_rechazo(
    aceptados_x,
    aceptados_y,
    rechazados_x,
    rechazados_y
)



# PARTE 2

theta_hat = graficar_maxima_verosimilitud(
    muestra
)


print("\n===================================")
print("MÁXIMA VEROSIMILITUD")
print("===================================")

print("Theta estimado:")
print(theta_hat)




#PARTE 3


theta_real = -1 / 3

cantidad_muestras = 50

estimaciones = estimar_muchas_muestras(
    simular_muestra,
    cantidad_muestras,
    10
)


# Calcular MSE, sesgo y varianza

mse, sesgo, varianza = calcular_metricas(
    estimaciones,
    theta_real
)


print("\n===================================")
print("PARTE 3")
print("===================================")

print("Cantidad de muestras:")
print(cantidad_muestras)

print("\nEstimaciones de theta:")

for i, theta_hat in enumerate(estimaciones, start=1):
    print(f"Muestra {i}: {theta_hat:.6f}")


print("\nMSE:")
print(mse)

print("\nSesgo:")
print(sesgo)

print("\nVarianza:")
print(varianza)


#MOSTRAR GRAFICAS
plt.show()

