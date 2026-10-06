#50 MUESTRAS TAMANIO 10
#SE USA LO MISMO QUE PARA LA PARTE 

import numpy as np
from problema1_2 import estimar_theta


def problema1_3():

    theta_real = -1 / 3

    cantidad_muestras = 50
    n = 10

    estimaciones = []

    for i in range(cantidad_muestras):

        muestra = []

        while len(muestra) < n:

            # X ~ U(-1,1)
            x = np.random.uniform(-1, 1)

            # U ~ U(0,1)
            u = np.random.uniform(0, 1)

            # Constante de aceptación-rechazo
            M = 4 / 3

            # Densidad propuesta
            g = 1 / 2

            # Altura del punto
            y = u * M * g

            # Densidad objetivo
            f = 0.5 * (1 - x / 3)

            if y <= f:
                muestra.append(x)

        # Estimar theta de esta muestra
        theta_hat = estimar_theta(muestra)

        estimaciones.append(theta_hat)

    estimaciones = np.array(estimaciones)
    print("\nEstimaciones:")
    for i, theta_hat in enumerate(estimaciones, start=1):
        print(f"Muestra {i}: θ̂ = {theta_hat:.4f}")


    # Media de las estimaciones
    media = np.mean(estimaciones)

    # Sesgo
    sesgo = media - theta_real

    # Varianza
    varianza = np.mean(
        (estimaciones - media) ** 2
    )

    # MSE
    mse = np.mean(
        (estimaciones - theta_real) ** 2
    )

    print("Resultados")
    print("--------------------")
    print(f"θ verdadero = {theta_real:.4f}")
    print(f"Media = {media:.4f}")
    print(f"Sesgo = {sesgo:.4f}")
    print(f"Varianza = {varianza:.4f}")
    print(f"MSE = {mse:.4f}")


problema1_3()