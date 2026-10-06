import numpy as np
import matplotlib.pyplot as plt


def simular_muestra(n):

    muestra = []

    # Para la gráfica
    aceptados_x = []
    aceptados_y = []

    rechazados_x = []
    rechazados_y = []

    while len(muestra) < n:

        # Generar candidato X ~ U(-1,1)
        x = np.random.uniform(-1, 1)

        # Generar U ~ U(0,1)
        u = np.random.uniform(0, 1)

        # Constante de aceptación-rechazo
        M = 4 / 3

        # Densidad de la uniforme U(-1,1)
        g = 1 / 2

        # Altura del punto
        y = u * M * g

        # Densidad objetivo
        f = 0.5 * (1 - x / 3)

        # Aceptación o rechazo
        if y <= f:

            # Guardamos el valor en la muestra
            muestra.append(x)

            # Guardamos el punto para graficarlo
            aceptados_x.append(x)
            aceptados_y.append(y)

        else:

            # Guardamos el punto rechazado
            rechazados_x.append(x)
            rechazados_y.append(y)

    return (
        np.array(muestra),
        np.array(aceptados_x),
        np.array(aceptados_y),
        np.array(rechazados_x),
        np.array(rechazados_y)
    )


def graficar_aceptacion_rechazo(
    aceptados_x,
    aceptados_y,
    rechazados_x,
    rechazados_y
):

    theta = -1 / 3

    x = np.linspace(-1, 1, 500)

    # Densidad objetivo
    f = 0.5 * (1 + theta * x)

    # Densidad propuesta U(-1,1)
    g = np.full_like(x, 0.5)

    # Constante M
    M = 4 / 3

    # Envolvente M*g(x)
    Mg = M * g

    plt.figure(figsize=(10, 6))

    # Densidad objetivo
    plt.plot(
        x,
        f,
        label=r'$f(x;\theta=-1/3)=\frac{1}{2}(1-x/3)$'
    )

    # Envolvente
    plt.plot(
        x,
        Mg,
        '--',
        label=r'$Mg(x)$, $M=4/3$'
    )

    # Región de aceptación-rechazo
    plt.fill_between(
        x,
        f,
        Mg,
        alpha=0.2
    )

    # Puntos aceptados
    plt.scatter(
        aceptados_x,
        aceptados_y,
        color='green',
        marker='$✓$',
        s=120,
        label='Aceptado'
    )

    # Puntos rechazados
    plt.scatter(
        rechazados_x,
        rechazados_y,
        color='red',
        marker='x',
        s=80,
        linewidths=2,
        label='Rechazado'
    )

    plt.xlabel('x')
    plt.ylabel('densidad')

    plt.title(
        'Método de aceptación-rechazo para θ = -1/3'
    )

    plt.legend()
    plt.grid(alpha=0.25)

    plt.tight_layout()
