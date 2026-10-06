import numpy as np
import matplotlib.pyplot as plt


def verosimilitud(theta, muestra):

    muestra = np.array(muestra)

    valores = 1 + theta * muestra

    # Si algún valor es <= 0,
    # la verosimilitud no es válida
    if np.any(valores <= 0):
        return 0

    n = len(muestra)

    return (1 / 2**n) * np.prod(valores)


def negativo_log_verosimilitud(theta, muestra):

    muestra = np.array(muestra)

    valores = 1 + theta * muestra

    # Evitamos calcular log(0) o log de números negativos
    if np.any(valores <= 0):
        return np.inf

    n = len(muestra)

    return n * np.log(2) - np.sum(
        np.log(valores)
    )


def golden_search(func, a=-1, b=1, tol=1e-9):

    phi = (1 + np.sqrt(5)) / 2

    c = b - (b - a) / phi
    d = a + (b - a) / phi

    fc = func(c)
    fd = func(d)

    while abs(b - a) > tol:

        if fc < fd:

            b = d
            d = c
            fd = fc

            c = b - (b - a) / phi
            fc = func(c)

        else:

            a = c
            c = d
            fc = fd

            d = a + (b - a) / phi
            fd = func(d)

    return (a + b) / 2


def estimar_theta(muestra):

    theta_hat = golden_search(
        lambda theta:
            negativo_log_verosimilitud(theta, muestra)
    )

    return theta_hat


def graficar_maxima_verosimilitud(muestra):

    theta_hat = estimar_theta(muestra)

    theta_values = np.linspace(-1, 1, 500)

    log_values = np.array([
        negativo_log_verosimilitud(theta, muestra)
        for theta in theta_values
    ])

    plt.figure(figsize=(8, 5))

    plt.plot(
        theta_values,
        log_values,
        label=r'$\ell_{10}(\theta)$'
    )

    # Marcar el mínimo
    plt.scatter(
        theta_hat,
        negativo_log_verosimilitud(
            theta_hat,
            muestra
        ),
        s=80,
        label=fr'Mínimo: $\hat{{\theta}}={theta_hat:.4f}$'
    )

    # Línea vertical
    plt.axvline(
        theta_hat,
        linestyle='--',
        label=fr'$\hat{{\theta}}={theta_hat:.4f}$'
    )

    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\ell_{10}(\theta)$')

    plt.title(
        'Negativo del logaritmo de la verosimilitud'
    )

    plt.legend()
    plt.grid(alpha=0.25)

    plt.tight_layout()

    return theta_hat



