# !pip install statsmodels

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

def display(df):
    print(df)

SEMILLA = 42


kf = KFold(n_splits=5, shuffle=True, random_state=SEMILLA)

#  color fijo por métrica
C_TRAIN, C_VAL, C_CV, C_REF = '#2a78d6', '#eb6834', '#1baf7a', '#555555'

plt.rcParams['figure.dpi'] = 110
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

from pathlib import Path
carpeta_img = Path(__file__).resolve().parent.parent / 'Imagenes' / 'problema2'
carpeta_img.mkdir(parents=True, exist_ok=True)

r"""## 1. Carga y Análisis Inicial"""

data = sm.datasets.get_rdataset('Boston', package='MASS').data
data = data.rename(columns={'medv': 'MEDV'})

print(f"{data.shape[0]} observaciones, {data.shape[1] - 1} atributos + MEDV")
print("Valores faltantes:", data.isna().sum().sum())
display(data.describe().T.round(2))

# MEDV está topeado en 50 (miles de USD): esos valores significan "50 o más"
print("Viviendas con MEDV = 50 (valor tope):", (data['MEDV'] == 50).sum())

corr = data.corr()

plt.figure(figsize=(10, 8))
mascara = np.triu(np.ones_like(corr, dtype=bool), k=1)   # mostramos sólo el triángulo inferior
sns.heatmap(corr, mask=mascara, annot=True, fmt='.2f', cmap='RdBu_r', vmin=-1, vmax=1,
            annot_kws={'size': 8})
plt.title('Matriz de correlación')
plt.tight_layout()
plt.savefig(carpeta_img / 'matriz_correlacion.png', dpi=200, bbox_inches='tight')
plt.show()

# (a) Correlación con MEDV, CON signo: el signo dice si el precio sube o baja con el atributo
print("Correlación con MEDV (ordenada por valor absoluto):")
print(corr['MEDV'].drop('MEDV').sort_values(key=abs, ascending=False).round(3).to_string())

# (b) Pares de atributos muy correlacionados ENTRE SÍ (multicolinealidad)
c = corr.drop(index='MEDV', columns='MEDV')
pares = c.where(np.triu(np.ones(c.shape, dtype=bool), k=1)).stack()
pares = pares[pares.abs() >= 0.7].sort_values(key=abs, ascending=False)
print("\nPares de atributos con |r| >= 0.7:")
print(pares.round(3).to_string())

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(data['MEDV'], bins=30, color=C_TRAIN, edgecolor='white')
axes[0].set_xlabel('MEDV (miles de USD)')
axes[0].set_ylabel('Cantidad de viviendas')
axes[0].set_title('Distribución de MEDV (pico artificial en 50)')

# Las dos variables más correlacionadas con MEDV: LSTAT muestra una relación curva
for ax, col in zip(axes[1:], ['lstat', 'rm']):
    ax.scatter(data[col], data['MEDV'], s=12, alpha=0.5, color=C_TRAIN)
    ax.set_xlabel(col.upper())
    ax.set_ylabel('MEDV')
    ax.set_title(f'{col.upper()} vs MEDV  (r = {corr.loc[col, "MEDV"]:.2f})')

plt.tight_layout()
plt.savefig(carpeta_img / 'exploratorio_medv.png', dpi=200, bbox_inches='tight')
plt.show()

r"""## 2. Preprocesamiento"""

X = data.drop(columns=['MEDV'])
y = data['MEDV'].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEMILLA)
print(f"Entrenamiento: {len(X_train)} muestras  |  Validación: {len(X_val)} muestras")


def mse_cv(modelo):
    # Métrica MSE por validación cruzada (promedio y desvío estándar) sobre X_train.
    scores = -cross_val_score(modelo, X_train, y_train, scoring='neg_mean_squared_error', cv=kf)
    return scores.mean(), scores.std()

r"""## 3. Modelo lineal"""

lin = LinearRegression().fit(X_train, y_train)

mse_train_lin = mean_squared_error(y_train, lin.predict(X_train))
mse_val_lin   = mean_squared_error(y_val, lin.predict(X_val))
cv_lin, cv_lin_sd = mse_cv(LinearRegression())

print(f"MSE entrenamiento: {mse_train_lin:6.2f}")
print(f"MSE CV (5 folds):  {cv_lin:6.2f} ± {cv_lin_sd:.2f}")
print(f"MSE validación:    {mse_val_lin:6.2f}")
print(f"R² validación:     {r2_score(y_val, lin.predict(X_val)):6.3f}")

r"""## 4. Modelo polinomial

Ojo con la cantidad de columnas: con 13 atributos, el grado 2 genera 104 y el grado 3 genera **559**, más que las 404 filas de entrenamiento. Además `CHAS` es binaria, así que $\text{CHAS}^2 = \text{CHAS}^3 = \text{CHAS}$: hay columnas repetidas exactas.
"""

grados = [1, 2, 3]
filas = []

for d in grados:
    modelo = make_pipeline(PolynomialFeatures(degree=d, include_bias=False),
                           StandardScaler(),
                           LinearRegression())
    modelo.fit(X_train, y_train)
    cv_m, cv_sd = mse_cv(modelo)
    filas.append({
        'grado': d,
        'n_columnas': modelo.named_steps['polynomialfeatures'].n_output_features_,
        'MSE train': mean_squared_error(y_train, modelo.predict(X_train)),
        'MSE CV': cv_m,
        'CV desvío': cv_sd,
        'MSE val': mean_squared_error(y_val, modelo.predict(X_val)),
        '||beta||': np.linalg.norm(modelo.named_steps['linearregression'].coef_),
        'modelo': modelo,
    })

res_ols = pd.DataFrame(filas).set_index('grado')
display(res_ols.drop(columns='modelo').round(2))

r"""## 5. Regularización Ridge

Para **cada grado** barremos $\lambda$ en escala logarítmica y nos quedamos con el que minimiza el MSE de CV. Lo hacemos para los tres grados para poder comparar con y sin regularización en cada caso.
"""

alphas = np.logspace(-3, 5, 50)
curvas = {}      # grado -> (MSE CV, MSE train) para cada lambda
filas = []

for d in grados:
    cv_vals, tr_vals = [], []
    for a in alphas:
        m = make_pipeline(PolynomialFeatures(degree=d, include_bias=False),
                          StandardScaler(), Ridge(alpha=a))
        cv_vals.append(mse_cv(m)[0])
        m.fit(X_train, y_train)
        tr_vals.append(mean_squared_error(y_train, m.predict(X_train)))
    cv_vals, tr_vals = np.array(cv_vals), np.array(tr_vals)
    curvas[d] = (cv_vals, tr_vals)

    a_opt = alphas[np.argmin(cv_vals)]
    mejor = make_pipeline(PolynomialFeatures(degree=d, include_bias=False),
                          StandardScaler(), Ridge(alpha=a_opt)).fit(X_train, y_train)
    filas.append({
        'grado': d,
        'lambda*': a_opt,
        'MSE train': mean_squared_error(y_train, mejor.predict(X_train)),
        'MSE CV': cv_vals.min(),
        'CV desvío': mse_cv(mejor)[1],
        'MSE val': mean_squared_error(y_val, mejor.predict(X_val)),
        '||beta||': np.linalg.norm(mejor.named_steps['ridge'].coef_),
        'modelo': mejor,
    })

res_ridge = pd.DataFrame(filas).set_index('grado')
display(res_ridge.drop(columns='modelo').round(3))

r"""## 6. Gráficas"""

# Mejor modelo de cada familia, elegido por MSE de CV (nunca por validación)
d_ols   = res_ols.loc[[2, 3], 'MSE CV'].idxmin()     # mejor polinomial sin regularizar
d_ridge = res_ridge['MSE CV'].idxmin()                # mejor Ridge
print(f"Mejor polinomial sin regularizar: grado {d_ols}")
print(f"Mejor Ridge: grado {d_ridge}, lambda* = {res_ridge.loc[d_ridge, 'lambda*']:.3g}")


def graf_mse_grado(ax):
    # color = métrica; línea sólida = sin regularizar, punteada = Ridge con su lambda*
    for col, color, nombre in [('MSE train', C_TRAIN, 'Entrenamiento'),
                               ('MSE CV', C_CV, 'CV'),
                               ('MSE val', C_VAL, 'Validación')]:
        ax.plot(grados, res_ols[col], 'o-', color=color, lw=2, label=f'{nombre} (sin reg.)')
        ax.plot(grados, res_ridge[col], 's--', color=color, lw=2, label=f'{nombre} (Ridge)')
    ax.set_yscale('log')
    ax.set_xticks(grados)
    ax.set_xlabel('Grado del polinomio')
    ax.set_ylabel('MSE (escala log)')
    ax.set_title('MSE en función del grado')
    ax.legend(fontsize=8)


def graf_mse_lambda(ax, d):
    cv_vals, tr_vals = curvas[d]
    a_opt = res_ridge.loc[d, 'lambda*']
    ax.plot(alphas, tr_vals, '-', color=C_TRAIN, lw=2, label='Entrenamiento')
    ax.plot(alphas, cv_vals, '-', color=C_CV, lw=2, label='CV (5 folds)')
    ax.axvline(a_opt, color=C_REF, ls=':', lw=1.5, label=rf'λ* = {a_opt:.2g}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    numeros = plt.FuncFormatter(lambda v, _: f'{v:g}') 
    ax.yaxis.set_major_formatter(numeros)
    ax.yaxis.set_minor_formatter(numeros)
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel('MSE (escala log)')
    ax.set_title(rf'Ridge grado {d}: MSE en función de λ')
    ax.legend(fontsize=8)


def graf_pred(ax, modelo, titulo):
    y_hat = modelo.predict(X_val)
    lo, hi = min(0, y_hat.min()), max(55, y_hat.max())
    ax.plot([lo, hi], [lo, hi], color=C_REF, ls='--', lw=1, label='predicción = real')
    ax.scatter(y_val, y_hat, s=18, alpha=0.7, color=C_VAL, label='viviendas de validación')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.set_xlabel('MEDV real')
    ax.set_ylabel('MEDV predicho')
    ax.set_title(f'{titulo}\nMSE val = {mean_squared_error(y_val, y_hat):.1f}')
    ax.legend(fontsize=8, loc='upper left')

fig, ax = plt.subplots(figsize=(7, 4.5))
graf_mse_grado(ax)
plt.tight_layout()
plt.savefig(carpeta_img / 'mse_vs_grado.png', dpi=200, bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, d in zip(axes, [2, 3]):
    graf_mse_lambda(ax, d)
plt.tight_layout()
plt.savefig(carpeta_img / 'mse_vs_lambda.png', dpi=200, bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
graf_pred(axes[0], res_ols.loc[d_ols, 'modelo'], f'Grado {d_ols} sin regularizar')
graf_pred(axes[1], res_ridge.loc[d_ridge, 'modelo'],
          rf'Grado {d_ridge} con Ridge (λ = {res_ridge.loc[d_ridge, "lambda*"]:.2g})')
plt.tight_layout()
plt.savefig(carpeta_img / 'prediccion_vs_real.png', dpi=200, bbox_inches='tight')
plt.show()


r"""## 7. Tabla resumen"""

cols = ['lambda*', 'MSE train', 'MSE CV', 'CV desvío', 'MSE val']
resumen = pd.concat({
    'Sin regularizar': res_ols.assign(**{'lambda*': 0.0})[cols],
    'Ridge': res_ridge[cols],
}, names=['modelo', 'grado'])
display(resumen.round(2))



fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
graf_mse_grado(axes[0])
graf_mse_lambda(axes[1], d_ridge)
graf_pred(axes[2], res_ridge.loc[d_ridge, 'modelo'],
          f'Mejor modelo: Ridge grado {d_ridge}')
plt.tight_layout()
plt.savefig(carpeta_img / 'figura_informe.png', dpi=200, bbox_inches='tight')
plt.show()