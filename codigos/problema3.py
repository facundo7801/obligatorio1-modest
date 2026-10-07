"""Problema 3: exploración y regresión con padre, madre y sexo.

Guardar este script y alturas.csv juntos en la carpeta codigos.
Desde la raíz del repositorio: python codigos/problema3.py
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error

# La ruta se construye desde este archivo, no desde la carpeta de la terminal.
ruta = Path(__file__).resolve().parent / "alturas.csv"
datos = pd.read_csv(ruta, encoding="utf-8-sig")

# Se acortan los nombres para trabajar con ellos sin alterar los valores.
datos = datos.rename(columns={
    "Response number": "id",
    "(Altura estudiante) Altura de la persona (estudiante del curso) en cm": "altura_hijo",
    "(sexo) Sexo biológico de la persona": "sexo",
    "(Altura padre) Altura padre en cm": "altura_padre",
    "(Altura madre) Altura madre en cm": "altura_madre",
})
columnas_modelo = ["altura_hijo", "sexo", "altura_padre", "altura_madre"]

print("Cantidad de respuestas:", len(datos))
print("\nPrimeras cinco respuestas:")
print(datos.head().to_string(index=False))
print("\nDatos faltantes por columna:")
print(datos.isna().sum())
print("\nRespuestas con datos faltantes para el modelo:")
print(datos.loc[datos[columnas_modelo].isna().any(axis=1)].to_string(index=False))
print("\nResumen de las alturas, en centímetros:")
print(datos[["altura_hijo", "altura_padre", "altura_madre"]].describe())

# Se conservan los datos originales en `datos` y se crea otra tabla.
# Compararemos todos los modelos sobre los mismos casos completos.
datos_limpios = datos.dropna(subset=columnas_modelo).copy()

print("\nCantidad de casos completos:", len(datos_limpios))
print("\nSexo registrado en los casos completos:")
print(datos_limpios["sexo"].value_counts())

# Altura media de ambos progenitores, en centímetros.
datos_limpios["altura_media_padres"] = (
    datos_limpios["altura_padre"] + datos_limpios["altura_madre"]
) / 2

print("\nCorrelaciones globales entre las alturas, en los 43 casos completos:")
print(datos_limpios[[
    "altura_hijo", "altura_padre", "altura_madre", "altura_media_padres"
]].corr().round(3).to_string())

print("\nResumen de altura del estudiante por sexo registrado:")
print(datos_limpios.groupby("sexo")["altura_hijo"].agg(
    ["count", "mean", "std", "min", "max"]
).round(2))

# Tres gráficos de dispersión: cada punto corresponde a un estudiante.
# El color distingue las dos categorías registradas en el archivo.
figura, ejes = plt.subplots(2, 2, figsize=(10, 7.5))
colores = {"Femenino": "#b45309", "Masculino": "#2563eb"}
predictores = [
    ("altura_padre", "Altura del padre (cm)"),
    ("altura_madre", "Altura de la madre (cm)"),
    ("altura_media_padres", "Altura media de los padres (cm)"),
]

for eje, (columna, etiqueta) in zip(ejes.flat, predictores):
    for sexo, color in colores.items():
        grupo = datos_limpios.loc[datos_limpios["sexo"] == sexo]
        eje.scatter(
            grupo[columna], grupo["altura_hijo"],
            color=color, label=f"{sexo} (n={len(grupo)})", alpha=0.8, s=35,
        )
    eje.set_xlabel(etiqueta)
    eje.set_ylabel("Altura del estudiante (cm)")
    eje.set_ylim(150, 195)
    eje.grid(alpha=0.2)
    eje.legend(fontsize=8)

# Diagrama de cajas: compara la distribución de alturas por categoría.
eje = ejes[1, 1]
categorias = list(colores)
alturas_por_sexo = [
    datos_limpios.loc[datos_limpios["sexo"] == sexo, "altura_hijo"]
    for sexo in categorias
]
cajas = eje.boxplot(alturas_por_sexo, patch_artist=True)
for caja, sexo in zip(cajas["boxes"], categorias):
    caja.set_facecolor(colores[sexo])
    caja.set_alpha(0.35)
eje.set_xticks([1, 2])
eje.set_xticklabels([
    f"{sexo}\n(n={len(grupo)})"
    for sexo, grupo in zip(categorias, alturas_por_sexo)
])
eje.set_ylabel("Altura del estudiante (cm)")
eje.set_ylim(150, 195)
eje.set_title("Distribución por sexo registrado")
eje.grid(axis="y", alpha=0.2)

figura.suptitle("Problema 3  Exploración de los 43 casos completos")
figura.tight_layout(rect=(0, 0, 1, 0.96))

# Si el script está en codigos, se guardará en Imagenes del repositorio.
carpeta_imagenes = Path(__file__).resolve().parent.parent / "Imagenes"
carpeta_imagenes.mkdir(exist_ok=True)
ruta_imagen = carpeta_imagenes / "problema3_exploracion.png"
figura.savefig(ruta_imagen, dpi=180)
plt.close(figura)
print("\nGráficas guardadas en:", ruta_imagen)

# -------------------------------------------------------------------
# MODELO: altura_hijo = a*altura_padre + b*altura_madre + d*S + c
# S = 1 para Masculino; S = 0 para Femenino.
# La media parental se utilizó para explorar, pero no entra en este modelo.
# -------------------------------------------------------------------
datos_limpios["S"] = datos_limpios["sexo"].map({
    "Masculino": 1,
    "Femenino": 0,
})
if datos_limpios["S"].isna().any():
    raise ValueError("Hay categorías de sexo no contempladas en la codificación.")

variables = ["altura_padre", "altura_madre", "S"]
X = datos_limpios[variables]
y = datos_limpios["altura_hijo"]

# VALIDACIÓN: en cada vuelta entrenamos con 42 personas y predecimos la otra.
# No se utiliza la altura de la persona apartada para ajustar coeficientes.
validacion = LeaveOneOut()
predicciones_loo = np.empty(len(datos_limpios))
predicciones_referencia = np.empty(len(datos_limpios))

for indices_entrenamiento, indice_validacion in validacion.split(X):
    X_entrenamiento = X.iloc[indices_entrenamiento]
    y_entrenamiento = y.iloc[indices_entrenamiento]
    X_validacion = X.iloc[indice_validacion]

    modelo_pliegue = LinearRegression()
    modelo_pliegue.fit(X_entrenamiento, y_entrenamiento)
    predicciones_loo[indice_validacion] = modelo_pliegue.predict(X_validacion)

    # Referencia sencilla: predecir la media de las alturas del entrenamiento.
    predicciones_referencia[indice_validacion] = y_entrenamiento.mean()

mse_loo = mean_squared_error(y, predicciones_loo)
rmse_loo = np.sqrt(mse_loo)
mse_referencia = mean_squared_error(y, predicciones_referencia)
rmse_referencia = np.sqrt(mse_referencia)

print("\nVALIDACIÓN LEAVE ONE OUT DEL MODELO CON TRES VARIABLES")
print(f"Personas evaluadas: {len(y)}")
print(f"MSE de validación: {mse_loo:.4f} cm²")
print(f"RMSE de validación: {rmse_loo:.4f} cm")
print(f"RMSE de la referencia que predice la media: {rmse_referencia:.4f} cm")

# AJUSTE FINAL: después de evaluar, aprovechamos los 43 casos completos.
# Sus coeficientes sirven para escribir la ecuación final.
# El error de validación anterior NO se calcula con este ajuste final.
modelo_final = LinearRegression()
modelo_final.fit(X, y)
a, b, d = modelo_final.coef_
c = modelo_final.intercept_

print("\nECUACIÓN FINAL AJUSTADA CON LOS 43 CASOS COMPLETOS")
print(f"altura_predicha = {a:.6f}*altura_padre "
      f"+ {b:.6f}*altura_madre + {d:.6f}*S + {c:.6f}")
print("S = 1 para Masculino y S = 0 para Femenino.")
print(f"a, coeficiente del padre: {a:.6f}")
print(f"b, coeficiente de la madre: {b:.6f}")
print(f"d, coeficiente de S: {d:.6f}")
print(f"c, intercepto: {c:.6f}")

# Ejemplo de aplicación del modelo final para padres de 180 y 165 cm.
ejemplo = pd.DataFrame({
    "altura_padre": [180, 180],
    "altura_madre": [165, 165],
    "S": [0, 1],
})
ejemplo["altura_predicha"] = modelo_final.predict(ejemplo)
print("\nEJEMPLO DE PREDICCIÓN, EN CENTÍMETROS")
print(ejemplo.round(3).to_string(index=False))

# Gráficas de validación: cada persona fue predicha sin participar del ajuste.
figura, ejes = plt.subplots(1, 2, figsize=(10, 4))
for sexo, color in colores.items():
    mascara = datos_limpios["sexo"].to_numpy() == sexo
    ejes[0].scatter(
        y.to_numpy()[mascara], predicciones_loo[mascara],
        color=color, label=sexo, alpha=0.8,
    )
    ejes[1].scatter(
        predicciones_loo[mascara],
        y.to_numpy()[mascara] - predicciones_loo[mascara],
        color=color, alpha=0.8,
    )

limite_inferior = min(y.min(), predicciones_loo.min()) - 2
limite_superior = max(y.max(), predicciones_loo.max()) + 2
ejes[0].plot(
    [limite_inferior, limite_superior],
    [limite_inferior, limite_superior], "--", color="gray",
)
ejes[0].set_xlabel("Altura observada (cm)")
ejes[0].set_ylabel("Altura predicha en validación (cm)")
ejes[0].set_title("Observado frente a predicho")
ejes[0].legend(fontsize=8)
ejes[1].axhline(0, color="gray", linestyle="--")
ejes[1].set_xlabel("Altura predicha en validación (cm)")
ejes[1].set_ylabel("Residuo: observado menos predicho (cm)")
ejes[1].set_title("Residuos de validación")
for eje in ejes:
    eje.grid(alpha=0.2)
figura.suptitle("Padre, madre y sexo  Validación leave one out")
figura.tight_layout()
ruta_validacion = carpeta_imagenes / "problema3_validacion.png"
figura.savefig(ruta_validacion, dpi=180)
plt.close(figura)
print("\nGráficas de validación guardadas en:", ruta_validacion)
