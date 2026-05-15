from utils import db_connect
engine = db_connect()

# PASO 1 - Carga del dataset

import pandas as pd

URL = "https://breathecode.herokuapp.com/asset/internal-link?id=733&path=demographic_health_data.csv"
df = pd.read_csv(URL)

print(f"Filas: {df.shape[0]} | Columnas: {df.shape[1]}")
df.head()

print(list(df.columns))

# PASO 2: Análisis inicial del dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

URL = "https://breathecode.herokuapp.com/asset/internal-link?id=733&path=demographic_health_data.csv"
df = pd.read_csv(URL)

print(f"Filas: {df.shape[0]} | Columnas: {df.shape[1]}")
print("\nPrimeras filas:")
df.head()

# Análisis de nulos

null_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)

print("Columnas con más del 50% de nulos:")
print(null_pct[null_pct > 50])

# solo grafico si hay columnas con nulos
if null_pct[null_pct > 0].shape[0] > 0:
    plt.figure(figsize=(12, 4))
    null_pct[null_pct > 0].plot(kind="bar", color="salmon")
    plt.axhline(50, color="red", linestyle="--", label="umbral 50%")
    plt.title("% de nulos por columna")
    plt.ylabel("% nulos")
    plt.xticks(rotation=90, fontsize=7)
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("El dataset no tiene valores nulos, no hay nada que graficar")


#Limpieza del dataset
TARGET = "anycondition_prevalence"

# elimino columnas con más del 50% de nulos
cols_muchos_nulos = null_pct[null_pct > 50].index.tolist()
df.drop(columns=cols_muchos_nulos, inplace=True)

# elimino columnas de intervalos de confianza y conteos (no aportan)
ci_cols   = [c for c in df.columns if "CI" in c or "Lower" in c or "Upper" in c]
leak_cols = [c for c in df.columns if "_number" in c]
df.drop(columns=ci_cols + leak_cols, inplace=True)

# elimino filas donde el target es nulo
df.dropna(subset=[TARGET], inplace=True)

# convierto todo a numérico (elimina columnas de texto como nombre de condado)
df = df.apply(pd.to_numeric, errors="coerce")
df.drop(columns=df.columns[df.isnull().all()].tolist(), inplace=True)

# imputo los nulos restantes con la mediana
df.fillna(df.median(numeric_only=True), inplace=True)

print(f"Dataset limpio: {df.shape[0]} filas x {df.shape[1]} columnas")
print(f"Nulos restantes: {df.isnull().sum().sum()}")

# Análisis de la variable objetivo
print(f"Variable objetivo: {TARGET}")
print(df[TARGET].describe())

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
df[TARGET].hist(bins=40, color="steelblue", edgecolor="white")
plt.title("Distribución de anycondition_prevalence")
plt.xlabel("Prevalencia (%)")
plt.ylabel("Frecuencia")

plt.subplot(1, 2, 2)
df[TARGET].plot(kind="box", color="steelblue")
plt.title("Boxplot de anycondition_prevalence")

plt.tight_layout()
plt.show()

# Correlación con la variable objetivo

corr_target = df.corr(numeric_only=True)[TARGET].drop(TARGET).sort_values()

print("Top 5 correlaciones negativas:")
print(corr_target.head(5))
print("\nTop 5 correlaciones positivas:")
print(corr_target.tail(5))

# elimino variables con correlación muy baja (no aportan al modelo)
baja_corr = corr_target[corr_target.abs() < 0.05].index.tolist()
df.drop(columns=baja_corr, inplace=True)

print(f"\nVariables eliminadas por baja correlación: {len(baja_corr)}")
print(f"Columnas finales: {df.shape[1]}")

# Mapa de calor y división train/test

from sklearn.model_selection import train_test_split

# mapa de calor de correlaciones entre variables finales
plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(numeric_only=True), cmap="RdYlGn", center=0, linewidths=0.3)
plt.title("Correlaciones entre variables")
plt.tight_layout()
plt.show()

# divido en train y test
X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

# Paso 3: Preparación de datos
# (reutilizo la limpieza del paso 2)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

URL = "https://breathecode.herokuapp.com/asset/internal-link?id=733&path=demographic_health_data.csv"
df = pd.read_csv(URL)

TARGET = "anycondition_prevalence"

# limpieza básica
null_pct = (df.isnull().sum() / len(df) * 100)
df.drop(columns=null_pct[null_pct > 50].index.tolist(), inplace=True)

ci_cols   = [c for c in df.columns if "CI" in c or "Lower" in c or "Upper" in c]
leak_cols = [c for c in df.columns if "_number" in c]
df.drop(columns=ci_cols + leak_cols, inplace=True)

df.dropna(subset=[TARGET], inplace=True)
df = df.apply(pd.to_numeric, errors="coerce")
df.drop(columns=df.columns[df.isnull().all()].tolist(), inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)

corr_target = df.corr(numeric_only=True)[TARGET].drop(TARGET)
baja_corr   = corr_target[corr_target.abs() < 0.05].index.tolist()
df.drop(columns=baja_corr, inplace=True)

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# escalo los datos porque Lasso lo necesita
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print("datos listos para entrenar")
print(f"train: {X_train.shape} | test: {X_test.shape}")


# Regresión Lineal base
model_lr = LinearRegression()
model_lr.fit(X_train_sc, y_train)

y_pred_lr = model_lr.predict(X_test_sc)

r2_lr   = r2_score(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

print("Regresión Lineal:")
print(f"  R2   = {r2_lr:.4f}")
print(f"  RMSE = {rmse_lr:.4f}")

# gráfica real vs predicho
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred_lr, alpha=0.4, color="steelblue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Valores reales")
plt.ylabel("Valores predichos")
plt.title(f"Regresión Lineal — R2 = {r2_lr:.4f}")
plt.tight_layout()
plt.show()


# Modelo Lasso y comparativa
model_lasso = Lasso(alpha=1.0)
model_lasso.fit(X_train_sc, y_train)

y_pred_lasso = model_lasso.predict(X_test_sc)

r2_lasso   = r2_score(y_test, y_pred_lasso)
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))

print("Lasso (alpha=1.0):")
print(f"  R2   = {r2_lasso:.4f}")
print(f"  RMSE = {rmse_lasso:.4f}")

print("\nComparativa:")
print(f"  Regresión Lineal -> R2: {r2_lr:.4f} | RMSE: {rmse_lr:.4f}")
print(f"  Lasso            -> R2: {r2_lasso:.4f} | RMSE: {rmse_lasso:.4f}")


# Cómo cambia el R2 según el alpha del Lasso
alphas = np.arange(0.01, 20, 0.1)
r2_scores = []

for a in alphas:
    lasso_temp = Lasso(alpha=a, max_iter=5000)
    lasso_temp.fit(X_train_sc, y_train)
    r2_scores.append(r2_score(y_test, lasso_temp.predict(X_test_sc)))

plt.figure(figsize=(10, 5))
plt.plot(alphas, r2_scores, color="steelblue")
plt.xlabel("Alpha")
plt.ylabel("R2")
plt.title("Evolución del R2 según el alpha en Lasso")
plt.tight_layout()
plt.show()

mejor_alpha = alphas[np.argmax(r2_scores)]
print(f"Mejor alpha encontrado: {mejor_alpha:.2f}")
print(f"R2 con ese alpha: {max(r2_scores):.4f}")


# PASO 4: Optimización del modelo Lasso con GridSearchCV
from sklearn.model_selection import GridSearchCV

# pruebo distintos valores de alpha para encontrar el mejor
param_grid = {"alpha": [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]}

lasso_grid = GridSearchCV(
    Lasso(max_iter=5000),
    param_grid,
    cv=5,
    scoring="r2"
)
lasso_grid.fit(X_train_sc, y_train)

print(f"Mejor alpha encontrado: {lasso_grid.best_params_['alpha']}")
print(f"R2 medio en validación: {lasso_grid.best_score_:.4f}")


# Evaluación del modelo optimizado
mejor_modelo = lasso_grid.best_estimator_

y_pred_opt = mejor_modelo.predict(X_test_sc)

r2_opt   = r2_score(y_test, y_pred_opt)
rmse_opt = np.sqrt(mean_squared_error(y_test, y_pred_opt))

print("Modelo optimizado:")
print(f"  R2   = {r2_opt:.4f}")
print(f"  RMSE = {rmse_opt:.4f}")

# comparativa final con los modelos anteriores
print("\nResumen final:")
print(f"  Regresión Lineal  -> R2: {r2_lr:.4f} | RMSE: {rmse_lr:.4f}")
print(f"  Lasso (alpha=1.0) -> R2: {r2_lasso:.4f} | RMSE: {rmse_lasso:.4f}")
print(f"  Lasso optimizado  -> R2: {r2_opt:.4f} | RMSE: {rmse_opt:.4f}")


# Gráfica del modelo optimizado

plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred_opt, alpha=0.4, color="steelblue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Valores reales")
plt.ylabel("Valores predichos")
plt.title(f"Lasso optimizado — R2 = {r2_opt:.4f}")
plt.tight_layout()
plt.show()

