# Proyecto1-LabEstadistico

# 🌸 Iris Flower Regression Analysis

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0+-orange.svg)
![Statsmodels](https://img.shields.io/badge/Statsmodels-0.13+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Un análisis completo de regresión sobre el clásico dataset Iris, comparando modelos OLS, Ridge, Lasso y ElasticNet.**

[Descripción](#-descripción) •
[Dataset](#-dataset) •
[Metodología](#-metodología) •
[Resultados](#-resultados) •
[Instalación](#-instalación) •
[Referencias](#-referencias)

</div>

---

## 📋 Descripción

Este proyecto realiza un análisis exhaustivo de regresión lineal sobre el dataset Iris, uno de los conjuntos de datos más icónicos en el campo del Machine Learning. El objetivo es predecir características morfológicas de las flores Iris utilizando diferentes técnicas de regresión y evaluar la significancia estadística de cada factor.

### Objetivos del Proyecto

- 🎯 Predecir `petal_length` usando características del sépalo, pétalo y especie
- 🎯 Predecir `petal_width` usando características del sépalo, pétalo y especie  
- 🎯 Predecir `sepal_length` usando características del sépalo, pétalo y especie
- 📊 Comparar el rendimiento de 4 tipos de regresión
- 🔍 Analizar la significancia estadística de cada variable

---

## 🌺 Dataset

### Origen

El dataset fue introducido por **Ronald Fisher** en 1936 en su paper *"The use of multiple measurements in taxonomic problems"*. Los datos fueron recolectados por **Edgar Anderson** para estudiar la variación morfológica de tres especies de Iris.

### Estructura

```
📁 Data/
└── IRIS.csv
```

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `sepal_length` | Float | Longitud del sépalo (cm) |
| `sepal_width` | Float | Ancho del sépalo (cm) |
| `petal_length` | Float | Longitud del pétalo (cm) |
| `petal_width` | Float | Ancho del pétalo (cm) |
| `species` | Categórica | Especie de la flor |

### Distribución de Especies

```
┌─────────────────────────────────────────────────────────┐
│                    150 Muestras Totales                 │
├───────────────────┬───────────────────┬─────────────────┤
│   Iris-setosa     │  Iris-versicolor  │  Iris-virginica │
│       50          │        50         │       50        │
│      33.3%        │       33.3%       │      33.3%      │
└───────────────────┴───────────────────┴─────────────────┘
```

### Visualización de Características

```
                    Estadísticas Descriptivas
    ┌────────────────┬────────┬────────┬────────┬────────┐
    │    Variable    │  Min   │  Mean  │  Max   │  Std   │
    ├────────────────┼────────┼────────┼────────┼────────┤
    │  sepal_length  │  4.3   │  5.84  │  7.9   │  0.83  │
    │  sepal_width   │  2.0   │  3.05  │  4.4   │  0.43  │
    │  petal_length  │  1.0   │  3.76  │  6.9   │  1.76  │
    │  petal_width   │  0.1   │  1.20  │  2.5   │  0.76  │
    └────────────────┴────────┴────────┴────────┴────────┘
```

---

## 🔬 Metodología

### Pipeline de Análisis

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PIPELINE DE REGRESIÓN                          │
└─────────────────────────────────────────────────────────────────────────────┘

     ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
     │  CARGAR  │────▶│ LIMPIAR  │────▶│TRANSFORMAR────▶│ SEPARAR  │
     │  DATOS   │     │  DATOS   │     │  DATOS   │     │ VARIABLES│
     └──────────┘     └──────────┘     └──────────┘     └──────────┘
                                                              │
          ┌───────────────────────────────────────────────────┘
          ▼
     ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
     │  TRAIN   │────▶│ ESCALAR  │────▶│ AJUSTAR  │────▶│ EVALUAR  │
     │  TEST    │     │  DATOS   │     │ MODELOS  │     │    R²    │
     │  SPLIT   │     │          │     │          │     │          │
     └──────────┘     └──────────┘     └──────────┘     └──────────┘
                                                              │
          ┌───────────────────────────────────────────────────┘
          ▼
     ┌──────────┐     ┌──────────┐
     │ ANÁLISIS │────▶│ CONCLU-  │
     │ SIGNIF.  │     │ SIONES   │
     └──────────┘     └──────────┘
```

### Transformaciones Aplicadas

| Transformación | Descripción | Propósito |
|----------------|-------------|-----------|
| **One-Hot Encoding** | `species` → variables dummy | Convertir categórica a numérica |
| **StandardScaler** | Media=0, Std=1 | Normalizar para modelos penalizados |
| **Train-Test Split** | 70% / 30% | Validar generalización del modelo |

### Modelos Implementados

```
┌─────────────────────────────────────────────────────────────┐
│                    MODELOS DE REGRESIÓN                     │
├─────────────────┬───────────────────────────────────────────┤
│                 │                                           │
│   OLS           │   Mínimos cuadrados ordinarios            │
│   (Sin penal.)  │   Minimiza: Σ(y - ŷ)²                     │
│                 │                                           │
├─────────────────┼───────────────────────────────────────────┤
│                 │                                           │
│   Ridge         │   Penalización L2                         │
│   (L2)          │   Minimiza: Σ(y - ŷ)² + λΣβⱼ²             │
│                 │                                           │
├─────────────────┼───────────────────────────────────────────┤
│                 │                                           │
│   Lasso         │   Penalización L1                         │
│   (L1)          │   Minimiza: Σ(y - ŷ)² + λΣ|βⱼ|            │
│                 │                                           │
├─────────────────┼───────────────────────────────────────────┤
│                 │                                           │
│   ElasticNet    │   Combinación L1 + L2                     │
│   (L1 + L2)     │   Minimiza: Σ(y - ŷ)² + λ₁Σ|βⱼ| + λ₂Σβⱼ² │
│                 │                                           │
└─────────────────┴───────────────────────────────────────────┘
```

---

## 📊 Resultados

### Comparación de R² por Modelo

```
                         R² de Prueba (Test)
     
     1.0 ┤
         │
     0.9 ┤  ████                    ████
         │  ████  ████              ████  ████
     0.8 ┤  ████  ████  ████  ████  ████  ████  ████  ████
         │  ████  ████  ████  ████  ████  ████  ████  ████
     0.7 ┤  ████  ████  ████  ████  ████  ████  ████  ████
         │  ████  ████  ████  ████  ████  ████  ████  ████          ████
     0.6 ┤  ████  ████  ████  ████  ████  ████  ████  ████  ████    ████  ████
         │  ████  ████  ████  ████  ████  ████  ████  ████  ████    ████  ████  ████
     0.5 ┤  ████  ████  ████  ████  ████  ████  ████  ████  ████    ████  ████  ████
         │  ████  ████  ████  ████  ████  ████  ████  ████  ████    ████  ████  ████
     0.0 ┼──────────────────────────────────────────────────────────────────────────
              OLS Ridge Lasso Elas   OLS Ridge Lasso Elas   OLS Ridge Lasso Elas
         │◄────── Modelo 1 ──────▶│◄────── Modelo 2 ──────▶│◄────── Modelo 3 ──────▶│
              petal_length              petal_width              sepal_length
```

### Resumen de Resultados

| Modelo | Variable Target | R² Train | R² Test | Mejor Versión |
|--------|-----------------|----------|---------|---------------|
| **Modelo 1** | petal_length | ~0.98 | ~0.96 | OLS |
| **Modelo 2** | petal_width | ~0.95 | ~0.94 | OLS |
| **Modelo 3** | sepal_length | ~0.85 | ~0.78 | OLS |

### Análisis de Significancia

```
┌────────────────────────────────────────────────────────────────┐
│              SIGNIFICANCIA ESTADÍSTICA (p < 0.05)              │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  MODELO 1 (petal_length):                                      │
│  ├─ petal_width ─────────────────── ✓ SIGNIFICATIVO            │
│  ├─ species_Iris-versicolor ─────── ✓ SIGNIFICATIVO            │
│  └─ species_Iris-virginica ──────── ✓ SIGNIFICATIVO            │
│                                                                │
│  MODELO 2 (petal_width):                                       │
│  ├─ petal_length ────────────────── ✓ SIGNIFICATIVO            │
│  └─ species_Iris-virginica ──────── ✓ SIGNIFICATIVO            │
│                                                                │
│  MODELO 3 (sepal_length):                                      │
│  ├─ sepal_width ─────────────────── ✓ SIGNIFICATIVO            │
│  └─ petal_length ────────────────── ✓ SIGNIFICATIVO            │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 🔑 Hallazgos Principales

### 1. Correlación entre Pétalos
> Las dimensiones de los pétalos (largo y ancho) están **altamente correlacionadas**, lo que permite predicciones muy precisas entre ellas (R² > 0.94).

### 2. Importancia de la Especie
> La variable `species` es **estadísticamente significativa** para predecir características de los pétalos, reflejando las diferencias morfológicas entre las tres especies de Iris.

### 3. Menor Predictibilidad del Sépalo
> El `sepal_length` es **más difícil de predecir** (R² ~ 0.78), sugiriendo menor correlación con otras variables morfológicas.

### 4. Modelos Penalizados
> Los modelos **Ridge, Lasso y ElasticNet** no mejoran significativamente sobre OLS, indicando que no hay overfitting severo en un dataset tan compacto (150 muestras, 5 variables).

---

## 💻 Instalación

### Requisitos

```bash
Python >= 3.8
```

### Dependencias

```bash
pip install pandas numpy matplotlib scikit-learn statsmodels
```

### Estructura del Proyecto

```
📁 iris-regression-analysis/
├── 📁 Data/
│   └── IRIS.csv
├── 📓 Proyecto_Regresion_Iris.ipynb
└── 📄 README.md
```

### Ejecución

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/iris-regression-analysis.git

# Navegar al directorio
cd iris-regression-analysis

# Abrir Jupyter Notebook
jupyter notebook Proyecto_Regresion_Iris.ipynb
```

---

## 📚 Referencias

- Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. *Annals of Eugenics*, 7(2), 179-188. https://doi.org/10.1111/j.1469-1809.1936.tb02137.x

- Dua, D., & Graff, C. (2019). UCI Machine Learning Repository. University of California, Irvine, School of Information and Computer Sciences. http://archive.ics.uci.edu/ml

- Kaggle. (s.f.). Iris Flower Dataset. Recuperado de https://www.kaggle.com/datasets/arshid/iris-flower-dataset

- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

- Seabold, S., & Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with Python. *Proceedings of the 9th Python in Science Conference*, 57-61.

---

<div align="center">


[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?style=flat&logo=github)](https://github.com/tu-usuario)

</div>