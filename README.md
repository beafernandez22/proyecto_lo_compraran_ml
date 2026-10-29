# 🛒 ¿Lo comprarán?
### Predicción de compras online mediante Machine Learning

**Beatriz Fenández | Ironhack Data Analytics**

## 🎯 Objetivo y contexto de negocio

Este proyecto busca predecir si una sesión de navegación en una tienda online terminará en compra mediante técnicas de Machine Learning.

El objetivo es ayudar a una empresa de comercio electrónico a comprender el comportamiento de sus visitantes y explorar cómo identificar posibles compradores para orientar futuras estrategias de marketing.

## 📊 Dataset

Se utiliza el [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset), disponible en UCI Machine Learning Repository.

- **12.330 sesiones** de navegación.
- **18 variables**, relacionadas con navegación, duración de visitas, tipo de usuario y contexto.
- **Variable objetivo:** `Revenue` (compra / no compra).
- **15,47 %** de sesiones con compra frente a **84,53 %** sin compra.

**Calidad del dato:** no se encontraron valores nulos. Se identificaron 125 registros duplicados, que se conservaron porque podrían corresponder a sesiones distintas.

## ❓ Preguntas clave

- ¿Podemos predecir si un visitante terminará comprando?
- ¿Cómo afecta el desequilibrio de clases al rendimiento?
- ¿Qué algoritmo ofrece el mejor equilibrio entre precision y recall?
- ¿Cómo podríamos utilizar estas predicciones en un contexto empresarial?

## ⚙️ Metodología

1. **EDA:** análisis de la distribución de compras, comportamiento de navegación y correlaciones.
2. **Preprocesamiento:** división estratificada 80/20, StandardScaler y OneHotEncoder.
3. **Selección de variables:** análisis de correlaciones y experimentos de feature engineering.
4. **Balanceo:** comparación de Oversampling, Undersampling y SMOTE. Se seleccionó Undersampling como estrategia común para comparar modelos.
5. **Modelado:** comparación de siete algoritmos mediante validación cruzada estratificada.
6. **Optimización:** GridSearchCV y RandomizedSearchCV, utilizando F1 como métrica principal.
7. **Despliegue:** aplicación interactiva desarrollada con Streamlit.

## 📈 Resultados principales

**Gradient Boosting** obtuvo el mayor F1 medio en la comparación inicial de siete modelos (**0,4109**).

Tras optimizar sus hiperparámetros, se seleccionó como modelo final.

| Métrica | Resultado en prueba |
|---|---:|
| Accuracy | 0,6468 |
| Precision | 0,2678 |
| Recall | 0,7382 |
| F1 | **0,3930** |

**Insight principal:** el modelo identifica aproximadamente el 74 % de las compras reales, pero genera numerosos falsos positivos. Por tanto, todavía no es adecuado para tomar decisiones comerciales automáticas.

## 💡 Recomendaciones de negocio

- Explorar el modelo como herramienta experimental para identificar sesiones con posible intención de compra.
- Ajustar el umbral de decisión para reducir falsos positivos según los costes de marketing.
- Realizar pruebas A/B antes de utilizar las predicciones en campañas reales.

## ⚠️ Limitaciones y próximos pasos

El dataset está desequilibrado, el undersampling descarta información y la precisión del modelo sigue siendo baja.

Como mejoras futuras se propone comparar técnicas de balanceo por algoritmo, evaluar la curva precision-recall, calibrar probabilidades y validar el modelo con nuevos datos.

## 🖥️ Cómo ejecutar el proyecto

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

Iniciar la aplicación:

```bash
python -m streamlit run app.py
```

## 📁 Estructura del proyecto

proyecto_lo_compraran_ml/
├── data/
│   └── online_shoppers_intention.csv
├── notebooks/
│   └── proyecto_lo_compraran_ml.ipynb
├── scripts/
│   └── funciones.py
├── outputs/
│   └── modelo_compra.pkl
├── app.py
├── requirements.txt
├── README.md
└── .gitignore