import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="¿Lo comprarán?",
    page_icon="🛒",
    layout="wide"
)
# Menú lateral para navegar por la presentación
st.sidebar.title("🛒 ¿Lo comprarán?")
st.sidebar.markdown("### Presentación del proyecto")

seccion = st.sidebar.radio(
    "Selecciona un apartado:",
    [
        "🏠 Inicio",
        "📊 Análisis exploratorio",
        "⚙️ Preprocesamiento",
        "⚖️ Balanceo de clases",
        "🤖 Comparación de modelos",
        "🔍 Optimización",
        "🛒 Predicción interactiva",
        "🎯 Conclusiones"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Proyecto Machine Learning · Beatriz Fernández")

# SECCIÓN 1: INICIO
if seccion == "🏠 Inicio":

    st.title("🛒 ¿Lo comprarán?")
    st.subheader("Predicción de compras online mediante Machine Learning")

    st.write(
        "Este proyecto analiza el comportamiento de los usuarios "
        "de una tienda online para predecir si una sesión terminará en compra."
    )

    st.divider()

    st.header("🎯 Objetivo del proyecto")

    st.write(
        "Desarrollar un modelo de clasificación binaria que permita "
        "predecir la variable Revenue: True si el usuario compra "
        "y False si no compra."
    )

    st.header("📊 Dataset")

    col1, col2, col3 = st.columns(3)

    col1.metric("Sesiones analizadas", "12.330")
    col2.metric("Sesiones con compra", "1.908")
    col3.metric("Porcentaje de compras", "15,47 %")

    st.info(
        "El dataset está desequilibrado: aproximadamente el 84,53 % "
        "de las sesiones no terminan en compra."
    )

elif seccion == "📊 Análisis exploratorio":
    if seccion == "📊 Análisis exploratorio":

        st.title("📊 Análisis exploratorio de datos")
        st.write(
            "Antes de entrenar los modelos, exploramos el dataset "
            "para comprender el comportamiento de los usuarios."
        )

        import pandas as pd

        @st.cache_data
        def cargar_datos():
            return pd.read_csv("online_shoppers_intention.csv")

        df = cargar_datos()

        st.subheader("1. Distribución de compras")

        conteo = df["Revenue"].value_counts().reindex(
            [False, True], fill_value=0
        )

        datos_grafico = pd.DataFrame({
            "Resultado": ["No compra", "Compra"],
            "Sesiones": conteo.values
        }).set_index("Resultado")

        st.bar_chart(datos_grafico)

        st.warning(
            "El 84,53 % de las sesiones no terminan en compra. "
            "Por eso no podemos evaluar los modelos únicamente con accuracy."
        )

        st.subheader("2. Exploración interactiva del dataset")

        tipo_sesion = st.selectbox(
            "¿Qué sesiones quieres explorar?",
            ["Todas", "Con compra", "Sin compra"]
        )

        if tipo_sesion == "Con compra":
            datos_filtrados = df[df["Revenue"] == True]
        elif tipo_sesion == "Sin compra":
            datos_filtrados = df[df["Revenue"] == False]
        else:
            datos_filtrados = df

        st.metric("Sesiones seleccionadas", len(datos_filtrados))

        st.dataframe(datos_filtrados.head(20), use_container_width=True)

        st.subheader("3. ¿Por qué es importante este análisis?")

        st.markdown("""
        - Identificamos un **desequilibrio de clases** en la variable `Revenue`.
        - Comprobamos que necesitamos prestar atención a las compras reales.
        - Decidimos utilizar **precision, recall y F1** para evaluar los modelos.
        - Posteriormente, probamos distintas técnicas de balanceo.
        """)

if seccion == "⚙️ Preprocesamiento":

    st.title("⚙️ Preprocesamiento de los datos")

    st.write(
        "Antes de entrenar los modelos, preparamos las variables "
        "para que los algoritmos puedan utilizarlas correctamente."
    )

    st.subheader("1. Limpieza y revisión")

    st.markdown("""
    - **Valores nulos:** el dataset no presenta valores ausentes.
    - **Duplicados:** encontramos 125 filas duplicadas, que conservamos
      porque podrían corresponder a sesiones diferentes.
    - **Variable objetivo:** `Revenue`, que indica si hubo compra.
    - **PageValues:** se excluyó de los modelos para evitar depender
      de una variable potencialmente problemática para la predicción.
    """)

    st.subheader("2. Transformación de variables")

    col1, col2 = st.columns(2)

    with col1:
        st.info("**StandardScaler**")
        st.write(
            "Estandarizamos las variables numéricas para que "
            "sus diferentes escalas no condicionen a los modelos "
            "sensibles a las distancias."
        )

    with col2:
        st.info("**OneHotEncoder**")
        st.write(
            "Convertimos las variables categóricas en columnas "
            "numéricas, sin imponer un orden artificial entre categorías."
        )

    st.subheader("3. División de los datos")

    col1, col2 = st.columns(2)

    col1.metric("Entrenamiento", "80 % · 9.864 sesiones")
    col2.metric("Prueba", "20 % · 2.466 sesiones")

    st.write(
        "Utilizamos una división estratificada para conservar "
        "aproximadamente la misma proporción de compras y no compras "
        "en ambos conjuntos."
    )

    st.subheader("4. Selección de características")

    st.markdown("""
    Estudiamos las correlaciones entre variables y probamos
    diferentes configuraciones:

    - Eliminación de `BounceRates`, por su alta correlación con `ExitRates`.
    - Creación de `TiempoMedioPorProducto` mediante feature engineering.

    Ninguno de estos experimentos mejoró el F1 medio de la
    Regresión Logística utilizada como referencia.

    **Decisión final:** conservar las 16 variables predictoras
    originales seleccionadas, excluyendo `PageValues`.
    """)

    st.success(
        "El preprocesamiento se integra dentro de los pipelines "
        "para evitar fugas de información durante la validación cruzada."
    )

if seccion == "⚖️ Balanceo de clases":

    st.title("⚖️ Balanceo de clases")

    st.write(
        "La variable Revenue presenta un desequilibrio importante: "
        "el 84,53 % de las sesiones no terminan en compra y "
        "solo el 15,47 % sí lo hacen."
    )

    st.subheader("1. ¿Por qué necesitamos estudiar el balanceo?")

    st.markdown("""
    Si un modelo predijera siempre **«no compra»**, obtendría
    aproximadamente un **84,53 % de accuracy**, pero no
    detectaría ninguna compra real.

    Por eso, nuestro objetivo no es maximizar únicamente
    la exactitud, sino encontrar un equilibrio entre
    **precision y recall**, utilizando F1 como métrica principal.
    """)

    st.subheader("2. Técnicas que hemos experimentado")

    st.markdown("""
    - **Sin balanceo:** entrenamos con la distribución original.
    - **Random Oversampling:** duplicamos ejemplos de la clase minoritaria.
    - **Random Undersampling:** reducimos ejemplos de la clase mayoritaria.
    - **SMOTE:** generamos ejemplos sintéticos de la clase minoritaria.
    """)

    import pandas as pd

    resultados_balanceo = pd.DataFrame({
        "Técnica": [
            "Sin balanceo",
            "Oversampling",
            "Undersampling",
            "SMOTE"
        ],
        "F1": [0.2055, 0.3346, 0.3452, 0.3388]
    })

    st.subheader("3. Comparación experimental con KNN")

    st.bar_chart(
        resultados_balanceo.set_index("Técnica")["F1"]
    )

    st.dataframe(
        resultados_balanceo.sort_values("F1", ascending=False),
        hide_index=True,
        use_container_width=True
    )

    st.subheader("4. ¿Por qué seleccionamos undersampling?")

    st.success(
        "En nuestro experimento con KNN, Random Undersampling "
        "obtuvo el F1 medio más alto (0,3452). "
        "Por eso lo seleccionamos como estrategia común "
        "para comparar posteriormente los siete modelos."
    )

    st.markdown("""
    **Ventajas:**
    - Es una técnica sencilla y rápida.
    - Reduce el predominio de la clase mayoritaria.
    - Permite comparar los modelos utilizando el mismo procedimiento.

    **Limitaciones:**
    - Descarta observaciones de la clase mayoritaria.
    - Puede perder información útil.
    - Que funcione mejor con KNN no garantiza que sea
      la mejor técnica para todos los algoritmos.
    """)

    st.info(
        "Para evitar fugas de información, el balanceo se aplica "
        "únicamente a los datos de entrenamiento dentro del pipeline "
        "y de cada partición de validación cruzada."
    )

if seccion == "🤖 Comparación de modelos":

    st.title("🤖 Comparación de modelos de clasificación")

    st.write(
        "Se compararon siete algoritmos utilizando el mismo "
        "preprocesamiento, Random Undersampling y validación "
        "cruzada estratificada de cinco particiones."
    )

    import pandas as pd

    resultados_modelos = pd.DataFrame({
        "Modelo": [
            "Gradient Boosting",
            "Bagging",
            "Random Forest",
            "Pasting",
            "Regresión Logística",
            "AdaBoost",
            "KNN"
        ],
        "Accuracy": [
            0.6599, 0.6560, 0.6672, 0.6552,
            0.6393, 0.5533, 0.6051
        ],
        "Precision": [
            0.2808, 0.2771, 0.2798, 0.2750,
            0.2651, 0.2350, 0.2322
        ],
        "Recall": [
            0.7667, 0.7595, 0.7320, 0.7503,
            0.7523, 0.8362, 0.6730
        ],
        "F1": [
            0.4109, 0.4060, 0.4049, 0.4025,
            0.3920, 0.3668, 0.3452
        ]
    })

    st.subheader("1. Cuadro resumen de los modelos")

    tabla_resumen = resultados_modelos.copy()

    tabla_resumen.insert(
        1, "Balanceo", "Undersampling"
    )

    parametros = {
        "Gradient Boosting": "100 árboles, learning_rate=0.1, max_depth=3",
        "Bagging": "100 árboles, max_depth=10",
        "Random Forest": "n_estimators=100",
        "Pasting": "100 árboles, max_depth=10",
        "Regresión Logística": "C=0.1",
        "AdaBoost": "100 árboles, learning_rate=0.1, max_depth=1",
        "KNN": "n_neighbors=5"
    }

    tabla_resumen.insert(
        2,
        "Parámetros",
        tabla_resumen["Modelo"].map(parametros)
    )

    st.dataframe(
        tabla_resumen.style.format({
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1": "{:.4f}"
        }),
        hide_index=True,
        use_container_width=True
    )
    st.subheader("2. Comparación visual del F1")

    st.bar_chart(
        resultados_modelos.set_index("Modelo")["F1"]
    )

    st.subheader("3. ¿Por qué utilizamos F1?")

    st.markdown("""
    Nuestro dataset está desequilibrado, por lo que la
    accuracy puede resultar engañosa.

    Utilizamos **F1 como métrica principal** porque combina:

    - **Precision:** de las sesiones que predecimos como compra,
      cuántas realmente compran.
    - **Recall:** de las sesiones que realmente terminan en compra,
      cuántas conseguimos identificar.

    F1 nos permite valorar conjuntamente ambas métricas.
    """)

    st.subheader("4. ¿Qué modelo seleccionamos?")

    col1, col2, col3 = st.columns(3)

    col1.metric("Mejor F1", "0,4109")
    col2.metric("Recall", "76,67 %")
    col3.metric("Precision", "28,08 %")

    st.success(
        "Gradient Boosting obtuvo el mayor F1 medio "
        "en nuestra comparación inicial de siete modelos."
    )

    st.markdown("""
    **¿Por qué Gradient Boosting?**

    - Obtuvo el F1 más alto de los modelos comparados.
    - Identificó aproximadamente el 77 % de las compras reales
      durante la validación cruzada.
    - Superó ligeramente a Bagging y Random Forest en F1.

    **Limitación importante:** su precision es relativamente baja,
    lo que significa que genera bastantes falsos positivos.

    Por tanto, se seleccionó como candidato para optimizar sus
    hiperparámetros, no como un modelo perfecto.
    """)

    st.info(
        "Los resultados mostrados corresponden a las medias "
        "de validación cruzada sobre los datos de entrenamiento, "
        "no a la evaluación final sobre el conjunto de prueba."
    )

if seccion == "🔍 Optimización":

    st.title("🔍 Optimización de hiperparámetros")

    st.write(
        "Una vez seleccionado Gradient Boosting como mejor candidato, "
        "exploramos distintas configuraciones para intentar mejorar "
        "su rendimiento."
    )

    st.subheader("1. ¿Qué es la optimización de hiperparámetros?")

    st.markdown("""
    Los hiperparámetros son ajustes que establecemos antes de
    entrenar un modelo.

    En Gradient Boosting estudiamos:

    - **n_estimators:** número de árboles.
    - **learning_rate:** cuánto contribuye cada árbol.
    - **max_depth:** profundidad máxima de los árboles.
    - **subsample:** proporción de observaciones utilizadas
      para entrenar cada árbol.
    """)

    st.subheader("2. Grid Search frente a Random Search")

    col1, col2 = st.columns(2)

    with col1:
        st.info("**Grid Search**")
        st.write(
            "Prueba todas las combinaciones de una cuadrícula "
            "de hiperparámetros previamente definida."
        )

    with col2:
        st.info("**Random Search**")
        st.write(
            "Prueba un número determinado de combinaciones "
            "seleccionadas aleatoriamente."
        )

    st.caption(
        "En este proyecto utilizamos GridSearchCV para Regresión "
        "Logística y RandomizedSearchCV para Gradient Boosting."
    )

    st.subheader("3. Resultado de RandomizedSearchCV")

    st.markdown("""
    Se probaron **15 combinaciones** de Gradient Boosting mediante
    validación cruzada estratificada de cinco particiones,
    realizando **75 ajustes**.
    """)

    parametros_optimos = pd.DataFrame({
        "Hiperparámetro": [
            "n_estimators",
            "learning_rate",
            "max_depth",
            "subsample"
        ],
        "Valor seleccionado": [
            "138", "0,134", "2", "0,7874"
        ]
    })

    st.dataframe(
        parametros_optimos,
        hide_index=True,
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    col1.metric("F1 CV inicial", "0,4109")
    col2.metric(
        "Mejor F1 CV en Random Search",
        "0,4136",
        delta="+0,0027"
    )

    st.write(
        "La mejora en validación cruzada fue pequeña, por lo que "
        "también comparamos ambas configuraciones sobre el "
        "mismo conjunto de prueba."
    )

    st.subheader("4. Comparación sobre el conjunto de prueba")

    resultados_optimizacion = pd.DataFrame({
        "Modelo": [
            "Gradient Boosting original",
            "Gradient Boosting optimizado"
        ],
        "Accuracy": [0.6419, 0.6468],
        "Precision": [0.2657, 0.2678],
        "Recall": [0.7435, 0.7382],
        "F1": [0.3915, 0.3930]
    })

    st.dataframe(
        resultados_optimizacion,
        hide_index=True,
        use_container_width=True
    )

    st.bar_chart(
        resultados_optimizacion.set_index("Modelo")["F1"]
    )

    st.subheader("5. Decisión final")

    st.success(
        "Seleccionamos Gradient Boosting optimizado como "
        "modelo final para nuestra aplicación Streamlit."
    )

    st.markdown("""
    **¿Por qué?**

    - Su F1 en prueba fue ligeramente superior: **0,3930 frente a 0,3915**.
    - La precision y la accuracy también mejoraron ligeramente.
    - El recall disminuyó un poco.

    **Limitación:** las diferencias son muy pequeñas y no permiten
    afirmar que el modelo optimizado sea claramente superior.
    """)

    st.warning(
        "El modelo identifica aproximadamente el 74 % de las "
        "compras reales, pero su precision es del 26,78 %. "
        "Por tanto, genera bastantes falsos positivos y "
        "necesitaría mejoras antes de utilizarse en producción."
    )

if seccion == "🛒 Predicción interactiva":

    st.title("🛒 Predicción interactiva")

    st.write(
        "En esta sección utilizaremos el modelo Gradient Boosting "
        "optimizado para predecir si una sesión de navegación "
        "terminará en compra."
    )

    import joblib

    @st.cache_resource
    def cargar_modelo():
        return joblib.load("modelo_compra.pkl")

    try:
        modelo = cargar_modelo()

        st.success("✅ Modelo de Machine Learning cargado correctamente")

        st.info(
            "El modelo incluye el preprocesamiento de variables "
            "y Gradient Boosting. A continuación añadiremos "
            "un formulario para introducir los datos de una sesión."
        )

    except FileNotFoundError:
        st.error(
            "No se encuentra modelo_compra.pkl. "
            "Comprueba que está guardado en la carpeta del proyecto."
        )

    st.divider()
    st.subheader("🔮 Simulador de compra online")

    st.write(
        "Modifica las características de una sesión y pulsa "
        "el botón para obtener una predicción del modelo."
    )

    with st.form("formulario_prediccion"):

        st.markdown("#### 🛍️ Comportamiento de navegación")

        col1, col2 = st.columns(2)

        with col1:
            administrative = st.number_input("Páginas administrativas", min_value=0, value=0)
            informational = st.number_input("Páginas informativas", min_value=0, value=0)
            product_related = st.number_input("Páginas de productos", min_value=0, value=10)
            bounce_rates = st.slider("Tasa de rebote", 0.0, 1.0, 0.02)

        with col2:
            administrative_duration = st.number_input("Tiempo administrativo (segundos)", min_value=0.0, value=0.0)
            informational_duration = st.number_input("Tiempo informativo (segundos)", min_value=0.0, value=0.0)
            product_related_duration = st.number_input("Tiempo en productos (segundos)", min_value=0.0, value=300.0)
            exit_rates = st.slider("Tasa de salida", 0.0, 1.0, 0.05)

        st.markdown("#### 🌍 Contexto de la sesión")

        col3, col4 = st.columns(2)

        with col3:
            month = st.selectbox(
                "Mes", ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            )
            visitor_type = st.selectbox(
                "Tipo de visitante", ["Returning_Visitor", "New_Visitor", "Other"]
            )
            weekend = st.checkbox("¿Es fin de semana?")
            special_day = st.slider("Proximidad a fecha especial", 0.0, 1.0, 0.0, step=0.2)

        with col4:
            operating_systems = st.number_input("Sistema operativo (código)", min_value=1, max_value=8, value=2)
            browser = st.number_input("Navegador (código)", min_value=1, max_value=13, value=2)
            region = st.number_input("Región (código)", min_value=1, max_value=9, value=1)
            traffic_type = st.number_input("Tipo de tráfico (código)", min_value=1, max_value=20, value=2)

        predecir = st.form_submit_button(
            "🔮 Predecir si comprará",
            use_container_width=True
        )

    if predecir:

        datos_sesion = pd.DataFrame([{
            "Administrative": administrative,
            "Administrative_Duration": administrative_duration,
            "Informational": informational,
            "Informational_Duration": informational_duration,
            "ProductRelated": product_related,
            "ProductRelated_Duration": product_related_duration,
            "BounceRates": bounce_rates,
            "ExitRates": exit_rates,
            "SpecialDay": special_day,
            "Month": month,
            "OperatingSystems": operating_systems,
            "Browser": browser,
            "Region": region,
            "TrafficType": traffic_type,
            "VisitorType": visitor_type,
            "Weekend": weekend
        }])

        prediccion = modelo.predict(datos_sesion)[0]

        st.subheader("📋 Resultado de la predicción")

        if prediccion:
            st.success("🛒 El modelo predice que esta sesión terminará en COMPRA.")
        else:
            st.warning("🚫 El modelo predice que esta sesión NO terminará en compra.")

        st.caption(
            "Esta predicción es orientativa. El modelo presenta falsos positivos "
            "y no debe interpretarse como una certeza."
        )

if seccion == "🎯 Conclusiones":

    st.title("🎯 Conclusiones del proyecto")

    st.subheader("1. ¿Qué hemos conseguido?")

    st.markdown("""
    Hemos desarrollado un proyecto de Machine Learning capaz de
    clasificar sesiones de navegación según si terminan o no en compra.

    Durante el proyecto:

    - Analizamos **12.330 sesiones** de una tienda online.
    - Identificamos un importante **desequilibrio de clases**.
    - Experimentamos con distintas técnicas de balanceo.
    - Comparamos **siete algoritmos de clasificación**.
    - Aplicamos **Grid Search y Random Search** para optimizar modelos.
    - Desarrollamos una **aplicación interactiva con Streamlit**.
    """)

    st.subheader("2. Resultados del modelo final")

    col1, col2, col3 = st.columns(3)

    col1.metric("F1 en prueba", "0,3930")
    col2.metric("Recall en prueba", "73,82 %")
    col3.metric("Precision en prueba", "26,78 %")

    st.write(
        "El modelo final es Gradient Boosting optimizado "
        "mediante RandomizedSearchCV."
    )

    st.subheader("3. Limitaciones")

    st.markdown("""
    - La precision es relativamente baja, por lo que existen
      numerosos falsos positivos.
    - El dataset está desequilibrado.
    - Random Undersampling descarta parte de las observaciones
      de la clase mayoritaria.
    - La mejora obtenida mediante optimización fue pequeña.
    - Los resultados no justifican todavía su uso en producción.
    """)

    st.subheader("4. Posibles mejoras futuras")

    st.markdown("""
    - Comparar técnicas de balanceo específicas para cada modelo.
    - Ajustar el umbral de decisión según las necesidades del negocio.
    - Explorar otras variables y nuevas técnicas de feature engineering.
    - Evaluar la calibración de probabilidades.
    - Validar el modelo con nuevos datos de navegación.
    """)

    st.subheader("5. Aplicación práctica")

    st.info(
        "Un modelo de este tipo podría ayudar a identificar "
        "sesiones con posible intención de compra y orientar "
        "acciones de marketing, siempre que su rendimiento "
        "se valide y mejore antes de utilizarlo realmente."
    )

    st.success(
        "Conclusión: hemos construido un flujo completo de "
        "Machine Learning, desde la exploración de datos "
        "hasta el despliegue de un modelo predictivo en Streamlit."
    )