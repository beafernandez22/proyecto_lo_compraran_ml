from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def crear_preprocesador(X):

    columnas_categoricas = [
        "Month",
        "VisitorType",
        "OperatingSystems",
        "Browser",
        "Region",
        "TrafficType"
    ]

    columnas_numericas = [
        columna for columna in X.columns
        if columna not in columnas_categoricas
    ]

    preprocesador = ColumnTransformer(
        transformers=[
            ("numericas", StandardScaler(), columnas_numericas),
            ("categoricas", OneHotEncoder(handle_unknown="ignore"),
             columnas_categoricas)
        ]
    )

    return preprocesador


from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def evaluar_modelo(modelo, X_test, y_test):
    predicciones = modelo.predict(X_test)

    print("Accuracy:", round(accuracy_score(y_test, predicciones), 4))

    print("\nInforme de clasificación:")
    print(classification_report(y_test, predicciones, zero_division=0))

    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, predicciones))


import numpy as np


def crear_caracteristicas(X):
    X = X.copy()

    X["TiempoMedioPorProducto"] = np.where(
        X["ProductRelated"] > 0,
        X["ProductRelated_Duration"] / X["ProductRelated"].replace(0, 1),
        0
    )

    return X