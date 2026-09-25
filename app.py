import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configuración inicial de la página
st.set_page_config(
    page_title="Clasificación de Tumores",
    page_icon="🔬",
    layout="wide"
)


CSV_FILE = "data_cancer.csv"
MODEL_FILE = "modelo_cancer.pkl"
SCALER_FILE = "scaler_cancer.pkl"

# Funciones de carga y entrenamiento del modelo
@st.cache_data
def cargar_datos():
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        return df
    else:
        st.error(f"No se encontró el archivo '{CSV_FILE}'. Asegúrate de que esté en el mismo directorio.")
        return None

def cargar_o_entrenar_modelo(df):
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        scaler = joblib.load(SCALER_FILE)
        modelo = joblib.load(MODEL_FILE)
    else:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        modelo = LogisticRegression(max_iter=1000, random_state=42)
        modelo.fit(X_train_scaled, y_train)
        
        joblib.dump(modelo, MODEL_FILE)
        joblib.dump(scaler, SCALER_FILE)

    X_test_scaled = scaler.transform(X_test)
    y_pred = modelo.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    return modelo, scaler, X, y, X_test, y_test, y_pred, acc

df = cargar_datos()

if df is not None:
    modelo, scaler, X, y, X_test, y_test, y_pred, acc = cargar_o_entrenar_modelo(df)

    # BARRA LATERAL (Navegación)
    st.sidebar.image("https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%2Fid%2FOIP.N6eUNMsWZJgzwYz22UlQyAHaEK%3Fpid%3DApi&f=1&ipt=bc5bacdc963f0897ff89e5b651c2de519d997ba09a6532d1fa42ad3766ff2165&ipo=images?auto=format&fit=crop&w=400&q=80",
    caption="Clasificación de tumor", use_container_width=True,)
    
    st.sidebar.title("📌 Menú de Navegación")
    opcion = st.sidebar.radio(
        "Seleccione una sección:",
        [
            "📖 Presentación (Markdown)",
            "📊 Resumen Estadístico",
            "🔮 Predicción Interactiva",
            "📜 Script General (Código)"
        ]
    )

    # 1. SECCIÓN DE PRESENTACIÓN (MARKDOWN)
    if opcion == "📖 Presentación (Markdown)":
        st.title("🔬 Clasificación de Tumores: Benigno o Maligno")
        st.markdown("""
        ## **PRESENTACIÓN:**

        El presente modelo aprende a clasificar tumores como **malignos** o **benignos** usando el dataset *Breast Cancer Wisconsin* a partir del archivo `data_cancer.csv`. Este es uno de los datasets más clásicos en Machine Learning y representa el tipo de problemas que los sistemas de Inteligencia Artificial ayudan a resolver en el ámbito médico e intrahospitalario.

        Autor: Rafael Ayuque Anccasi (rafsys@gmail.com)
        ---

        ### 📋 **Características del Dataset y Proyecto:**
        * **Pacientes registrados:** 569
        * **Características médicas por muestra:** 30 (Radio medio, textura, perímetro, área, suavidad, concavidad, etc.)
        * **Variable objetivo (`target`):**
            * `0`: Maligno (Atención prioritaria / Peligroso)
            * `1`: Benigno (No peligroso)
        
        ### 🤖 **Modelo Utilizado:**
        * **Algoritmo:** Regresión Logística (`LogisticRegression`)
        * **Preprocesamiento:** Escalado estándar (`StandardScaler`)
        * **Precisión del Modelo en Test:** `{:.2f}%`
        """.format(acc * 100))

    # 2. SECCIÓN DE RESUMEN ESTADÍSTICO
    elif opcion == "📊 Resumen Estadístico":
        st.title("📊 Resumen Estadístico del Dataframe")
        
        st.subheader("Vista Previa del Dataset")
        st.dataframe(df.head(10), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total de Muestras", df.shape[0])
            st.metric("Total de Columnas", df.shape[1])
        with col2:
            conteo_target = df["target"].value_counts()
            st.metric("Casos Benignos (1)", conteo_target.get(1, 0))
            st.metric("Casos Malignos (0)", conteo_target.get(0, 0))

        st.subheader("Estadísticas Descriptivas (Min, Max, Media, Desviación)")
        st.dataframe(df.describe().T, use_container_width=True)

        st.subheader("Matriz de Confusión del Modelo")
        fig, ax = plt.subplots(figsize=(6, 4))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Maligno (0)', 'Benigno (1)'],
                    yticklabels=['Maligno (0)', 'Benigno (1)'], ax=ax)
        plt.title('Matriz de Confusión')
        plt.ylabel('Valor Real')
        plt.xlabel('Predicción')
        st.pyplot(fig)

    # 3. SECCIÓN DE PREDICCIÓN INTERACTIVA
    elif opcion == "🔮 Predicción Interactiva":
        st.title("🔮 Formulario de Predicción de Tumores")
        st.info("Ingrese los valores de las características dentro de los límites permitidos (Min - Max) basados en el archivo CSV original.")

        # Gestión de estado de variables
        if "form_values" not in st.session_state:
            st.session_state.form_values = {col: float(X[col].mean()) for col in X.columns}

        # Botón para reiniciar valores de prueba
        if st.button("🔄 Reiniciar Valores de Prueba"):
            for col in X.columns:
                st.session_state.form_values[col] = float(X[col].mean())
            st.rerun()

        # Agrupación de características en pestañas para mejor experiencia visual
        tabs = st.tabs(["Valores Promedio (Mean)", "Errores Estándar (SE)", "Peores Valores (Worst)"])

        user_input = {}
        cols_mean = [c for c in X.columns if c.startswith("mean")]
        cols_error = [c for c in X.columns if "error" in c]
        cols_worst = [c for c in X.columns if c.startswith("worst")]

        with tabs[0]:
            st.subheader("Características Medias")
            for col_name in cols_mean:
                min_val = float(X[col_name].min())
                max_val = float(X[col_name].max())
                default_val = float(st.session_state.form_values[col_name])
                user_input[col_name] = st.number_input(
                    label=f"{col_name} (Permitido: {min_val:.4f} - {max_val:.4f})",
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    key=f"input_{col_name}"
                )

        with tabs[1]:
            st.subheader("Errores de Medición")
            for col_name in cols_error:
                min_val = float(X[col_name].min())
                max_val = float(X[col_name].max())
                default_val = float(st.session_state.form_values[col_name])
                user_input[col_name] = st.number_input(
                    label=f"{col_name} (Permitido: {min_val:.4f} - {max_val:.4f})",
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    key=f"input_{col_name}"
                )

        with tabs[2]:
            st.subheader("Peores Valores Medidos")
            for col_name in cols_worst:
                min_val = float(X[col_name].min())
                max_val = float(X[col_name].max())
                default_val = float(st.session_state.form_values[col_name])
                user_input[col_name] = st.number_input(
                    label=f"{col_name} (Permitido: {min_val:.4f} - {max_val:.4f})",
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    key=f"input_{col_name}"
                )

        st.markdown("---")
        if st.button("🚀 Realizar Predicción"):
            # Preparar dataframe en el mismo orden exacto de columnas
            input_df = pd.DataFrame([user_input])[X.columns]
            
            # Normalizar y predecir
            input_scaled = scaler.transform(input_df)
            prediction = modelo.predict(input_scaled)[0]
            probabilities = modelo.predict_proba(input_scaled)[0]

            st.subheader("📋 Resultado del Diagnóstico:")
            if prediction == 1:
                st.success("✅ **Predicción: BENIGNO** (No peligroso)")
                st.info(f"📊 **Nivel de Confianza:** {probabilities[1]*100:.2f}%")
            else:
                st.error("⚠️ **Predicción: MALIGNO** (Peligroso / Requiere atención médica)")
                st.info(f"📊 **Nivel de Confianza:** {probabilities[0]*100:.2f}%")

    # 4. SECCIÓN DE SCRIPT GENERAL
    elif opcion == "📜 Script General (Código)":
        st.title("📜 Script General de Entrenamiento")
        st.markdown("A continuación se muestra el código completo utilizado para procesar los datos, entrenar y evaluar el modelo basándose en el notebook `MalignoBenigno.ipynb`:")
        
        codigo_general = """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. CARGAR DATOS
cancer = pd.read_csv("data_cancer.csv")
X = cancer.drop(columns=["Unnamed: 0", "target"], errors="ignore")
y = cancer["target"]

# 2. DIVIDIR EN ENTRENAMIENTO Y PRUEBA
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3. NORMALIZAR
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 4. ENTRENAR EL MODELO
modelo = LogisticRegression(max_iter=1000, random_state=42)
modelo.fit(X_train_scaled, y_train)

# 5. EVALUAR
y_pred = modelo.predict(X_test_scaled)
print(f"Precisión: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=['Maligno','Benigno']))

# 6. GUARDAR MODELOS
joblib.dump(modelo, 'modelo_cancer.pkl')
joblib.dump(scaler, 'scaler_cancer.pkl')
        """
        st.code(codigo_general, language="python")
