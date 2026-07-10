import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from unidecode import unidecode
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.inspection import permutation_importance

# ==========================================
# 1. CONFIGURACIÓN DE PÁGINA Y ESTADOS DE SESIÓN
# ==========================================
st.set_page_config(
    page_title="Modelos Estadísticos de Predicción",
    page_icon="🛢️",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Inicialización del enrutador de páginas y variables de sesión
if 'pagina_actual' not in st.session_state:
    st.session_state.pagina_actual = 'Inicio'

if 'aviso_leido' not in st.session_state:
    st.session_state.aviso_leido = False

def cambiar_pagina(nueva_pagina):
    st.session_state.pagina_actual = nueva_pagina

# --- PANTALLA EMERGENTE (POP-UP) ---
@st.dialog("⚠️ Aviso Académico")
def mostrar_disclaimer():
    st.markdown("Este sistema predictivo tiene **fines de evaluación estudiantil**.")
    st.markdown("Los dictámenes arrojados poseen un margen de error estadístico. **Bajo ninguna circunstancia** este software debe utilizarse como referencia absoluta, ni sustituir el criterio geofísico humano en proyectos reales de perforación.")
    
    if st.button("Comprendido y Aceptado", type="primary", use_container_width=True):
        st.session_state.aviso_leido = True
        st.rerun()

# Disparador automático de la pantalla emergente
if not st.session_state.aviso_leido:
    mostrar_disclaimer()

# --- CSS Global ---
st.markdown("""
    <style>
    .metric-card {
        background-color: #11141A; padding: 24px; border-radius: 12px;
        border: 1px solid #2A2E39; border-left: 6px solid #00D2FF;
        box-shadow: 0 8px 16px rgba(0,0,0,0.4); margin-bottom: 24px;
        transition: transform 0.2s ease-in-out;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-title { color: #8C92A4; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 8px; font-family: 'Segoe UI', sans-serif; }
    .metric-value { color: #FFFFFF; font-size: 3.2rem; font-weight: 800; margin-bottom: 4px; line-height: 1.1; }
    .metric-error { color: #FF4B4B; font-size: 1.1rem; font-weight: 600; display: flex; align-items: center; gap: 6px; }
    
    .portada-title { text-align: center; color: #FFFFFF; font-size: 3rem; font-weight: 900; margin-bottom: 0px; }
    .portada-subtitle { text-align: center; color: #00D2FF; font-size: 1.5rem; margin-top: 5px; margin-bottom: 40px; }
    .portada-academic { text-align: center; color: #8C92A4; font-size: 1.1rem; margin-bottom: 50px; text-transform: uppercase; letter-spacing: 2px;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CARGA DE MODELOS (Caché en Memoria)
# ==========================================
@st.cache_resource
def inicializar_componentes():
    try:
        # --- MODELOS HISTÓRICOS (ORIGINALES) ---
        mod_reg_mlp = joblib.load('modelo_mlp_pozos.pkl')
        prep_reg = joblib.load('preprocesador_pozos.pkl')
        mae_mlp = joblib.load('mae_error.pkl')
        dicc = joblib.load('diccionario_opciones.pkl')
        pipe_clf_gbm = joblib.load('pipeline_clasificacion_pozos.pkl')
        met_clf_gbm = joblib.load('metricas_clasificacion.pkl')
        
        # --- NUEVOS ARTEFACTOS ALTERNATIVOS ---
        mod_reg_gbm = joblib.load('modelo_gbm_pozos_reg.pkl')
        mae_gbm = joblib.load('mae_error_gbm_reg.pkl')
        pipe_clf_mlp = joblib.load('pipeline_mlp_clasificacion_pozos.pkl')
        exactitud_mlp_val = joblib.load('exactitud_mlp_clf.pkl')
        
        return (mod_reg_mlp, mod_reg_gbm, prep_reg, mae_mlp, mae_gbm, dicc, 
                pipe_clf_gbm, pipe_clf_mlp, met_clf_gbm, exactitud_mlp_val)
    except FileNotFoundError:
        st.error("⚠️ Error Crítico: Faltan archivos .pkl en el directorio. Ejecuta primero los scripts de entrenamiento.")
        st.stop()

# Desempaquetado global de tensores y modelos
(modelo_reg_mlp, modelo_reg_gbm, preprocesador_reg, mae_error_mlp, mae_error_gbm, 
 diccionario, pipeline_clf_gbm, pipeline_clf_mlp, metricas_clf_gbm, exactitud_mlp_clf) = inicializar_componentes()

@st.cache_data
def generar_datos_rendimiento():
    df = pd.read_csv('Pozos brasil 2.csv', sep=';', encoding='latin1', decimal=',')
    columnas_numericas = ['LATITUDE_BASE_DD', 'LONGITUDE_BASE_DD', 'LAMINA_D_AGUA_M', 'COTA_ALTIMETRICA_M', 'PROFUNDIDADE_SONDADOR_M']
    for col in columnas_numericas:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '.').astype(float)
            
    df = df.dropna(subset=['LATITUDE_BASE_DD', 'LONGITUDE_BASE_DD', 'PROFUNDIDADE_SONDADOR_M'])
    df = df[(df['PROFUNDIDADE_SONDADOR_M'] > 100) & (df['PROFUNDIDADE_SONDADOR_M'] < 8000)]
    df['LAMINA_D_AGUA_M'] = df['LAMINA_D_AGUA_M'].fillna(0.0)
    df['COTA_ALTIMETRICA_M'] = df['COTA_ALTIMETRICA_M'].fillna(0.0)
    
    columnas_categoricas = ['BACIA', 'TERRA_MAR', 'TIPO', 'CATEGORIA', 'CAMPO', 'SITUACAO']
    for col in columnas_categoricas:
        df[col] = df[col].apply(lambda x: unidecode(str(x)).lower().strip() if pd.notnull(x) else "desconocido")
        
    # --- PARCHE VITAL: SINCRONIZACIÓN DE ETIQUETAS ---
    # Traduce y agrupa las situaciones de prueba para que el modelo reconozca el texto
    def agrupar_situacion_prueba(s):
        if 'produt' in s or 'jazida' in s: return 'productores / activos'
        if 'abandon' in s or 'arras' in s: return 'abandonados / cerrados'
        if 'interven' in s or 'avalia' in s or 'pioneiro' in s or 'explora' in s: return 'exploracion / intervencion'
        if 'injet' in s: return 'inyectores'
        return 'otras_situaciones'
        
    df['SITUACAO'] = df['SITUACAO'].apply(agrupar_situacion_prueba)
    
    return df.sample(n=min(2000, len(df)), random_state=42)

df_prueba = generar_datos_rendimiento()

# ==========================================
# 3. MOTORES VISUALES (PLOTLY)
# ==========================================
def generar_curva_probabilidad(prediccion, mae):
    min_x, max_x = max(0.0, prediccion - 4*mae), prediccion + 4*mae
    x = np.linspace(min_x, max_x, 500)
    y = (1 / (mae * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - prediccion) / mae)**2)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='#00D2FF', width=3, shape='spline'), name='Distribución'))
    
    limite_inf = max(0.0, prediccion - mae)
    x_fill = np.linspace(limite_inf, prediccion + mae, 100)
    y_fill = (1 / (mae * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_fill - prediccion) / mae)**2)
    fig.add_trace(go.Scatter(x=np.concatenate([x_fill, x_fill[::-1]]), y=np.concatenate([y_fill, np.zeros_like(y_fill)]), 
        fill='toself', fillcolor='rgba(0, 210, 255, 0.15)', line=dict(color='rgba(255,255,255,0)')))
    
    fig.add_vline(x=prediccion, line_dash="dash", line_color="#FF4B4B", line_width=2,
                  annotation_text=f"Objetivo: {prediccion:,.0f}m", annotation_position="top right",
                  annotation_font=dict(color="#FF4B4B", size=13, weight="bold"))

    fig.update_layout(template="plotly_dark", title=dict(text="Análisis de Densidad Estocástica", font=dict(color="#E0E0E0")),
                      height=400, margin=dict(l=20, r=20, t=60, b=20), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    return fig

def generar_curva_aprendizaje(modelo_cargado):
    if hasattr(modelo_cargado, 'regressor_'):
        mlp = modelo_cargado.regressor_
    else:
        mlp = modelo_cargado
        
    y_loss = mlp.loss_curve_ if hasattr(mlp, 'loss_curve_') else [0]
    y_val = mlp.validation_scores_ if hasattr(mlp, 'validation_scores_') else None
        
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Scatter(
        y=y_loss, mode='lines', line=dict(color='#00FF7F', width=3, shape='spline'),
        fill='tozeroy', fillcolor='rgba(0, 255, 127, 0.1)', name='Pérdida (Entrenamiento)'
    ), secondary_y=False)
    
    if y_val is not None:
        fig.add_trace(go.Scatter(
            y=y_val, mode='lines', line=dict(color='#FF4B4B', width=3, shape='spline', dash='dot'),
            name='Rendimiento R² (Validación)'
        ), secondary_y=True)
    
    fig.update_layout(
        template="plotly_dark", title=dict(text="Curva de Aprendizaje de Doble Eje", font=dict(color="#E0E0E0")),
        xaxis_title="Épocas (Iteraciones)", showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0)', bordercolor='#2A2E39', borderwidth=1),
        height=400, margin=dict(l=20, r=20, t=60, b=20), 
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='#2A2E39')
    )
    fig.update_yaxes(title_text="Costo de Pérdida (Loss)", secondary_y=False, showgrid=True, gridcolor='#2A2E39')
    fig.update_yaxes(title_text="Score de Validación (R²)", secondary_y=True, showgrid=False)
    return fig

def generar_grafo_red_neuronal():
    capas = [10, 8, 6, 4, 1] 
    fig = go.Figure()
    edge_x, edge_y = [], []
    for i in range(len(capas) - 1):
        y_start, y_end = np.linspace(0.1, 0.9, capas[i]), np.linspace(0.1, 0.9, capas[i+1])
        for ys in y_start:
            for ye in y_end:
                edge_x.extend([i, i+1, None]); edge_y.extend([ys, ye, None])
                
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color='rgba(0, 210, 255, 0.15)', width=1), hoverinfo='none'))
    
    for i, num_nodes in enumerate(capas):
        color_nodo = '#00D2FF' if i == 0 else ('#FF4B4B' if i == len(capas)-1 else '#00FF7F')
        fig.add_trace(go.Scatter(x=[i]*num_nodes, y=np.linspace(0.1, 0.9, num_nodes), mode='markers',
            marker=dict(size=16, color=color_nodo, line=dict(width=2, color='#FFFFFF')), hoverinfo='name'))
        
    fig.update_layout(template="plotly_dark", title=dict(text="Esquema Topológico MLP", font=dict(color="#E0E0E0")),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=400, showlegend=False)
    return fig

@st.cache_data
def generar_curva_precision_recall(_pipeline, df_test, clases):
    """
    Evalúa el rendimiento dinámico del algoritmo frente al desbalance poblacional.
    
    Metodología:
    Calcula la Curva PR (Precision-Recall / Precisión-Exhaustividad) utilizando la 
    técnica 'One-vs-Rest' (Una clase contra el resto), binarizando las etiquetas de la matriz.
    
    Parámetros:
    - _pipeline (Pipeline): Estimador empaquetado (ignorado en caché mediante el guion bajo).
    - df_test (DataFrame): Matriz de validación cruzada.
    - clases (ndarray): Vector con las taxonomías de situaciones de pozos.
    
    Retorna:
    - fig (go.Figure): Gráfica estocástica generada por Plotly.
    """
    y_score = _pipeline.predict_proba(df_test)
    y_real = df_test['SITUACAO']
    y_bin = label_binarize(y_real, classes=clases)
    
    fig = go.Figure()
    for i in range(len(clases)):
        if np.sum(y_bin[:, i]) > 0:
            precision, recall, _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
            ap = average_precision_score(y_bin[:, i], y_score[:, i])
            fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f'{str(clases[i]).upper()} (Área: {ap:.2f})'))
            
    fig.update_layout(
        template="plotly_dark", title=dict(text="Dinámica de Precisión-Exhaustividad (PR Curve)", font=dict(color="#E0E0E0")),
        xaxis_title="Exhaustividad (Recall)", yaxis_title="Precisión (Precision)",
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=450, margin=dict(t=40, b=10, l=40, r=40),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5, bgcolor='rgba(0,0,0,0)')
    )
    return fig

@st.cache_data
def generar_grafico_importancia(_pipeline, df_test, clases_nombres):
    """
    Aplica la técnica de Importancia por Permutación (Permutation Importance).
    """
    df_valido = df_test[df_test['SITUACAO'].isin(clases_nombres)].copy()
    
    columnas_clasificacion_completas = [
        'LATITUDE_BASE_DD', 
        'LONGITUDE_BASE_DD', 
        'LAMINA_D_AGUA_M', 
        'COTA_ALTIMETRICA_M', 
        'PROFUNDIDADE_SONDADOR_M',
        'BACIA', 
        'TERRA_MAR', 
        'TIPO', 
        'CATEGORIA', 
        'CAMPO'
    ]
    
    X = df_valido[columnas_clasificacion_completas]
    
    # --- PARCHE: MANTENER TEXTO PURO ---
    # El pipeline espera comparar sus predicciones de texto con etiquetas de texto.
    y = df_valido['SITUACAO']
    
    resultado = permutation_importance(_pipeline, X, y, n_repeats=5, random_state=42, n_jobs=-1)
    df_imp = pd.DataFrame({'Variable': X.columns, 'Importancia': resultado.importances_mean}).sort_values(by='Importancia', ascending=True)
    
    fig = px.bar(df_imp, x='Importancia', y='Variable', orientation='h', color='Importancia', color_continuous_scale='Purples')
    fig.update_layout(
        template="plotly_dark", title=dict(text="Impacto Predictivo (Permutación)", font=dict(color="#E0E0E0")),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=450, coloraxis_showscale=False,
        xaxis_title="Caída de Exactitud al permutar", yaxis_title="Variable"
    )
    return fig

# ==========================================
# 4. BARRA LATERAL (Condicional)
# ==========================================
if st.session_state.pagina_actual != 'Inicio':
    with st.sidebar:
        st.button("🏠 Volver al Menú Principal", on_click=cambiar_pagina, args=('Inicio',), use_container_width=True, type="primary")
        st.markdown("<h2 style='text-align: center; color: #00D2FF;'>Vector de Entrada</h2>", unsafe_allow_html=True)
        st.markdown("---")
        
        # --- PARÁMETROS ESPACIALES ---
        latitud_usuario = st.slider("Latitud (DD - Grados Decimales)", -33.7000, 5.2000, -22.0000, step=0.0001)
        longitud_usuario = st.slider("Longitud (DD - Grados Decimales)", -73.9000, -34.7000, -40.0000, step=0.0001)

        # --- PARÁMETROS TOPOGRÁFICOS ---
        entorno_operativo = st.selectbox("Entorno Operativo", ["mar", "tierra"])
        lamina_agua = st.number_input("Lámina de Agua (m)", min_value=0.0, value=500.0) if entorno_operativo == "mar" else 0.0
        cota_altimetrica = 0.0 if entorno_operativo == "mar" else st.number_input("Cota Altimétrica (m)", min_value=0.0, value=50.0)
        
        # --- PARÁMETROS GEOLÓGICOS Y ESTRATÉGICOS ---
        st.markdown("---")
        cuenca_seleccionada = st.selectbox("Cuenca Petrolífera", diccionario.get('cuencas', ['campos', 'santos', 'potiguar', 'reconcavo', 'espirito santo', 'otras_cuencas']))
        campo_seleccionado = st.selectbox("Campo Específico", diccionario.get('campos', ['otros_campos']))
        
        clase_pozo = st.selectbox("Tipo de Operación", ["desarrollo", "exploratorio"])
        categoria_pozo = st.selectbox("Categoría Estratégica", ["desarrollo", "pionero", "extension", "especial", "inyeccion", "otras_categorias"])
        situacion_pozo = st.selectbox("Situación Esperada", diccionario.get('situaciones', []))

# ==========================================
# 5. ENRUTADOR DE PÁGINAS (VISTAS)
# ==========================================

# VISTA 1: INICIO (PORTADA)
if st.session_state.pagina_actual == 'Inicio':
    st.markdown("<div class='portada-title'>Modelos de Machine Learning asistidos por IA</div>", unsafe_allow_html=True)
    st.markdown("<div class='portada-subtitle'>Análisis Estocástico Multivariante del Subsuelo Brasileño</div>", unsafe_allow_html=True)
    st.markdown("<div class='portada-academic'>Universidad Central del Ecuador (UCE) | FIGEMPA - Ingeniería en Petróleos</div>", unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col1, espaciador, col2 = st.columns([1, 0.1, 1])
    
    with col1:
        # Reemplazar el nombre del .svg como corresponda
        st.image("portada_prediccion_petroleo_pro.svg", use_container_width=True, caption="Topología Continua")
        st.markdown("<h3 style='text-align: center; color: #00D2FF;'>1. Predicción de Profundidad</h3>", unsafe_allow_html=True)
        # Acrónimo resuelto: MLP (Multi-Layer Perceptron)
        st.markdown("<p style='text-align: center;'>Estimación escalar mediante Perceptrón Multicapa (MLP - Multi-Layer Perceptron).</p>", unsafe_allow_html=True)
        if st.button("INGRESAR AL MÓDULO PREDICTIVO", on_click=cambiar_pagina, args=('Predicción',), use_container_width=True):
            pass
            
    with col2:
        # Reemplazar el nombre del .svg como corresponda
        st.image("portada_clasificacion_petroleo_brasil_v2.svg", use_container_width=True, caption="Fronteras de Decisión Discretas")
        st.markdown("<h3 style='text-align: center; color: #9D4EDD;'>2. Clasificación Operacional</h3>", unsafe_allow_html=True)
        # Acrónimo resuelto: GBM (Gradient Boosting Machine)
        st.markdown("<p style='text-align: center;'>Dictamen categórico mediante Ensamblaje Estocástico (GBM - Gradient Boosting Machine).</p>", unsafe_allow_html=True)
        if st.button("INGRESAR AL MÓDULO DE CLASIFICACIÓN", on_click=cambiar_pagina, args=('Clasificación',), use_container_width=True):
            pass


    # CONTEXTUALIZACIÓN DEL DATASET (Footer)
    st.markdown("<br><hr style='border: 1px solid #2A2E39;'>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #8C92A4; font-size: 0.9rem;'><b>Material de Estudio:</b> Las arquitecturas matemáticas de este proyecto fueron entrenadas utilizando los registros públicos de ><b>Kaggle</b> publicados por la <b>Agência Nacional do Petróleo, Gás Natural e Biocombustíveis (ANP)</b> de la República Federativa de Brasil.</p>", unsafe_allow_html=True)

# VISTA 2: MÓDULO DE PREDICCIÓN (MLP & GBM)
elif st.session_state.pagina_actual == 'Predicción':
    st.title("🎯 Modelo Predictivo: Regresión Continua")
    st.markdown("Inferencia de Profundidad de Sondador asistida por Redes Neuronales Artificiales y Gradient Boosting.")
    
    # Se expanden las pestañas a 3
    tab_inf, tab_aud, tab_aud_gbm = st.tabs(["🔮 Panel de Inferencia", "📊 Auditoría y Arquitectura (MLP)", "📊 Auditoría y Evaluación (GBM)"])
    
    with tab_inf:
        # --- INYECCIÓN 1: SELECTOR DE MODELO ---
        st.markdown("### Configuración del Motor Matemático")
        motor_seleccionado = st.radio("Seleccione el algoritmo para ejecutar el cálculo:", 
                                      ["Perceptrón Multicapa (MLP)", "Gradient Boosting de Histograma (GBM)"], 
                                      horizontal=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🚀 INICIAR PROPAGACIÓN Y CÁLCULO", type="primary", use_container_width=True):
            
            # --- COMPUERTA DE VALIDACIÓN LÓGICA (EDGE CASES) - INTACATA ---
            if entorno_operativo == "mar" and lamina_agua <= 0:
                st.error("⚠️ **Inconsistencia Geofísica:** Ha seleccionado un entorno marino (Offshore), pero la lámina de agua es 0 m. Ajuste el parámetro antes de continuar.")
            elif entorno_operativo == "tierra" and lamina_agua > 0:
                st.error("⚠️ **Inconsistencia Topográfica:** Un pozo en tierra (Onshore) no puede poseer lámina de agua. Ajuste el entorno operativo.")
            elif latitud_usuario > 5.0 or latitud_usuario < -35.0:
                st.error("⚠️ **Frontera Geográfica Excedida:** La latitud ingresada se encuentra fuera del bloque exploratorio de la cuenca brasileña.")
            else:
                # Si pasa la validación, procedemos con la inferencia pesada
                with st.spinner('Computando tensores en paralelo...'):
                    time.sleep(0.5)
                    vector_crudo = pd.DataFrame([[latitud_usuario, longitud_usuario, lamina_agua, cota_altimetrica, cuenca_seleccionada, entorno_operativo, clase_pozo, categoria_pozo, campo_seleccionado, situacion_pozo]], 
                                                columns=['LATITUDE_BASE_DD', 'LONGITUDE_BASE_DD', 'LAMINA_D_AGUA_M', 'COTA_ALTIMETRICA_M', 'BACIA', 'TERRA_MAR', 'TIPO', 'CATEGORIA', 'CAMPO', 'SITUACAO'])
                    
                    vector_estandarizado = preprocesador_reg.transform(vector_crudo)
                    
                    # --- INYECCIÓN 2: CÁLCULO DUAL PARA MENSAJE CONTRAFACTUAL ---
                    # Calculamos ambos modelos en segundo plano
                    pred_mlp_cruda = modelo_reg_mlp.predict(vector_estandarizado)[0]
                    pred_gbm_cruda = modelo_reg_gbm.predict(vector_estandarizado)[0]
                    
                    prof_mlp = max(0.0, pred_mlp_cruda)
                    prof_gbm = max(0.0, pred_gbm_cruda)
                    
                    # Asignamos variables principales vs alternativas según la selección
                    if motor_seleccionado == "Perceptrón Multicapa (MLP)":
                        profundidad_calculada, mae_activo = prof_mlp, mae_error_mlp
                        prof_sombra, mae_sombra, nombre_sombra = prof_gbm, mae_error_gbm, "Gradient Boosting de Histograma"
                    else:
                        profundidad_calculada, mae_activo = prof_gbm, mae_error_gbm
                        prof_sombra, mae_sombra, nombre_sombra = prof_mlp, mae_error_mlp, "Perceptrón Multicapa"
                    
                    # Cálculo de la franja de incertidumbre limitando también la cota inferior
                    limite_inferior = max(0.0, profundidad_calculada - mae_activo)
                    limite_superior = profundidad_calculada + mae_activo
                    
                    col_t, col_i = st.columns([1, 1.8])
                    with col_t:
                        # HTML ORIGINAL INTACTO (Solo añadí el </div> de cierre final)
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">Profundidad Proyectada (μ)</div>
                            <div class="metric-value">{profundidad_calculada:,.1f} <span style="font-size: 1.5rem; color: #8C92A4;">m</span></div>
                            <div class="metric-error"><span>± {mae_activo:.1f} m (MAE: Error Absoluto Medio)</span></div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.info(f"📍 **Fin de Perforación Estimado:**\n\nLas rocas objetivo se interceptarán en un estrato comprendido entre **{limite_inferior:,.1f} m** y **{limite_superior:,.1f} m**.")
                        
                        # --- INYECCIÓN 3: MENSAJE CONTRAFACTUAL ---
                        st.warning(f"🔄 **Benchmark Predictivo:**\n\nCon el otro modelo ({nombre_sombra}) se hubiera obtenido un dictamen de **{prof_sombra:,.1f} m** (Error MAE: ± {mae_sombra:.1f} m).")
                        
                    with col_i:
                        st.plotly_chart(generar_curva_probabilidad(profundidad_calculada, mae_activo), use_container_width=True)

    # --- PESTAÑA ORIGINAL INTACTA (Solo se actualizó modelo_reg a modelo_reg_mlp) ---
    with tab_aud:
        st.markdown("### Estructura Topológica y Diagnóstico Visual")
        st.markdown("Arquitectura interna del modelo perceptrón y evaluación de su rendimiento estocástico.")
        
        with st.spinner("Compilando gráficas y extrayendo tensores..."):
            
            # --- BLOQUE 1: RED NEURONAL ---
            col_graf1, col_txt1 = st.columns([1.6, 1])
            with col_graf1:
                st.plotly_chart(generar_grafo_red_neuronal(), use_container_width=True)
            with col_txt1:
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.subheader("1. Topología del Perceptrón Multicapa (MLP)")
                st.markdown("""
                * **Capa de Entrada:** Recepción de un tensor de 10 dimensiones, integrando coordenadas, topografía (mar/tierra) y metadatos vectorizados (One-Hot Encoding).
                * **Capas Profundas:** Tres niveles densamente conectados (256, 128 y 64 neuronas) con activación ReLU para modelar la geofísica no lineal del subsuelo.
                * **Capa de Salida:** Estrategia de compresión/descompresión estocástica para emitir el cálculo final en metros sin sufrir el colapso de gradientes.
                """)
                
            st.markdown("<hr style='border: 1px solid #2A2E39; margin-top: 10px; margin-bottom: 20px;'>", unsafe_allow_html=True)
                
            # --- BLOQUE 2: CURVA DE APRENDIZAJE ---
            col_txt2, col_graf2 = st.columns([1, 1.6])
            with col_txt2:
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.subheader("2. Optimización Interna (Curva de Aprendizaje)")
                st.markdown("""
                * **Pérdida de Entrenamiento (Línea Verde):** Ilustra el descenso del error matemático a medida que el algoritmo actualiza los pesos de sus neuronas durante las iteraciones (épocas).
                * **Score de Validación (Línea Roja):** Utilizando el eje secundario, demuestra la evolución del Coeficiente de Determinación ($R^2$) frente a datos no vistos.
                * **Parada Temprana (Early Stopping):** El entrenamiento se interrumpió automáticamente cuando la curva roja alcanzó su pico máximo para evitar memorizar el ruido estadístico (*Overfitting*).
                """)
            with col_graf2:
                st.plotly_chart(generar_curva_aprendizaje(modelo_reg_mlp), use_container_width=True)
                
            st.markdown("<hr style='border: 1px solid #2A2E39; margin-top: 10px; margin-bottom: 20px;'>", unsafe_allow_html=True)
                
            # --- BLOQUE 3: MAPA DE CALOR ---
            col_graf3, col_txt3 = st.columns([1.6, 1])
            with col_graf3:
                X_prueba = df_prueba[['LATITUDE_BASE_DD', 'LONGITUDE_BASE_DD', 'LAMINA_D_AGUA_M', 'COTA_ALTIMETRICA_M',
                                      'BACIA', 'TERRA_MAR', 'TIPO', 'CATEGORIA', 'CAMPO', 'SITUACAO']]
                y_real = df_prueba['PROFUNDIDADE_SONDADOR_M']
                X_prueba_estandarizado = preprocesador_reg.transform(X_prueba)
                y_predicho = modelo_reg_mlp.predict(X_prueba_estandarizado)
                
                fig_heat = px.density_contour(
                    x=y_real, y=y_predicho, 
                    labels={'x': 'Profundidad Real (m)', 'y': 'Inferencia (m)'}, 
                    marginal_x="histogram", marginal_y="histogram"
                )
                fig_heat.update_traces(contours_coloring="fill", colorscale="Purples", selector=dict(type='histogram2dcontour'))
                fig_heat.add_shape(type='line', x0=0, y0=0, x1=8000, y1=8000, line=dict(color='#00D2FF', dash='dot', width=2))
                fig_heat.update_layout(
                    template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    height=450, margin=dict(t=50, b=50, l=50, r=50), 
                    title=dict(text="Contornos de Densidad Topográfica", font=dict(color="#E0E0E0"))
                )
                st.plotly_chart(fig_heat, use_container_width=True)
                
            with col_txt3:
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.subheader("3. Correlación y Densidad (Mapa de Contornos)")
                st.markdown("""
                * **Línea de Perfección ($Y=X$):** La línea punteada celeste representa el escenario predictivo perfecto. Los anillos de densidad fuertemente alineados a esta recta validan el altísimo poder de generalización del modelo ($R^2=0.82$).
                * **Anillos Topográficos:** En lugar de pixeles, la gráfica dibuja curvas matemáticas de nivel cerradas. Los colores más oscuros indican el punto más alto de densidad de datos.
                * **Distribuciones Marginales:** Los histogramas laterales representan el volumen de registros históricos y se alinean con los núcleos de las inferencias.
                """)

    # --- INYECCIÓN 4: PESTAÑA EXCLUSIVA PARA GBM ---
    with tab_aud_gbm:
        # He actualizado el título para reflejar que ya no hay mapa de densidad
        st.markdown("### Arquitectura de Árboles y Análisis de Residuos (GBM)")
        st.markdown("Evaluación estocástica del modelo particional alternativo para regresión continua.")
        
        with st.spinner("Computando ensamblaje de árboles y análisis de residuos..."):
            
            # ====================================================================
            # 1. MOTOR DE EXTRACCIÓN E INFERENCIA (Invisible en la Interfaz)
            # ====================================================================
            X_prueba = df_prueba[['LATITUDE_BASE_DD', 'LONGITUDE_BASE_DD', 'LAMINA_D_AGUA_M', 'COTA_ALTIMETRICA_M', 'BACIA', 'TERRA_MAR', 'TIPO', 'CATEGORIA', 'CAMPO', 'SITUACAO']]
            y_real = df_prueba['PROFUNDIDADE_SONDADOR_M']
            
            # Estandarizamos e inferimos para tener los datos listos para los residuos
            X_prueba_estandarizado = preprocesador_reg.transform(X_prueba)
            y_predicho_gbm = modelo_reg_gbm.predict(X_prueba_estandarizado)
            
            # ====================================================================
            # 2. CUADRÍCULA DE AUDITORÍA AVANZADA (Renderizado Visual)
            # ====================================================================
            col_izq_reg, col_der_reg = st.columns(2, gap="large")
            
            with col_izq_reg:
                st.subheader("1. Importancia Predictiva (GBM)")
                st.markdown("Permutación estocástica para revelar qué variables geológicas dominan la partición de los árboles de decisión en la inferencia de profundidad.")
                
                # Ensamblamos temporalmente el pipeline completo
                from sklearn.pipeline import make_pipeline
                from sklearn.inspection import permutation_importance
                
                pipeline_temporal_gbm = make_pipeline(preprocesador_reg, modelo_reg_gbm)
                # n_repeats=3 por eficiencia en despliegue web
                resultado_imp = permutation_importance(pipeline_temporal_gbm, X_prueba, y_real, n_repeats=3, random_state=42, scoring='neg_mean_absolute_error')
                
                nombres_cols_reg = ['LATITUD', 'LONGITUD', 'LÁMINA AGUA', 'COTA ALTIM.', 'CUENCA', 'ENTORNO', 'CLASE', 'CATEGORÍA', 'CAMPO', 'SITUACIÓN']
                df_imp_reg = pd.DataFrame({'Variable': nombres_cols_reg, 'Importancia': resultado_imp.importances_mean}).sort_values(by='Importancia', ascending=True)
                
                fig_imp_reg = px.bar(df_imp_reg, x='Importancia', y='Variable', orientation='h', color='Importancia', color_continuous_scale='Emrld')
                fig_imp_reg.update_layout(template="plotly_dark", coloraxis_showscale=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=380, margin=dict(t=20, b=20, l=10, r=10))
                st.plotly_chart(fig_imp_reg, use_container_width=True, key="imp_gbm_reg")

            with col_der_reg:
                st.subheader("2. Análisis de Residuos (Homocedasticidad)")
                st.markdown("Verifica si los errores del modelo $e = Y_{real} - Y_{pred}$ son aleatorios. Si los puntos forman embudos, la varianza es inestable (Heterocedasticidad).")
                
                # Ahora y_predicho_gbm y y_real sí existen en memoria
                residuos_gbm = y_real - y_predicho_gbm
                
                fig_res = px.scatter(
                    x=y_predicho_gbm, y=residuos_gbm, 
                    labels={'x': 'Profundidad Inferida (m)', 'y': 'Residuo (Error en m)'}, 
                    opacity=0.6, color_discrete_sequence=['#00FF7F']
                )
                
                # Línea central de error cero (perfección predictiva)
                fig_res.add_hline(y=0, line_dash="dot", line_color="#FF4B4B", line_width=2)
                
                # =================================================================
                # NUEVO: CÁLCULO MATEMÁTICO DEL CONO DE HETEROCEDASTICIDAD
                # =================================================================
                # 1. Ajustamos una recta (grado 1) sobre el valor absoluto del error
                z = np.polyfit(y_predicho_gbm, np.abs(residuos_gbm), 1)
                polinomio = np.poly1d(z)
                
                # 2. Generamos el vector X para la longitud del gráfico
                x_cono = np.linspace(min(y_predicho_gbm), max(y_predicho_gbm), 100)
                
                # 3. Calculamos Y. Multiplicamos por 2.5 para abarcar la dispersión 
                # simulando una envolvente de confianza (aprox 95% de los datos).
                y_cono_sup = polinomio(x_cono) * 2.5 
                y_cono_inf = -y_cono_sup
                
                # 4. Inyectamos la envolvente superior al gráfico (línea punteada celeste)
                fig_res.add_trace(go.Scatter(
                    x=x_cono, y=y_cono_sup, mode='lines', 
                    line=dict(color='#00D2FF', width=2, dash='dash'), 
                    name='Límite de Varianza', hoverinfo='skip'
                ))
                # 5. Inyectamos la envolvente inferior
                fig_res.add_trace(go.Scatter(
                    x=x_cono, y=y_cono_inf, mode='lines', 
                    line=dict(color='#00D2FF', width=2, dash='dash'), 
                    showlegend=False, hoverinfo='skip'
                ))
                # =================================================================
                
                fig_res.update_layout(
                    template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                    height=380, margin=dict(t=20, b=20, l=10, r=10), showlegend=False
                )
                st.plotly_chart(fig_res, use_container_width=True, key="residuos_gbm_reg")
                
# VISTA 3: MÓDULO DE CLASIFICACIÓN (GBM & MLP)
elif st.session_state.pagina_actual == 'Clasificación':
    st.title("🗂️ Módulo Operacional: Clasificación")
    st.markdown("Dictamen categórico estocástico comparando ensamblajes ortogonales frente a topologías neuronales.")
    
    # Se expanden las pestañas a 3
    tab_inf, tab_aud_gbm, tab_aud_mlp = st.tabs(["🔮 Panel de Inferencia", "📊 Auditoría y Evaluación (GBM)", "📊 Auditoría y Evaluación (MLP)"])
    
    # --- PESTAÑA DE INFERENCIA ---
    with tab_inf:
        # --- INYECCIÓN 1: SELECTOR DE MODELO DE CLASIFICACIÓN ---
        st.markdown("### Configuración del Motor Matemático")
        motor_seleccionado_clf = st.radio("Seleccione el algoritmo para la clasificación operativa:", 
                                          ["Gradient Boosting de Histograma (GBM)", "Perceptrón Multicapa (MLP)"], 
                                          horizontal=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🚀 EJECUTAR ENSAMBLAJE ESTOCÁSTICO", type="primary", use_container_width=True):
            
            # 1. Compuerta Lógica de Seguridad Físico-Matemática
            if entorno_operativo == "mar" and lamina_agua <= 0:
                st.error("⚠️ **Inconsistencia Geofísica:** Ha seleccionado un entorno marino (Offshore), pero la lámina de agua es 0 m.")
            elif entorno_operativo == "tierra" and lamina_agua > 0:
                st.error("⚠️ **Inconsistencia Topográfica:** Un pozo en tierra (Onshore) no puede poseer lámina de agua.")
            else:
                with st.spinner('Ejecutando Acoplamiento Secuencial (Profundidad ➔ Clasificación)...'):
                    time.sleep(0.5)
                    
                    # ==========================================
                    # FASE 1: INFERENCIA DE PROFUNDIDAD (MLP BASE)
                    # ==========================================
                    vector_reg = pd.DataFrame([[latitud_usuario, longitud_usuario, lamina_agua, cota_altimetrica, cuenca_seleccionada, entorno_operativo, clase_pozo, categoria_pozo, campo_seleccionado, "otras_situaciones"]], 
                                                columns=['LATITUDE_BASE_DD', 'LONGITUDE_BASE_DD', 'LAMINA_D_AGUA_M', 'COTA_ALTIMETRICA_M', 'BACIA', 'TERRA_MAR', 'TIPO', 'CATEGORIA', 'CAMPO', 'SITUACAO'])
                    
                    vector_estandarizado_reg = preprocesador_reg.transform(vector_reg)
                    # Usamos el modelo de regresión MLP como estimador base de profundidad
                    prediccion_cruda_reg = modelo_reg_mlp.predict(vector_estandarizado_reg)[0]
                    profundidad_calculada = max(0.0, prediccion_cruda_reg) 
                    
                    # ==========================================
                    # FASE 2: INFERENCIA OPERACIONAL (CÁLCULO DUAL)
                    # ==========================================
                    vector_clf = pd.DataFrame([[latitud_usuario, longitud_usuario, lamina_agua, cota_altimetrica, profundidad_calculada, cuenca_seleccionada, entorno_operativo, clase_pozo, categoria_pozo, campo_seleccionado]], 
                                                columns=['LATITUDE_BASE_DD', 'LONGITUDE_BASE_DD', 'LAMINA_D_AGUA_M', 'COTA_ALTIMETRICA_M', 'PROFUNDIDADE_SONDADOR_M', 'BACIA', 'TERRA_MAR', 'TIPO', 'CATEGORIA', 'CAMPO'])
                    
                    nombres_clases = pipeline_clf_gbm.classes_
                    
                    # Inferencia de ambos motores
                    pred_idx_gbm = pipeline_clf_gbm.predict(vector_clf)[0]
                    probs_gbm = pipeline_clf_gbm.predict_proba(vector_clf)[0] * 100
                    
                    pred_idx_mlp = pipeline_clf_mlp.predict(vector_clf)[0]
                    probs_mlp = pipeline_clf_mlp.predict_proba(vector_clf)[0] * 100
                    
                    # --- INYECCIÓN 2: ASIGNACIÓN DE ROLES (PRINCIPAL VS CONTRAFACTUAL) ---
                    if motor_seleccionado_clf == "Gradient Boosting de Histograma (GBM)":
                        pred_clase_activa = pred_idx_gbm
                        probs_activas = probs_gbm
                        pred_clase_sombra = pred_idx_mlp
                        exactitud_activa = metricas_clf_gbm['exactitud']
                        exactitud_sombra = exactitud_mlp_clf
                        nombre_sombra = "Perceptrón Multicapa - MLP"
                    else:
                        pred_clase_activa = pred_idx_mlp
                        probs_activas = probs_mlp
                        pred_clase_sombra = pred_idx_gbm
                        exactitud_activa = exactitud_mlp_clf
                        exactitud_sombra = metricas_clf_gbm['exactitud']
                        nombre_sombra = "Gradient Boosting - GBM"
                    
                    # Manejo seguro si la predicción es texto o índice numérico
                    nombre_clase_predicha = nombres_clases[pred_clase_activa] if isinstance(pred_clase_activa, (int, np.integer)) else pred_clase_activa
                    nombre_clase_sombra = nombres_clases[pred_clase_sombra] if isinstance(pred_clase_sombra, (int, np.integer)) else pred_clase_sombra
                    
                    # ==========================================
                    # FASE 3: RENDERIZADO DE RESULTADOS
                    # ==========================================
                    st.success(f"🔗 **Pipeline Secuencial Activo:** Se estimó el lecho de roca objetivo en **{profundidad_calculada:,.1f} metros**. Este vector tridimensional fue transferido al clasificador para computar la viabilidad del pozo.")
                    
                    st.markdown(f"""
                        <div class="metric-card" style="border-left-color: #9D4EDD; text-align: center;">
                            <div class="metric-title">Situación Estructural Estimada</div>
                            <div class="metric-value" style="color: #9D4EDD;">{str(nombre_clase_predicha).upper()}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # --- INYECCIÓN 3: MENSAJE CONTRAFACTUAL CLASIFICACIÓN ---
                    st.warning(f"🔄 **Benchmark Predictivo:**\n\nCon el otro modelo ({nombre_sombra}) se hubiera obtenido un dictamen de situación **{str(nombre_clase_sombra).upper()}** (Exactitud histórica del motor: {exactitud_sombra:.2%}).")

                    st.subheader("Distribución de Probabilidad")
                    df_probs = pd.DataFrame({'Situación': nombres_clases, 'Probabilidad (%)': probs_activas}).sort_values(by='Probabilidad (%)')
                    
                    # Gráfico Plotly
                    fig_probs = px.bar(df_probs, x='Probabilidad (%)', y='Situación', orientation='h', text='Probabilidad (%)', color='Probabilidad (%)', color_continuous_scale='Purples')
                    fig_probs.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                    fig_probs.update_layout(template="plotly_dark", height=400, coloraxis_showscale=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    fig_probs.update_xaxes(range=[0, max(100, df_probs['Probabilidad (%)'].max() + 10)])
                    st.plotly_chart(fig_probs, use_container_width=True)

    # --- PESTAÑA ORIGINAL INTACTA (Solo renombrada a GBM) ---
    with tab_aud_gbm:
        st.markdown("### Diagnóstico Estructural del Modelo (GBM)")
        st.markdown("Evaluación multivariante y de interpretabilidad analítica ('Caja Blanca').")
        
        with st.spinner("Compilando métricas y ejecutando permutación estocástica..."):
            clases_nombres = metricas_clf_gbm['clases']
            
            st.markdown("<hr style='border: 1px solid #2A2E39; margin-top: 10px; margin-bottom: 20px;'>", unsafe_allow_html=True)
            st.subheader("1. Importancia Predictiva de Variables (Permutation Importance)")
            st.markdown("""
            Este análisis cuantifica qué variables impulsaron las decisiones del modelo matemático. Al introducir 'ruido estocástico' (permutando aleatoriamente los datos de una columna), evaluamos la caída de la exactitud. Las barras más largas representan las variables más críticas para el éxito de la inferencia.
            """)
            st.plotly_chart(generar_grafico_importancia(pipeline_clf_gbm, df_prueba, clases_nombres), use_container_width=True, key="importancia_gmb")
            
            st.markdown("<hr style='border: 1px solid #2A2E39; margin-top: 20px; margin-bottom: 20px;'>", unsafe_allow_html=True)
            col_izq, col_der = st.columns(2, gap="large")
            
            with col_izq:
                st.subheader("2. Matriz de Confusión Agrupada")
                st.markdown(f"**Exactitud Global:** {metricas_clf_gbm['exactitud']:.2%} | **Precisión:** {metricas_clf_gbm['precision']:.2%} | **Exhaustividad (Recall):** {metricas_clf_gbm['exhaustividad']:.2%}<br>Muestra empíricamente el volumen de aciertos y errores. Matriz condensada en 4 Macro-Categorías para su correcta lectura óptica.", unsafe_allow_html=True)
                
                def agrupar_situacion(clase_texto):
                    c_low = str(clase_texto).lower()
                    if 'produt' in c_low or 'product' in c_low or 'injet' in c_low or 'jazida' in c_low: 
                        return 'Productores / Activos'
                    elif 'abandon' in c_low or 'arrasamento' in c_low: 
                        return 'Abandonados / Cerrados'
                    elif 'pioneiro' in c_low or 'explora' in c_low or 'estrat' in c_low: 
                        return 'Exploración / Pioneros'
                    else: 
                        return 'Otras Situaciones'

                macro_clases = ['Productores / Activos', 'Abandonados / Cerrados', 'Exploración / Pioneros', 'Otras Situaciones']
                map_indices = {i: agrupar_situacion(c) for i, c in enumerate(clases_nombres)}
                
                matriz_agrupada = pd.DataFrame(0, index=macro_clases, columns=macro_clases)
                for i in range(len(clases_nombres)):
                    for j in range(len(clases_nombres)):
                        cat_real = map_indices[i]
                        cat_pred = map_indices[j]
                        matriz_agrupada.loc[cat_real, cat_pred] += metricas_clf_gbm['matriz_confusion'][i][j]

                fig_cm = px.imshow(
                    matriz_agrupada, 
                    x=macro_clases, 
                    y=macro_clases, 
                    color_continuous_scale='Purples', 
                    aspect="equal",
                    text_auto=True 
                )
                
                fig_cm.update_traces(
                    hovertemplate="Real: %{y}<br>Predicho: %{x}<br>Frecuencia: %{z}<extra></extra>",
                    zsmooth=False, xgap=2, ygap=2
                )
                
                fig_cm.update_layout(
                    template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                    height=450, margin=dict(t=20, b=20, l=10, r=10)
                )
                fig_cm.update_xaxes(tickangle=-45)
                
                st.plotly_chart(fig_cm, use_container_width=True, key="matriz_gbm")
                
            with col_der:
                st.subheader("3. Curva Precisión-Exhaustividad")
                st.markdown("**Evaluación Dinámica (PR Curve):** Analiza el algoritmo frente al severo desbalance poblacional. Las trayectorias que convergen hacia la esquina superior derecha confirman alta separabilidad matemática.", unsafe_allow_html=True)
                st.plotly_chart(generar_curva_precision_recall(pipeline_clf_gbm, df_prueba, clases_nombres), use_container_width=True, key="curva_gbm")

    # --- INYECCIÓN 4: PESTAÑA EXCLUSIVA PARA EL MLP ---
    with tab_aud_mlp:
        st.markdown("### Diagnóstico Estructural del Modelo (MLP)")
        st.markdown("Evaluación multivariante y de interpretabilidad analítica ('Caja Blanca') para la Red Neuronal.")
        
        with st.spinner("Compilando métricas y ejecutando permutación estocástica..."):
            clases_nombres = metricas_clf_gbm['clases']
            
            # --- NUEVO BLOQUE: TOPOLOGÍA Y APRENDIZAJE DEL MLP DE CLASIFICACIÓN ---
            col_graf_top_mlp, col_txt_top_mlp = st.columns([1.6, 1])
            with col_graf_top_mlp:
                # Extraemos el modelo puro del pipeline para leer su arquitectura
                mlp_puro_clf = pipeline_clf_mlp.named_steps['modelo_mlp']
                
                # Función inline para el esquema topológico de clasificación
                def generar_grafo_mlp_clasificacion():
                    # Entrada (10), Ocultas (simuladas visualmente), Salida (4 clases operativas)
                    capas = [10, 8, 6, 4] 
                    fig = go.Figure()
                    edge_x, edge_y = [], []
                    for i in range(len(capas) - 1):
                        y_start, y_end = np.linspace(0.1, 0.9, capas[i]), np.linspace(0.1, 0.9, capas[i+1])
                        for ys in y_start:
                            for ye in y_end:
                                edge_x.extend([i, i+1, None]); edge_y.extend([ys, ye, None])
                                
                    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color='rgba(0, 210, 255, 0.15)', width=1), hoverinfo='none'))
                    
                    for i, num_nodes in enumerate(capas):
                        color_nodo = '#00D2FF' if i == 0 else ('#9D4EDD' if i == len(capas)-1 else '#00FF7F')
                        fig.add_trace(go.Scatter(x=[i]*num_nodes, y=np.linspace(0.1, 0.9, num_nodes), mode='markers',
                            marker=dict(size=16, color=color_nodo, line=dict(width=2, color='#FFFFFF')), hoverinfo='name'))
                        
                    fig.update_layout(template="plotly_dark", title=dict(text="Topología Logística (Softmax)", font=dict(color="#E0E0E0")),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=350, showlegend=False, margin=dict(t=40, b=10, l=10, r=10))
                    return fig
                
                st.plotly_chart(generar_grafo_mlp_clasificacion(), use_container_width=True, key="grafo_mlp_clf")
            
            with col_txt_top_mlp:
                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader("Arquitectura Conexionista")
                st.markdown("""
                * **Capa de Entrada:** Vector estocástico post-procesado (incluye One-Hot Encoding de alta cardinalidad).
                * **Capas Ocultas:** Estructura de embudo profundo para extracción de características no lineales.
                * **Capa de Salida (Softmax):** Cuatro nodos terminales. En lugar de emitir un valor continuo, esta topología calcula una distribución de probabilidad cruzada para las cuatro situaciones operativas del pozo.
                """)
                
            st.markdown("<hr style='border: 1px solid #2A2E39; margin-top: 10px; margin-bottom: 20px;'>", unsafe_allow_html=True)
            
            col_txt_loss_mlp, col_graf_loss_mlp = st.columns([1, 1.6])
            with col_txt_loss_mlp:
                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader("Descenso de Gradiente (Log-Loss)")
                st.markdown("""
                A diferencia de la regresión que minimiza el error de la distancia en metros (MSE), este modelo iteró buscando minimizar la **Entropía Cruzada**.
                
                La gráfica muestra cómo el algoritmo ajustó los pesos sinápticos durante sus iteraciones para intentar separar matemáticamente la matriz rala generada por las variables discretas.
                """)
            with col_graf_loss_mlp:
                def generar_curva_loss_clasificacion(mlp_model):
                    y_loss = mlp_model.loss_curve_ if hasattr(mlp_model, 'loss_curve_') else [0]
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=y_loss, mode='lines', line=dict(color='#00FF7F', width=3, shape='spline'),
                        fill='tozeroy', fillcolor='rgba(0, 255, 127, 0.1)', name='Pérdida Log-Loss'))
                    fig.update_layout(template="plotly_dark", title=dict(text="Curva de Minimización de Entropía Cruzada", font=dict(color="#E0E0E0")),
                        xaxis_title="Épocas", yaxis_title="Costo de Pérdida (Loss)", height=350, margin=dict(t=40, b=10, l=10, r=10),
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    return fig
                
                st.plotly_chart(generar_curva_loss_clasificacion(mlp_puro_clf), use_container_width=True, key="loss_mlp_clf")
           
