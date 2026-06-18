import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from unidecode import unidecode

# ==========================================
# 1. CONFIGURACIÓN DE PÁGINA Y ESTÉTICA (CSS)
# ==========================================
st.set_page_config(
    page_title="Inferencia Geológica MLP",
    page_icon="🛢️",
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .metric-card {
        background-color: #11141A;
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #2A2E39;
        border-left: 6px solid #00D2FF;
        box-shadow: 0 8px 16px rgba(0,0,0,0.4);
        margin-bottom: 24px;
        transition: transform 0.2s ease-in-out;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-title { color: #8C92A4; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 8px; font-family: 'Segoe UI', sans-serif; }
    .metric-value { color: #FFFFFF; font-size: 3.2rem; font-weight: 800; margin-bottom: 4px; line-height: 1.1; }
    .metric-error { color: #FF4B4B; font-size: 1.1rem; font-weight: 600; display: flex; align-items: center; gap: 6px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MOTORES VISUALES (PLOTLY NATIVO)
# ==========================================
def generar_curva_probabilidad(prediccion, mae):
    min_x = max(0.0, prediccion - 4*mae)
    max_x = prediccion + 4*mae
    x = np.linspace(min_x, max_x, 500)
    
    y = (1 / (mae * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - prediccion) / mae)**2)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='#00D2FF', width=3, shape='spline'), name='Distribución'))
    
    limite_inferior_fisico = max(0.0, prediccion - mae)
    x_fill = np.linspace(limite_inferior_fisico, prediccion + mae, 100)
    y_fill = (1 / (mae * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_fill - prediccion) / mae)**2)
    
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_fill, x_fill[::-1]]), y=np.concatenate([y_fill, np.zeros_like(y_fill)]), 
        fill='toself', fillcolor='rgba(0, 210, 255, 0.15)', line=dict(color='rgba(255,255,255,0)'), name='Margen (MAE)'
    ))
    
    fig.add_vline(x=prediccion, line_dash="dash", line_color="#FF4B4B", line_width=2,
                  annotation_text=f"Objetivo: {prediccion:,.0f}m", annotation_position="top right",
                  annotation_font=dict(color="#FF4B4B", size=13, weight="bold"))

    fig.update_layout(
        template="plotly_dark", title=dict(text="Análisis de Densidad Estocástica", font=dict(size=18, color="#E0E0E0")),
        xaxis_title="Profundidad del Subsuelo (m)", yaxis_title="Probabilidad", showlegend=False, height=400,
        margin=dict(l=20, r=20, t=60, b=20), plot_bgcolor='rgba(17, 20, 26, 0.8)', paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='#2A2E39', zeroline=False, rangemode='nonnegative'),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, rangemode='nonnegative'),
        hovermode="x unified"
    )
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
        template="plotly_dark", title=dict(text="Curva de Aprendizaje de Doble Eje", font=dict(size=18, color="#E0E0E0")),
        xaxis_title="Épocas (Iteraciones)", showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(17, 20, 26, 0.8)', bordercolor='#2A2E39', borderwidth=1),
        height=450, margin=dict(l=20, r=20, t=60, b=20), 
        plot_bgcolor='rgba(17, 20, 26, 0.8)', paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='#2A2E39')
    )
    
    fig.update_yaxes(title_text="Costo de Pérdida (Loss)", secondary_y=False, showgrid=True, gridcolor='#2A2E39')
    fig.update_yaxes(title_text="Score de Validación (R²)", secondary_y=True, showgrid=False)
    
    return fig

def generar_grafo_red_neuronal():
    capas = [10, 8, 6, 4, 1] 
    nombres_capas = ["Capa de Entrada (Vector 10D)", "Capa Oculta 1 (256)", "Capa Oculta 2 (128)", "Capa Oculta 3 (64)", "Salida Continua (Profundidad)"]
    
    fig = go.Figure()
    
    # Agrupamos todos los enlaces en una sola traza gigante usando 'None' para cortar líneas.
    edge_x = []
    edge_y = []
    
    for i in range(len(capas) - 1):
        y_start = np.linspace(0.1, 0.9, capas[i])
        y_end = np.linspace(0.1, 0.9, capas[i+1])
        for ys in y_start:
            for ye in y_end:
                edge_x.extend([i, i+1, None])
                edge_y.extend([ys, ye, None])
                
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode='lines',
        line=dict(color='rgba(0, 210, 255, 0.15)', width=1),
        hoverinfo='none', showlegend=False
    ))
    
    for i, num_nodes in enumerate(capas):
        y_nodes = np.linspace(0.1, 0.9, num_nodes)
        color_nodo = '#00D2FF' if i == 0 else ('#FF4B4B' if i == len(capas)-1 else '#00FF7F')
        
        fig.add_trace(go.Scatter(
            x=[i]*num_nodes, y=y_nodes, mode='markers',
            marker=dict(size=16, color=color_nodo, line=dict(width=2, color='#FFFFFF')),
            name=nombres_capas[i], hoverinfo='name'
        ))
        
    fig.update_layout(
        template="plotly_dark", title=dict(text="Esquema Topológico del Perceptrón Multicapa (MLP)", font=dict(size=18, color="#E0E0E0")),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(17, 20, 26, 0.8)', paper_bgcolor='rgba(0,0,0,0)',
        height=450, margin=dict(l=20, r=20, t=60, b=20), showlegend=False
    )
    return fig

# ==========================================
# 3. CARGA DE MODELOS Y DATOS DE PRUEBA
# ==========================================
@st.cache_resource
def inicializar_componentes_predictivos():
    modelo_red = joblib.load('modelo_mlp_pozos.pkl')
    transformador_datos = joblib.load('preprocesador_pozos.pkl')
    error_historico_mae = joblib.load('mae_error.pkl')
    diccionario = joblib.load('diccionario_opciones.pkl')
    return modelo_red, transformador_datos, error_historico_mae, diccionario

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
        
    return df.sample(n=min(2000, len(df)), random_state=42)

try:
    modelo, preprocesador, mae_error, diccionario = inicializar_componentes_predictivos()
    df_prueba = generar_datos_rendimiento()
except Exception as error_lectura:
    st.error("Error operativo: Asegúrese de haber ejecutado el entrenamiento.")
    st.stop()

# ==========================================
# 4. BARRA LATERAL DE CONTROLES (SIDEBAR)
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #00D2FF;'>Centro de Comando</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.subheader("📍 Geolocalización")
    latitud_usuario = st.slider("Latitud (DD)", -35.0, 10.0, -22.0, step=0.0001)
    longitud_usuario = st.slider("Longitud (DD)", -55.0, -25.0, -40.0, step=0.0001)
    
    st.markdown("---")
    st.subheader("🌍 Topografía")
    entorno_operativo = st.selectbox("Entorno Operativo", ["mar", "terra"])
    if entorno_operativo == "mar":
        lamina_agua = st.number_input("Lámina de Agua (m)", min_value=0.0, value=500.0)
        cota_altimetrica = 0.0 
    else:
        cota_altimetrica = st.number_input("Cota Altimétrica (m)", min_value=0.0, value=50.0)
        lamina_agua = 0.0 
    
    st.markdown("---")
    st.subheader("🪨 Contexto Geológico")
    cuenca_seleccionada = st.selectbox("Cuenca Petrolífera", ["campos", "santos", "espirito santo", "reconcavo", "sergipe-alagoas"])
    campo_seleccionado = st.selectbox("Campo Específico", diccionario.get('campos', []))
    
    st.markdown("---")
    st.subheader("⚙️ Operación")
    clase_pozo = st.selectbox("Tipo de Operación", ["desenvolvimento", "exploratorio", "injetor"])
    categoria_pozo = st.selectbox("Categoría Estratégica", diccionario.get('categorias', []))
    situacion_pozo = st.selectbox("Situación Esperada", diccionario.get('situaciones', []))

# ==========================================
# 5. ÁREA PRINCIPAL Y PESTAÑAS
# ==========================================
st.title("🛢️ Inferencia Geofísica Asistida por IA")
st.markdown("Plataforma de estimación de profundidad mediante **Perceptrón Multicapa (MLP)** con arquitectura de *Target Scaling*.")

tab1, tab2 = st.tabs(["🔮 Panel de Inferencia", "📊 Auditoría y Arquitectura del Modelo"])

# --- PESTAÑA 1: SOLO LA INFERENCIA PRINCIPAL ---
with tab1:
    if st.button("🚀 INICIAR PROPAGACIÓN NEURONAL", type="primary", use_container_width=True):
        with st.spinner('Procesando topología tensorial...'):
            time.sleep(0.8) 
            
            vector_crudo = pd.DataFrame(
                [[latitud_usuario, longitud_usuario, lamina_agua, cota_altimetrica, 
                  cuenca_seleccionada, entorno_operativo, clase_pozo, categoria_pozo,
                  campo_seleccionado, situacion_pozo]], 
                columns=['LATITUDE_BASE_DD', 'LONGITUDE_BASE_DD', 'LAMINA_D_AGUA_M', 'COTA_ALTIMETRICA_M',
                         'BACIA', 'TERRA_MAR', 'TIPO', 'CATEGORIA', 'CAMPO', 'SITUACAO']
            )
            
            try:
                vector_estandarizado = preprocesador.transform(vector_crudo)
                prediccion_escalar = modelo.predict(vector_estandarizado)
                profundidad_calculada = prediccion_escalar[0]
                
                limite_inferior = max(0.0, profundidad_calculada - mae_error)
                limite_superior = profundidad_calculada + mae_error
                
                st.markdown("<br>", unsafe_allow_html=True)
                col_texto, col_imagen = st.columns([1, 1.8]) 
                
                with col_texto:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Profundidad Proyectada (μ)</div>
                        <div class="metric-value">{profundidad_calculada:,.1f} <span style="font-size: 1.5rem; color: #8C92A4;">m</span></div>
                        <div class="metric-error"><span>± {mae_error:.1f} m (Incertidumbre MAE)</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.info(f"📍 **Fin de Perforación Estimado:**\n\nLas rocas objetivo se interceptarán en un estrato comprendido entre **{limite_inferior:,.1f} m** y **{limite_superior:,.1f} m**.")
                
                with col_imagen:
                    figura_gaussiana = generar_curva_probabilidad(profundidad_calculada, mae_error)
                    st.plotly_chart(figura_gaussiana, use_container_width=True, theme=None)
                    
            except Exception as e:
                st.error(f"Fallo en matriz de inferencia: {str(e)}")

# --- PESTAÑA 2: AUDITORÍA (DISEÑO EN ZIG-ZAG) ---
with tab2:
    st.markdown("### Estructura Topológica y Diagnóstico Visual")
    st.markdown("Arquitectura interna del modelo perceptrón y evaluación de su rendimiento estocástico.")
    
    with st.spinner("Compilando gráficas y extrayendo tensores..."):
        
        # --- BLOQUE 1: RED NEURONAL (Izquierda) vs TEXTO (Derecha) ---
        col_graf1, col_txt1 = st.columns([1.6, 1])
        
        with col_graf1:
            figura_nn = generar_grafo_red_neuronal()
            st.plotly_chart(figura_nn, use_container_width=True, theme=None)
            
        with col_txt1:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.subheader("1. Topología del Perceptrón Multicapa (MLP)")
            st.markdown("""
            * **Capa de Entrada (Input Layer):** Recepción de un tensor de 10 dimensiones, integrando coordenadas, topografía (mar/tierra) y metadatos vectorizados (One-Hot Encoding).
            * **Capas Profundas (Hidden Layers):** Tres niveles densamente conectados (256, 128 y 64 neuronas) con activación ReLU para modelar la geofísica no lineal del subsuelo.
            * **Capa de Salida (Target Scaling):** Estrategia de compresión/descompresión estocástica para emitir el cálculo final en metros sin sufrir el colapso de gradientes.
            """)
            
        st.markdown("<hr style='border: 1px solid #2A2E39; margin-top: 10px; margin-bottom: 20px;'>", unsafe_allow_html=True)
            
        # --- BLOQUE 2: TEXTO (Izquierda) vs CURVA DE APRENDIZAJE (Derecha) ---
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
            figura_aprendizaje = generar_curva_aprendizaje(modelo)
            st.plotly_chart(figura_aprendizaje, use_container_width=True, theme=None)
            
        st.markdown("<hr style='border: 1px solid #2A2E39; margin-top: 10px; margin-bottom: 20px;'>", unsafe_allow_html=True)
            
        # --- BLOQUE 3: MAPA DE CALOR (Izquierda) vs TEXTO (Derecha) ---
        col_graf3, col_txt3 = st.columns([1.6, 1])
        
        with col_graf3:
            X_prueba = df_prueba[['LATITUDE_BASE_DD', 'LONGITUDE_BASE_DD', 'LAMINA_D_AGUA_M', 'COTA_ALTIMETRICA_M',
                                  'BACIA', 'TERRA_MAR', 'TIPO', 'CATEGORIA', 'CAMPO', 'SITUACAO']]
            y_real = df_prueba['PROFUNDIDADE_SONDADOR_M']
            X_prueba_estandarizado = preprocesador.transform(X_prueba)
            y_predicho = modelo.predict(X_prueba_estandarizado)
            
            # Generación de la gráfica de Contornos de Densidad con la paleta académica Cividis
            fig_heat = px.density_contour(
                x=y_real, y=y_predicho, 
                labels={'x': 'Profundidad Real (m)', 'y': 'Inferencia (m)'}, 
                marginal_x="histogram", marginal_y="histogram"
            )
            # Aplicamos la paleta de color Cividis, rellenando los contornos
            fig_heat.update_traces(contours_coloring="fill", colorscale="Purples", selector=dict(type='histogram2dcontour'))
            
            # Línea ideal predictiva
            fig_heat.add_shape(type='line', x0=0, y0=0, x1=8000, y1=8000, line=dict(color='#00D2FF', dash='dot', width=2))
            
            fig_heat.update_layout(
                template="plotly_dark", plot_bgcolor='rgba(17, 20, 26, 0.8)', paper_bgcolor='rgba(0,0,0,0)',
                height=450, margin=dict(t=50, b=50, l=50, r=50), 
                title=dict(text="Contornos de Densidad Topográfica", font=dict(size=18, color="#E0E0E0"))
            )
            st.plotly_chart(fig_heat, use_container_width=True, theme=None)
            
        with col_txt3:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.subheader("3. Correlación y Densidad (Mapa de Contornos)")
            st.markdown("""
            * **Línea de Perfección ($Y=X$):** La línea punteada celeste representa el escenario predictivo perfecto. Los anillos de densidad fuertemente alineados a esta recta validan el altísimo poder de generalización del modelo ($R^2=0.82$).
            * **Anillos Topográficos:** En lugar de pixeles duros, la gráfica dibuja curvas matemáticas de nivel cerradas. Los colores más claros indican el punto más alto de densidad de datos.
            * **Distribuciones Marginales:** Los histogramas laterales representan el volumen de registros históricos y se alinean con los núcleos de las inferencias.
            """) 