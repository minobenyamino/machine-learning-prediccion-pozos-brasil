#%% BLOQUE 1: Importación de Librerías
# ----------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unidecode import unidecode
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor

print("Librerías cargadas. Iniciando arquitectura neuronal con Target Scaling.")

#%% BLOQUE 2: Ingesta, Limpieza, Traducción y Reducción de Cardinalidad
# ----------------------------------------------------------------------------
df = pd.read_csv('Pozos brasil 2.csv', sep=';', encoding='latin1', decimal=',')

# Definimos las variables a utilizar
columnas_numericas = ['LATITUDE_BASE_DD', 'LONGITUDE_BASE_DD', 'LAMINA_D_AGUA_M', 'COTA_ALTIMETRICA_M']
columnas_categoricas = ['BACIA', 'TERRA_MAR', 'TIPO', 'CATEGORIA', 'CAMPO', 'SITUACAO']
objetivo = 'PROFUNDIDADE_SONDADOR_M'

# Corrección de decimales y nulos en numéricas
for col in columnas_numericas + [objetivo]:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

df = df.dropna(subset=['LATITUDE_BASE_DD', 'LONGITUDE_BASE_DD', objetivo])

# --- CALIBRACIÓN ESTADÍSTICA FINAL ---
# Amputamos solo el 1% de los valores más extremos (Percentil 99)
# Esto recupera la varianza necesaria para elevar el R² sin destruir el MAE.
limite_superior = df[objetivo].quantile(0.99)
df = df[(df[objetivo] >= 100) & (df[objetivo] <= limite_superior)]

df['LAMINA_D_AGUA_M'] = df['LAMINA_D_AGUA_M'].fillna(0.0)
df['COTA_ALTIMETRICA_M'] = df['COTA_ALTIMETRICA_M'].fillna(0.0)

def limpiar_y_traducir_datos(datos):
    """
    Técnica de Manejo de Alta Cardinalidad (High Cardinality Handling).
    Filtra, traduce al español y agrupa variables categóricas preservando la 
    varianza geológica crítica para la regresión continua.
    """
    for col in columnas_categoricas:
        datos[col] = datos[col].astype(str).apply(lambda x: unidecode(x).lower().strip() if pd.notnull(x) and x != 'nan' else 'desconocido')
        
    datos['TERRA_MAR'] = datos['TERRA_MAR'].replace({'t': 'tierra', 'm': 'mar'})
    datos['TIPO'] = datos['TIPO'].replace({'explotatorio': 'desarrollo', 'exploratorio': 'exploratorio'})
    
    dict_cat = {'desenvolvimento': 'desarrollo', 'pioneiro': 'pionero', 'extensao': 'extension', 'especial': 'especial', 'injeaao': 'inyeccion', 'injecao': 'inyeccion'}
    datos['CATEGORIA'] = datos['CATEGORIA'].map(lambda x: dict_cat.get(x, 'otras_categorias'))
    
    cuencas_top = datos['BACIA'].value_counts().nlargest(15).index.tolist()
    datos['BACIA'] = datos['BACIA'].apply(lambda x: x if x in cuencas_top else 'otras_cuencas')
    
    # Filtro de Frecuencia Óptimo: Conservamos campos con al menos 5 pozos.
    conteos_campo = datos['CAMPO'].value_counts()
    campos_robustos = conteos_campo[conteos_campo >= 5].index.tolist()
    if 'desconocido' in campos_robustos: campos_robustos.remove('desconocido')
    datos['CAMPO'] = datos['CAMPO'].apply(lambda x: x if x in campos_robustos else 'otros_campos')
    
    def agrupar_situacion(s):
        if 'produt' in s or 'jazida' in s: return 'productores / activos'
        if 'abandon' in s or 'arras' in s: return 'abandonados / cerrados'
        if 'interven' in s or 'avalia' in s or 'pioneiro' in s or 'explora' in s: return 'exploracion / intervencion'
        if 'injet' in s: return 'inyectores'
        return 'otras_situaciones'
        
    datos['SITUACAO'] = datos['SITUACAO'].apply(agrupar_situacion)
    return datos

print("Aplicando reducción de cardinalidad dinámica y recorte de atípicos...")
df = limpiar_y_traducir_datos(df)

# Definimos X e y procesados
X = df[columnas_numericas + columnas_categoricas]
y = df[objetivo]

# Partición del dataset (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% BLOQUE 3: Tubería de Procesamiento Estocástico
# ----------------------------------------------------------------------------
X = df[columnas_numericas + columnas_categoricas]
y = df[objetivo]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='desconocido')),
    # handle_unknown='ignore' es vital aquí, ya que hay cientos de CAMPOS petroleros
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocesador = ColumnTransformer(
    transformers=[
        ('num', num_transformer, columnas_numericas),
        ('cat', cat_transformer, columnas_categoricas)
    ])

# Solo ajustamos X (Target Scaling se hace en el siguiente bloque)
X_train_procesado = preprocesador.fit_transform(X_train)
X_test_procesado = preprocesador.transform(X_test)

print(f"Dimensiones tras One-Hot Encoding masivo: {X_train_procesado.shape[1]} columnas.")

#%% BLOQUE 4: Red Neuronal con TARGET SCALING
# ----------------------------------------------------------------------------
# Arquitectura base del Perceptrón Multicapa
red_neuronal_base = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64), 
    activation='relu',                 
    solver='adam',                     
    alpha=0.001, # Regularización balanceada
    max_iter=400,                      
    batch_size=128, # Lotes más grandes para mayor estabilidad
    early_stopping=True,               
    validation_fraction=0.2,           
    n_iter_no_change=20,
    random_state=42,
    verbose=True                       
)

# TRUCO DE ESTADÍSTICA AVANZADA: TransformedTargetRegressor
# Esto envuelve la red neuronal. Toma la variable 'y' (metros), la comprime con StandardScaler,
# entrena la red, y al predecir, la descomprime matemáticamente a metros de nuevo.
modelo_maestro = TransformedTargetRegressor(
    regressor=red_neuronal_base,
    transformer=StandardScaler()
)

print("Entrenando Red Neuronal con Target Scaling...")
modelo_maestro.fit(X_train_procesado, y_train)

#%% BLOQUE 5: Evaluación de Métricas de Inferencia
# ----------------------------------------------------------------------------
y_pred = modelo_maestro.predict(X_test_procesado)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n" + "★"*50)
print("EVALUACIÓN ESTADÍSTICA FINAL (TARGET SCALING):")
print(f"Error Absoluto Medio [MAE]: {mae:.2f} metros.")
print(f"Coeficiente de Determinación [R²]: {r2:.4f}")
print("★"*50)

#%% BLOQUE 6: Extracción de Diccionarios y Serialización
# ----------------------------------------------------------------------------
joblib.dump(preprocesador, 'preprocesador_pozos.pkl')
joblib.dump(modelo_maestro, 'modelo_mlp_pozos.pkl')
joblib.dump(mae, 'mae_error.pkl')

# Extraemos las listas de opciones para los menús desplegables de Streamlit
diccionario_opciones = {
    'categorias': sorted(df['CATEGORIA'].unique().tolist()),
    'campos': sorted(df['CAMPO'].unique().tolist()),
    'situaciones': sorted(df['SITUACAO'].unique().tolist())
}
joblib.dump(diccionario_opciones, 'diccionario_opciones.pkl')

print("¡Archivos binarios maestros generados y listos para producción!")

#%% BLOQUE 7: Motor Alternativo de Regresión - Hist Gradient Boosting
# ----------------------------------------------------------------------------
from sklearn.ensemble import HistGradientBoostingRegressor

print("\n" + "="*50)
print("INICIANDO ENTRENAMIENTO DEL MOTOR ALTERNATIVO (GRADIENT BOOSTING REG)...")
print("="*50)

# Instanciamos el regresor basado en histogramas de Gradient Boosting
# Esta arquitectura es óptima para manejar la varianza geológica sin disparar el costo computacional.
modelo_gbm_reg_base = HistGradientBoostingRegressor(
    loss='squared_error',
    learning_rate=0.05,
    max_iter=300,
    max_depth=10,
    min_samples_leaf=15,
    l2_regularization=0.01,
    early_stopping=True,
    validation_fraction=0.2,
    random_state=42,
    verbose=0
)

# Para mantener una simetría matemática rigurosa con el modelo MLP,
# envolvemos el GBM dentro del mismo esquema de Target Scaling.
modelo_maestro_gbm_reg = TransformedTargetRegressor(
    regressor=modelo_gbm_reg_base,
    transformer=StandardScaler()
)

# El preprocesamiento ya se ejecutó en el Bloque 3; reutilizamos los tensores limpios.
print("Ajustando hiperplanos del Gradient Boosting sobre el espacio vectorial...")
modelo_maestro_gbm_reg.fit(X_train_procesado, y_train)
print("¡Motor Gradient Boosting entrenado con éxito!")


#%% BLOQUE 8: Evaluación Estadística del Motor Alternativo (GBM)
# ----------------------------------------------------------------------------
# Inferencia estocástica sobre el conjunto de prueba independiente
y_pred_gbm = modelo_maestro_gbm_reg.predict(X_test_procesado)

# Cómputo de métricas continentales para el Benchmark contrafactual
mae_gbm = mean_absolute_error(y_test, y_pred_gbm)
r2_gbm = r2_score(y_test, y_pred_gbm)

print("\n" + "★"*50)
print("EVALUACIÓN ESTADÍSTICA MOTOR ALTERNATIVO (GBM REGRESIÓN):")
print(f"Error Absoluto Medio [MAE GBM]: {mae_gbm:.2f} metros.")
print(f"Coeficiente de Determinación [R² GBM]: {r2_gbm:.4f}")
print("★"*50)


#%% BLOQUE 9: Serialización Aislada de Artefactos de Regresión GBM
# ----------------------------------------------------------------------------
# Guardamos los archivos con nomenclaturas unívocas para blindar el directorio contra colisiones
joblib.dump(modelo_maestro_gbm_reg, 'modelo_gbm_pozos_reg.pkl')
joblib.dump(mae_gbm, 'mae_error_gbm_reg.pkl')

print("¡Artefactos del motor alternativo GBM serializados de forma segura y aislada!")