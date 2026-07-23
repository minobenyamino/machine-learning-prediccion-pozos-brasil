# Motor Principal: Red Neuronal Multicapa (MLP)

#%% Importación de Librerías
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

print("Librerías cargadas. Iniciando arquitectura neuronal principal.")

#%% PASO 1: Carga de Dataset
# ----------------------------------------------------------------------------
print("PASO 1: Carga de dataset... Completa")
df = pd.read_csv('Pozos brasil 2.csv', sep=';', encoding='latin1', decimal=',')


#%% PASO 2: Preparación de Datos
# ----------------------------------------------------------------------------
print("PASO 2: Preparación de datos...")
# Definimos las variables operativas
columnas_numericas = ['LATITUDE_BASE_DD', 'LONGITUDE_BASE_DD', 'LAMINA_D_AGUA_M', 'COTA_ALTIMETRICA_M']
columnas_categoricas = ['BACIA', 'TERRA_MAR', 'TIPO', 'CATEGORIA', 'CAMPO', 'SITUACAO']
objetivo = 'PROFUNDIDADE_SONDADOR_M'

# Corrección de decimales y coerción de nulos en numéricas
for col in columnas_numericas + [objetivo]:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

df = df.dropna(subset=['LATITUDE_BASE_DD', 'LONGITUDE_BASE_DD', objetivo])

# Calibración Estadística: Amputamos el 1% de valores atípicos (Percentil 99)
limite_superior = df[objetivo].quantile(0.99)
df = df[(df[objetivo] >= 100) & (df[objetivo] <= limite_superior)]

df['LAMINA_D_AGUA_M'] = df['LAMINA_D_AGUA_M'].fillna(0.0)
df['COTA_ALTIMETRICA_M'] = df['COTA_ALTIMETRICA_M'].fillna(0.0)

def limpiar_y_traducir_datos(datos):
    """Manejo de Alta Cardinalidad, Traducción y Agrupación"""
    for col in columnas_categoricas:
        datos[col] = datos[col].astype(str).apply(lambda x: unidecode(x).lower().strip() if pd.notnull(x) and x != 'nan' else 'desconocido')
        
    datos['TERRA_MAR'] = datos['TERRA_MAR'].replace({'t': 'tierra', 'm': 'mar'})
    datos['TIPO'] = datos['TIPO'].replace({'explotatorio': 'desarrollo', 'exploratorio': 'exploratorio'})
    
    dict_cat = {'desenvolvimento': 'desarrollo', 'pioneiro': 'pionero', 'extensao': 'extension', 'especial': 'especial', 'injeaao': 'inyeccion', 'injecao': 'inyeccion'}
    datos['CATEGORIA'] = datos['CATEGORIA'].map(lambda x: dict_cat.get(x, 'otras_categorias'))
    
    cuencas_top = datos['BACIA'].value_counts().nlargest(15).index.tolist()
    datos['BACIA'] = datos['BACIA'].apply(lambda x: x if x in cuencas_top else 'otras_cuencas')
    
    conteos_campo = datos['CAMPO'].value_counts()
    campos_robustos = conteos_campo[conteos_campo >= 5].index.tolist()
    if 'desconocido' in campos_robustos: campos_robustos.remove('desconocido')
    datos['CAMPO'] = datos['CAMPO'].apply(lambda x: x if x in campos_robustos else 'otros_campos')
    
    def agrupar_situacion(s):
        if 'produt' in s or 'jazida' in s: return 'productores / activos'
        if 'abandon' in s or 'arras' in s: return 'abandonados / cerrados'
        return 'otras_operaciones'
        
    datos['SITUACAO'] = datos['SITUACAO'].apply(agrupar_situacion)
    return datos

df = limpiar_y_traducir_datos(df)

# Generación temprana del diccionario para la interfaz interactiva
diccionario_opciones = {
    'categorias': sorted(df['CATEGORIA'].unique().tolist()),
    'campos': sorted(df['CAMPO'].unique().tolist()),
    'situaciones': sorted(df['SITUACAO'].unique().tolist())
}


#%% PASO 3: División de Datos
# ----------------------------------------------------------------------------
print("PASO 3: División de datos...")
X = df[columnas_numericas + columnas_categoricas]
y = df[objetivo]

# Partición de tensores (80% Entrenamiento / 20% Prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#%% PASO 4: Construcción del Modelo
# ----------------------------------------------------------------------------
print("PASO 4: Construcción del modelo y Tuberías de Preprocesamiento...")
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='desconocido')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocesador = ColumnTransformer(
    transformers=[
        ('num', num_transformer, columnas_numericas),
        ('cat', cat_transformer, columnas_categoricas)
    ])

# Arquitectura Perceptrón Multicapa
red_neuronal_base = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64), 
    activation='relu',                 
    solver='adam',                     
    alpha=0.001, 
    max_iter=400,                      
    batch_size=128, 
    early_stopping=True,               
    validation_fraction=0.2,           
    n_iter_no_change=20,
    random_state=42,
    verbose=True                       
)

# Envoltura estocástica
modelo_maestro = TransformedTargetRegressor(
    regressor=red_neuronal_base,
    transformer=StandardScaler()
)


#%% PASO 5: Entrenamiento
# ----------------------------------------------------------------------------
print("PASO 5: Entrenamiento (Ajuste de pesos y Target Scaling)...")
# Ajuste espacial vectorial justo antes de alimentar la red
X_train_procesado = preprocesador.fit_transform(X_train)
X_test_procesado = preprocesador.transform(X_test)

modelo_maestro.fit(X_train_procesado, y_train)


#%% PASO 6: Evaluación
# ----------------------------------------------------------------------------
print("PASO 6: Evaluación estocástica del modelo principal...")
y_pred = modelo_maestro.predict(X_test_procesado)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n" + "★"*50)
print("EVALUACIÓN ESTADÍSTICA FINAL (RED NEURONAL):")
print(f"Error Absoluto Medio [MAE]: {mae:.2f} metros.")
print(f"Coeficiente de Determinación [R²]: {r2:.4f}")
print("★"*50)


#%% PASO 7: Predicción y Presentación (Serialización)
# ----------------------------------------------------------------------------
print("PASO 7: Serialización de artefactos para despliegue...")
joblib.dump(preprocesador, 'preprocesador_pozos.pkl')
joblib.dump(modelo_maestro, 'modelo_mlp_pozos.pkl')
joblib.dump(mae, 'mae_error.pkl')
joblib.dump(diccionario_opciones, 'diccionario_opciones.pkl')

print("¡Artefactos del motor principal serializados con éxito!")