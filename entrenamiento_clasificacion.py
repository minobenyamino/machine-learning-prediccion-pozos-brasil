# Motor Principal: HistGradientBoosting + SMOTETomek

#%% Importación de Librerías y Módulos Estadísticos
# ----------------------------------------------------------------------------
import pandas as pd
import numpy as np
from unidecode import unidecode
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTETomek

print("Librerías cargadas. Iniciando motor principal de Clasificación Operacional (HGB + SMOTETomek).")

#%% PASO 1: Carga de Dataset
# ----------------------------------------------------------------------------
print("PASO 1: Carga de dataset...")
df = pd.read_csv('Pozos brasil 2.csv', sep=';', encoding='latin1', decimal=',')

#%% PASO 2: Preparación de Datos (3 Macro-Clases)
# ----------------------------------------------------------------------------
print("PASO 2: Preparación de datos (Limpieza y Traducción)...")
columnas_numericas = ['LATITUDE_BASE_DD', 'LONGITUDE_BASE_DD', 'LAMINA_D_AGUA_M', 'COTA_ALTIMETRICA_M', 'PROFUNDIDADE_SONDADOR_M']
columnas_categoricas_X = ['BACIA', 'TERRA_MAR', 'TIPO', 'CATEGORIA', 'CAMPO']
objetivo = 'SITUACAO'

for col in columnas_numericas:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

df = df.dropna(subset=[objetivo, 'PROFUNDIDADE_SONDADOR_M'])

def limpiar_y_traducir_datos_clasificacion(datos):
    for col in columnas_categoricas_X + [objetivo]:
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
        return 'otras_operaciones' # <-- Agrupación definitiva
        
    datos[objetivo] = datos[objetivo].apply(agrupar_situacion)
    return datos

df = limpiar_y_traducir_datos_clasificacion(df)

#%% PASO 3: División de Datos
# ----------------------------------------------------------------------------
print("PASO 3: División de datos con partición estratificada...")
X = df[columnas_numericas + columnas_categoricas_X]
y = df[objetivo]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#%% PASO 4: Construcción del Modelo Híbrido
# ----------------------------------------------------------------------------
print("PASO 4: Construcción del modelo (Imblearn Pipeline)...")
transformador_num = SklearnPipeline(steps=[
    ('imputador', SimpleImputer(strategy='median')),
    ('escalador', RobustScaler())
])

transformador_cat = SklearnPipeline(steps=[
    ('imputador', SimpleImputer(strategy='constant', fill_value='desconocido')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocesador = ColumnTransformer(transformers=[
    ('num', transformador_num, columnas_numericas),
    ('cat', transformador_cat, columnas_categoricas_X)
], remainder='drop')

pipeline_maestro = ImbPipeline(steps=[
    ('preprocesamiento', preprocesador),
    ('smote_tomek', SMOTETomek(random_state=42)),
    ('modelo_gbm', HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_iter=500,
        max_depth=15,
        min_samples_leaf=10,
        l2_regularization=0.1,
        early_stopping=True,
        random_state=42
    ))
])

#%% PASO 5: Entrenamiento
# ----------------------------------------------------------------------------
print("PASO 5: Entrenamiento (Acoplamiento Secuencial con SMOTETomek)...")
pipeline_maestro.fit(X_train, y_train)

#%% PASO 6: Evaluación
# ----------------------------------------------------------------------------
print("PASO 6: Evaluación multivariante y extracción del modelo óptimo...")
y_pred = pipeline_maestro.predict(X_test)

exactitud = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
exhaustividad = recall_score(y_test, y_pred, average='weighted', zero_division=0)

# CÁLCULO DE LA MATRIZ NORMALIZADA DIRECTO A MEMORIA
matriz_conf_norm = confusion_matrix(y_test, y_pred, normalize='true')

print("\n" + "★"*50)
print("EVALUACIÓN ESTADÍSTICA FINAL (HGB + SMOTETomek):")
print(f"Exactitud (Accuracy):      {exactitud:.4f}")
print(f"Precisión (Precision):     {precision:.4f}")
print(f"Exhaustividad (Recall):    {exhaustividad:.4f}")
print("★"*50)

#%% PASO 7: Predicción y Presentación (Serialización)
# ----------------------------------------------------------------------------
print("PASO 7: Serialización de artefactos para despliegue...")
nombres_clases = pipeline_maestro.classes_

diccionario_metricas = {
    'clases': nombres_clases,
    'matriz_confusion': matriz_conf_norm, # Guardamos la matriz en formato porcentual (0 a 1)
    'exactitud': exactitud,
    'precision': precision,
    'exhaustividad': exhaustividad
}

joblib.dump(pipeline_maestro, 'pipeline_clasificacion_pozos.pkl')
joblib.dump(diccionario_metricas, 'metricas_clasificacion.pkl')

print("¡Artefactos óptimos del HGB serializados correctamente!")