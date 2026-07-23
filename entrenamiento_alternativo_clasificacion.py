# Motor Alternativo: Perceptrón Multicapa (MLP Classifier)

#%% Importación de Librerías y Módulos Estadísticos
# ----------------------------------------------------------------------------
import pandas as pd
import numpy as np
from unidecode import unidecode
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

print("="*65)
print("⚙️ INICIANDO MOTOR ALTERNATIVO: CLASIFICACIÓN NEURONAL (MLP)")
print("="*65)

#%% PASO 1: Carga de Dataset
# ----------------------------------------------------------------------------
df = pd.read_csv('Pozos brasil 2.csv', sep=';', encoding='latin1', decimal=',')
print(f"         ➜ Dataset cargado exitosamente. Dimensión inicial: {df.shape[0]} registros.")

#%% PASO 2: Preparación de Datos
# ----------------------------------------------------------------------------
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
        return 'otras_operaciones'
        
    datos[objetivo] = datos[objetivo].apply(agrupar_situacion)
    return datos

df = limpiar_y_traducir_datos_clasificacion(df)
print("         ➜ Alta cardinalidad reducida, nulos purgados y clases operacionales agrupadas.")

#%% PASO 3: División de Datos
# ----------------------------------------------------------------------------
X = df[columnas_numericas + columnas_categoricas_X]
y = df[objetivo]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"         ➜ Varianza poblacional asegurada. Tensor de entrenamiento listo ({X_train.shape[0]} filas).")

#%% PASO 4: Construcción del Modelo
# ----------------------------------------------------------------------------
transformador_numerico = Pipeline(steps=[
    ('imputador', SimpleImputer(strategy='median')),
    ('escalador', RobustScaler())
])

transformador_categorico = Pipeline(steps=[
    ('imputador', SimpleImputer(strategy='constant', fill_value='desconocido')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocesador_clasificacion = ColumnTransformer(transformers=[
    ('num', transformador_numerico, columnas_numericas),
    ('cat', transformador_categorico, columnas_categoricas_X)
], remainder='drop')

# Red Neuronal con compuerta modificada
modelo_mlp_clf_base = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    alpha=0.001,
    max_iter=400,
    early_stopping=False, # Modificación crucial
    random_state=42,
    verbose=False
)

pipeline_mlp_clf = Pipeline(steps=[
    ('preprocesamiento', preprocesador_clasificacion),
    ('modelo_mlp', modelo_mlp_clf_base)
])
print("         ➜ Red multicapa (128, 64) definida.")
print("         ➜ Early Stopping desactivado para evitar colisiones internas con tensores de texto.")

#%% PASO 5: Entrenamiento
# ----------------------------------------------------------------------------
print("\n[PASO 5] 🧠 Entrenando Perceptrón Multicapa (MLP)...")
print("         ➜ Propagando datos a través de las capas ocultas y ajustando pesos logísticos...")
pipeline_mlp_clf.fit(X_train, y_train)
print("         ➜ ¡Convergencia alcanzada! Red Neuronal entrenada con éxito.")

#%% PASO 6: Evaluación
# ----------------------------------------------------------------------------
print("\n[PASO 6] 📊 Evaluando rendimiento del motor alternativo...")
y_pred_mlp_clf = pipeline_mlp_clf.predict(X_test)
exactitud_mlp = accuracy_score(y_test, y_pred_mlp_clf)

print("\n" + "★"*50)
print("EVALUACIÓN ESTADÍSTICA MOTOR ALTERNATIVO (MLP):")
print(f"Exactitud general del algoritmo [Accuracy]: {exactitud_mlp:.4f}")
print("★"*50)

#%% PASO 7: Predicción y Presentación (Serialización)
# ----------------------------------------------------------------------------
print("\n[PASO 7] 💾 Serializando artefactos tecnológicos...")
joblib.dump(pipeline_mlp_clf, 'pipeline_mlp_clasificacion_pozos.pkl')
joblib.dump(exactitud_mlp, 'exactitud_mlp_clf.pkl')
print("         ➜ ¡Archivos binarios .pkl generados de forma aislada y listos para el backend!")
print("="*65 + "\n")