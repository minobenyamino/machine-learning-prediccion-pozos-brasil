#%% BLOQUE 1: Importación de Librerías y Módulos Estadísticos
# ----------------------------------------------------------------------------
import pandas as pd
import numpy as np
from unidecode import unidecode
import joblib

# Módulos de Scikit-Learn
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold


print("Iniciando topología neuronal para CLASIFICACIÓN OPERACIONAL con Optimización de Hiperparámetros...")

#%% BLOQUE 2: Ingesta, Limpieza, Traducción y Reducción de Cardinalidad
# ----------------------------------------------------------------------------
df = pd.read_csv('Pozos brasil 2.csv', sep=';', encoding='latin1', decimal=',')

columnas_numericas = ['LATITUDE_BASE_DD', 'LONGITUDE_BASE_DD', 'LAMINA_D_AGUA_M', 'COTA_ALTIMETRICA_M', 'PROFUNDIDADE_SONDADOR_M']
columnas_categoricas_X = ['BACIA', 'TERRA_MAR', 'TIPO', 'CATEGORIA', 'CAMPO']
objetivo = 'SITUACAO'

print("Limpiando ruido estadístico y caracteres especiales...")
for col in columnas_numericas:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

# Blindaje: Descartamos las filas que no tengan el Eje Z histórico
df = df.dropna(subset=[objetivo, 'PROFUNDIDADE_SONDADOR_M'])

def limpiar_y_traducir_datos_clasificacion(datos):
    """
    Traducción al español y Reducción de Cardinalidad.
    Se agrupan campos raros para evitar sobreajuste y mantener la interfaz limpia.
    """
    for col in columnas_categoricas_X + [objetivo]:
        datos[col] = datos[col].astype(str).apply(lambda x: unidecode(x).lower().strip() if pd.notnull(x) and x != 'nan' else 'desconocido')
        
    datos['TERRA_MAR'] = datos['TERRA_MAR'].replace({'t': 'tierra', 'm': 'mar'})
    datos['TIPO'] = datos['TIPO'].replace({'explotatorio': 'desarrollo', 'exploratorio': 'exploratorio'})
    
    dict_cat = {'desenvolvimento': 'desarrollo', 'pioneiro': 'pionero', 'extensao': 'extension', 'especial': 'especial', 'injeaao': 'inyeccion', 'injecao': 'inyeccion'}
    datos['CATEGORIA'] = datos['CATEGORIA'].map(lambda x: dict_cat.get(x, 'otras_categorias'))
    
    # Sincronización topológica: Top 15 cuencas
    cuencas_top = datos['BACIA'].value_counts().nlargest(15).index.tolist()
    datos['BACIA'] = datos['BACIA'].apply(lambda x: x if x in cuencas_top else 'otras_cuencas')
    
    # Sincronización topológica: Campos robustos (>= 5 pozos)
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
        
    datos[objetivo] = datos[objetivo].apply(agrupar_situacion)
    return datos

print("Aplicando reducción de cardinalidad y traducción al español...")
df = limpiar_y_traducir_datos_clasificacion(df)

#%% BLOQUE 3: Partición Estratificada (Preservación de la Varianza Natural)
# ----------------------------------------------------------------------------

print("Mapeando el 100% del espacio vectorial geológico...")

# Definimos los tensores usando la totalidad de los registros limpios de la ANP
X = df[columnas_numericas + columnas_categoricas_X]
y = df[objetivo]

# Partición (80/20) estratificada: Garantiza que el set de prueba tenga la 
# misma proporción natural de clases que el set de entrenamiento.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Dimensión del tensor de entrenamiento puro: {X_train.shape[0]} filas.")
#%% BLOQUE 4: Preprocesamiento con Soporte Categórico Nativo
# ----------------------------------------------------------------------------

print("Configurando OrdinalEncoder para activar soporte nativo en el Gradient Boosting...")

# Al evitar el OneHotEncoder, impedimos que la dimensionalidad estalle. 
# Los campos y cuencas se procesarán como nodos enteros puros.
preprocesador_clasificacion = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), columnas_numericas),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), columnas_categoricas_X)
    ]
)

#%% BLOQUE 5: Construcción de Tuberías (Pipelines) de Preprocesamiento
# ----------------------------------------------------------------------------
transformador_numerico = Pipeline(steps=[
    ('imputador', SimpleImputer(strategy='median')),
    ('escalador', RobustScaler())
])

transformador_categorico = Pipeline(steps=[
    ('imputador', SimpleImputer(strategy='constant', fill_value='desconocido')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# CRÍTICO: remainder='drop' evita el desajuste de dimensiones (Shape Mismatch) en inferencia web.
preprocesador_clasificacion = ColumnTransformer(transformers=[
    ('num', transformador_numerico, columnas_numericas),
    ('cat', transformador_categorico, columnas_categoricas_X)
], remainder='drop')

#%% BLOQUE 6: Arquitectura de Ensamblaje Estocástico y GridSearchCV
# ----------------------------------------------------------------------------
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Índices exactos de las columnas categóricas tras el ColumnTransformer
indices_categoricos = [5, 6, 7, 8, 9]

modelo_ml_gb = HistGradientBoostingClassifier(
    categorical_features=indices_categoricos,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42
)

pipeline_base = Pipeline(steps=[
    ('preprocesamiento', preprocesador_clasificacion),
    ('modelo_ml', modelo_ml_gb) 
])

# Malla paramétrica balanceada: protege contra el ruido estadístico
espacio_parametros = {
    'modelo_ml__learning_rate': [0.05, 0.1],           
    'modelo_ml__max_iter': [300, 500],            
    'modelo_ml__max_depth': [25, None],          
    'modelo_ml__l2_regularization': [0.01, 0.1],   
    'modelo_ml__min_samples_leaf': [5, 15]       
}

val_cruzada = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

print("\nIniciando GridSearchCV con Acoplamiento Secuencial y Soporte Nativo...")
optimizador_grid = GridSearchCV(
    estimator=pipeline_base, param_grid=espacio_parametros,
    cv=val_cruzada, scoring='accuracy', n_jobs=-1, verbose=2
)

optimizador_grid.fit(X_train, y_train)

#%% BLOQUE 7: Evaluación Multivariante y Extracción del Modelo Óptimo
# ----------------------------------------------------------------------------
print("\n" + "="*50)
print("¡OPTIMIZACIÓN FINALIZADA CON ÉXITO!")
print("Mejor topología matemática encontrada:")
print(optimizador_grid.best_params_)
print("="*50)

# Extraemos el mejor Pipeline ya entrenado
pipeline_maestro = optimizador_grid.best_estimator_

print("\nExtrayendo predicciones sobre el conjunto de prueba (Test Set)...")
y_pred = pipeline_maestro.predict(X_test)

exactitud = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
exhaustividad = recall_score(y_test, y_pred, average='weighted', zero_division=0)
matriz_conf = confusion_matrix(y_test, y_pred)

print("\n" + "★"*50)
print("EVALUACIÓN ESTADÍSTICA FINAL (CLASIFICACIÓN OPERACIONAL):")
print(f"Exactitud (Accuracy):      {exactitud:.4f}")
print(f"Precisión (Precision):     {precision:.4f}")
print(f"Exhaustividad (Recall):    {exhaustividad:.4f}")
print("★"*50)

#%% BLOQUE 8: Serialización y Exportación de Artefactos Tecnológicos
# ----------------------------------------------------------------------------

# Extraemos dinámicamente el array de clases directamente del codificador del modelo
nombres_clases = pipeline_maestro.classes_

# 1. Serialización del Pipeline (El "Cerebro")
joblib.dump(pipeline_maestro, 'pipeline_clasificacion_pozos.pkl')

# 2. Serialización del Diccionario de Métricas (La "Memoria" para la App Web)
diccionario_metricas = {
    'clases': nombres_clases,
    'matriz_confusion': matriz_conf,
    'exactitud': exactitud,
    'precision': precision,
    'exhaustividad': exhaustividad
}
joblib.dump(diccionario_metricas, 'metricas_clasificacion.pkl')

print("¡Artefactos óptimos serializados correctamente para su despliegue web!")

#%% BLOQUE 9: Motor Alternativo de Clasificación - Perceptrón Multicapa (MLP)
# ----------------------------------------------------------------------------

print("\n" + "="*50)
print("INICIANDO ENTRENAMIENTO DEL MOTOR ALTERNATIVO (RED NEURONAL CLF)...")
print("="*50)

# Corrección de Arquitectura: early_stopping=False para evitar el bug 
# de np.isnan() de Scikit-Learn sobre matrices de etiquetas categóricas puras (strings).
modelo_mlp_clf_base = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    alpha=0.001,
    max_iter=400,
    early_stopping=False, # <--- COMPUERTA MODIFICADA
    random_state=42,
    verbose=False
)

# Empaquetamos la topología completa para prevenir la fuga de datos en inferencia
pipeline_mlp_clf = Pipeline(steps=[
    ('preprocesamiento', preprocesador_clasificacion),
    ('modelo_mlp', modelo_mlp_clf_base)
])

print("Propagando tensores a través de las capas ocultas y ajustando pesos logísticos...")
pipeline_mlp_clf.fit(X_train, y_train)
print("¡Red Neuronal de clasificación entrenada con éxito!")


#%% BLOQUE 10: Evaluación Estadística del Motor Alternativo (MLP)
# ----------------------------------------------------------------------------
# Inferencia sobre la matriz ortogonal de prueba (Test Set)
y_pred_mlp_clf = pipeline_mlp_clf.predict(X_test)

# Extraemos una métrica directa de eficiencia para el benchmark de la interfaz web
exactitud_mlp = accuracy_score(y_test, y_pred_mlp_clf)

print("\n" + "★"*50)
print("EVALUACIÓN ESTADÍSTICA MOTOR ALTERNATIVO (MLP CLASIFICACIÓN):")
print(f"Exactitud [Accuracy MLP]: {exactitud_mlp:.4f}")
print("★"*50)


#%% BLOQUE 11: Serialización Aislada de Artefactos de Clasificación MLP
# ----------------------------------------------------------------------------
# Guardamos los binarios con nomenclaturas unívocas
joblib.dump(pipeline_mlp_clf, 'pipeline_mlp_clasificacion_pozos.pkl')
joblib.dump(exactitud_mlp, 'exactitud_mlp_clf.pkl')

print("¡Artefactos del motor alternativo MLP serializados de forma segura y aislada!")