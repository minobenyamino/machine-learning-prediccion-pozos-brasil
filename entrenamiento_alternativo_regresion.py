# Motor Alternativo: HistGradientBoostingRegressor (GBM)

#%% Importación de Librerías
# ----------------------------------------------------------------------------
import pandas as pd
import numpy as np
from unidecode import unidecode
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor

print("Librerías cargadas. Iniciando motor alternativo (GBM).")


#%% PASO 1: Carga de Dataset
# ----------------------------------------------------------------------------
print("PASO 1: Carga de dataset...")
df = pd.read_csv('Pozos brasil 2.csv', sep=';', encoding='latin1', decimal=',')


#%% PASO 2: Preparación de Datos
# ----------------------------------------------------------------------------
print("PASO 2: Preparación de datos...")
columnas_numericas = ['LATITUDE_BASE_DD', 'LONGITUDE_BASE_DD', 'LAMINA_D_AGUA_M', 'COTA_ALTIMETRICA_M']
columnas_categoricas = ['BACIA', 'TERRA_MAR', 'TIPO', 'CATEGORIA', 'CAMPO', 'SITUACAO']
objetivo = 'PROFUNDIDADE_SONDADOR_M'

for col in columnas_numericas + [objetivo]:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

df = df.dropna(subset=['LATITUDE_BASE_DD', 'LONGITUDE_BASE_DD', objetivo])

# Calibración Estadística
limite_superior = df[objetivo].quantile(0.99)
df = df[(df[objetivo] >= 100) & (df[objetivo] <= limite_superior)]

df['LAMINA_D_AGUA_M'] = df['LAMINA_D_AGUA_M'].fillna(0.0)
df['COTA_ALTIMETRICA_M'] = df['COTA_ALTIMETRICA_M'].fillna(0.0)

def limpiar_y_traducir_datos(datos):
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


#%% PASO 3: División de Datos
# ----------------------------------------------------------------------------
print("PASO 3: División de datos...")
X = df[columnas_numericas + columnas_categoricas]
y = df[objetivo]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#%% PASO 4: Construcción del Modelo
# ----------------------------------------------------------------------------
print("PASO 4: Construcción del modelo GBM y tuberías de preprocesamiento...")
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

modelo_maestro_gbm_reg = TransformedTargetRegressor(
    regressor=modelo_gbm_reg_base,
    transformer=StandardScaler()
)


#%% PASO 5: Entrenamiento
# ----------------------------------------------------------------------------
print("PASO 5: Entrenamiento del motor alternativo...")
X_train_procesado = preprocesador.fit_transform(X_train)
X_test_procesado = preprocesador.transform(X_test)

modelo_maestro_gbm_reg.fit(X_train_procesado, y_train)


#%% PASO 6: Evaluación
# ----------------------------------------------------------------------------
print("PASO 6: Evaluación del modelo alternativo...")
y_pred_gbm = modelo_maestro_gbm_reg.predict(X_test_procesado)

mae_gbm = mean_absolute_error(y_test, y_pred_gbm)
r2_gbm = r2_score(y_test, y_pred_gbm)

print("\n" + "★"*50)
print("EVALUACIÓN ESTADÍSTICA MOTOR ALTERNATIVO (GBM REGRESIÓN):")
print(f"Error Absoluto Medio [MAE GBM]: {mae_gbm:.2f} metros.")
print(f"Coeficiente de Determinación [R² GBM]: {r2_gbm:.4f}")
print("★"*50)


#%% PASO 7: Predicción y Presentación (Serialización)
# ----------------------------------------------------------------------------
print("PASO 7: Serialización aislada de artefactos GBM...")
joblib.dump(modelo_maestro_gbm_reg, 'modelo_gbm_pozos_reg.pkl')
joblib.dump(mae_gbm, 'mae_error_gbm_reg.pkl')

print("¡Artefactos del motor alternativo GBM serializados de forma segura y aislada!")