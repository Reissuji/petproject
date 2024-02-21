from fastapi import FastAPI, Form, Request
from starlette.responses import HTMLResponse
import uvicorn
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder

from pydantic_types import Dropdown_str
from typing import List




def traduccion_inputs(df=pd.DataFrame, df_distrito=pd.DataFrame):
    ## Parte del distrito
    df_preguntas = pd.merge(df, df_distrito, on='Distrito', how='left')
    df_preguntas = df_preguntas.drop(['Distrito', 'Renta media/hogar (€)', 'Perros censados', ' Ratio perros/m²'], axis=1)
    
    ## Parte de tricks
    if df['Tricks'] == 'Sí':
        df_preguntas.at[0,'tricks_yes'] = 1
    else:
        df_preguntas.at[0,'tricks_yes'] = 0
    
    ## Parte de casa
    if df["Casa"] == 'Apartamento pequeño':
        df_preguntas.at[0,'house_type_enc'] = 0
    elif df["Casa"] == 'Apartamento mediano':
        df_preguntas.at[0,'house_type_enc'] = 1
    elif df["Casa"] == 'Casa pequeña':
        df_preguntas.at[0,'house_type_enc'] = 2
    else:
        df_preguntas.at[0,'house_type_enc'] = 3
    
    ## Parte Actividad Física:
    if df["ActividadFisica"] == 'No':
        df_preguntas.at[0,'house_type_enc'] = 0
    elif df["ActividadFisica"] == 'Rara vez':
        df_preguntas.at[0,'house_type_enc'] = 1
    elif df["ActividadFisica"] == 'Un par de días a la semana':
        df_preguntas.at[0,'house_type_enc'] = 2
    else:
        df_preguntas.at[0,'house_type_enc'] = 3
    return df_preguntas

## Las de predicciones:
# Tamaño
def predict_tam(df=pd.DataFrame, scaler=None, model=None):
    orden_tam = ['Renta media/pers (€)', 'Áreas caninas (m²)', 'tricks_yes','house_type_enc','owners_physical_activity_enc']
    df_pred_tam = scaler.transform(df[orden_tam])
    pred_tam = model.predict(df_pred_tam)
    return float(pred_tam)

#Longevidad
def predict_lon(df=pd.DataFrame, tam=float, scaler=None, model=None):
    orden_lon = ['size_category_enc', 'Renta media/pers (€)', 'Áreas caninas (m²)','house_type_enc','owners_physical_activity_enc']
    df.at[0, 'size_category_enc'] = tam
    df_pred_lon = scaler.transform(df[orden_lon])
    pred_lon = model.predict(df_pred_lon)
    return float(pred_lon)

#Inteligencia
def predict_int(df=pd.DataFrame, tam=float, lon=float, scaler=None, model=None):
    orden_int = ['size_category_enc', 'longevity', 'Renta media/pers (€)', 'Áreas caninas (m²)', 'tricks_yes','house_type_enc','owners_physical_activity_enc']
    df.at[0, 'size_category_enc'] = tam
    df.at[0, 'longevity'] = lon
    df_pred_int = scaler.transform(df[orden_int])
    pred_int = model.predict(df_pred_int)
    pred_int = np.argmax(pred_int, axis=1)
    return float(pred_int)


## Distancias:
# Punto dado:
def punto_dado(df=pd.DataFrame, tam=float, lon=float, int=float, scaler=None):
    # Para formar de nuevas el df:
    df.at[0, 'size_category_enc'] = tam
    df.at[0, 'longevity'] = lon
    df.at[0, 'int_cat_enc_fixed'] = int
    
    df['size_cat_enc', 'lon_enc', 'int_cat_enc'] = scaler.transform(df[['size_category_enc', 'longevity', 'int_cat_enc_fixed']])
    
    ## Los componentes de nuestro punto dado:
    pto_tam = df['size_cat_enc']
    pto_lon = df['lon_enc']
    pto_int = df['int_cat_enc']
    
    punto_dado = np.array([pto_tam*3, pto_lon*2, pto_int])
    return punto_dado

#Calculo distancias para df de perros:
def cal_distancias(df=pd.DataFrame, punto_dado=np.array):
    # Calcula la distancia euclidiana entre el punto dado y todos los puntos en el DataFrame
    df['size_cat_enc'] = df['size_cat_enc']*3
    df['lon_enc'] = df['lon_enc']*2

    # Ajustando array para que no pete al comparar con df
    punto_dado_broadcasted = punto_dado.reshape(1, -1)

    # Calculando distancias en todo el df
    df['distancia'] = np.linalg.norm(df[['size_cat_enc', 'lon_enc', 'int_cat_enc']].values - punto_dado_broadcasted, axis=1)
    
    sorted_df = df.sort_values(by='distancia', ascending=True)
    sorted_df['Compatibilidad (%)'] = 100-(sorted_df['distancia']*100)/max(sorted_df['distancia'])
    return sorted_df