from fastapi import FastAPI, Request
from starlette.responses import HTMLResponse
import uvicorn
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder

from pydantic_types import Dropdown_str
from typing import List
from funciones import traduccion_inputs, predict_tam, predict_lon, predict_int, punto_dado, cal_distancias


# Debería de poderse hacer en plan bien, bonito, no tan horrible, pero en fin...
# Abriendo los modelos y los scalers:
with open ('../Models/scaler_X_tam.pkl', 'rb') as f1:
    scaler_X_tam = joblib.load(f1)
with open ('../Models/scaler_X_lon.pkl', 'rb') as f2:
    scaler_X_lon = joblib.load(f2)
with open ('../Models/scaler_X_int.pkl', 'rb') as f3:
    scaler_X_int = joblib.load(f3)
with open ('../Models/scaler_y_final.pkl', 'rb') as f4:
    scaler_y = joblib.load(f4)

with open ('../Models/modelo_svc_tam_final.pkl', 'rb') as f5:
    model_tam = joblib.load(f5)
with open ('../Models/modelo_rfr_lon_final.pkl', 'rb') as f6:
    model_lon = joblib.load(f6)
with open ('../Models/modelo_rn_int_1.pkl', 'rb') as f7:
    model_int = joblib.load(f7)

df_distrito = pd.read_csv('../Datasets/info_distritos_madrid.csv')
df_razas = pd.read_csv('../Datasets/razas-perretes.csv')


app = FastAPI(title='Furrends')



## Inputs:
dropdown_options_distrito = ['Fuencarral-El Pardo','Moncloa-Aravaca','Latina','Carabanchel','Usera','Villaverde','Puente de Vallecas','Villa de Vallecas','Moratalaz','Vicálvaro','Ciudad Lineal','San Blas-Canillejas','Hortaleza','Barajas','Tetuán','Chamartín','Chamberí','Centro','Retiro','Arganzuela','Salamanca']
dropdown_options_tricks = ['Sí', 'No']
dropdown_options_casa = ['Apartamento pequeño', 'Apartamento mediano', 'Casa pequeña', 'Casa con jardín']
dropdown_options_actfis = ['No', 'Rara vez', 'Un par de días a la semana', 'Siempre que puedo']
selected_values_df = pd.DataFrame(columns=["Distrito", "Tricks", "Casa", "ActividadFisica"])

@app.post("/process_dropdown/")
async def process_dropdown(request: Request, dropdown_input1: Dropdown_str, dropdown_input2: Dropdown_str, dropdown_input3: Dropdown_str, dropdown_input4: Dropdown_str):
    global selected_values_df
    selected_option1 = dropdown_input1
    selected_option2 = dropdown_input2
    selected_option3 = dropdown_input3
    selected_option4 = dropdown_input4
    
    # Formando de los inputs un df
    selected_values_df = selected_values_df.append({"Distrito": selected_option1, "Tricks": selected_option2, "Casa":selected_option3, "ActividadFisica":selected_option4}, ignore_index=True)
    df_calculus = traduccion_inputs(df=selected_values_df, df_distrito=df_distrito)
    
    ## Predicciones:
    pred_tam= predict_tam(df=df_calculus, scaler=scaler_X_tam, model=model_tam)
    pred_lon = predict_lon(df=df_calculus, scaler=scaler_X_lon, model=model_lon, tam=pred_tam)
    pred_int = predict_int(df=df_calculus, scaler=scaler_X_int, model=model_int, tam=pred_tam, lon=pred_lon)
    
    ## Cálculo distancias:
    punto = punto_dado(df=df_calculus, tam=pred_tam, lon=pred_lon, int=pred_int, scaler=scaler_y)
    df_dist = cal_distancias(df=df_razas, punto_dado=punto)
    
    ## Dibujando gráfica:
    plt.figure(figsize=(24,20))
    df_sorted = df_dist[['Dog breed', 'Compatibilidad (%)']].round(2).head(3)
    df_sorted.sort_values(by='Compatibilidad (%)', ascending=True).plot(kind='barh', x='Dog breed', y='Compatibilidad (%)', colormap='viridis')
    plt.legend().remove()
    plt.xlabel(xlabel='Compatibilidad', fontweight='bold', labelpad=20)
    plt.ylabel(ylabel='Top 3 Razas más compatibles', fontweight='bold', labelpad=20)
    plt.tick_params(axis='y', which='major', labelleft=False, left=False, labelright=True, pad=10)
    
    ## Guardando gráfica:
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png')
    img_bytes.seek(0)
    
    # Al parecer este paso es necesario para traducirlo a html
    img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
    
    return HTMLResponse(content=f"<h1>Selected Option 1: {selected_option1}</h1><h1>Selected Option 2: {selected_option2}</h1><h1>Selected Option 3: {selected_option3}</h1><h1>Selected Option 4: {selected_option4}</h1><img src='data:image/png;base64,{img_base64}'><br><a href='/'>Go back to select another option</a>")

@app.get("/")
async def home():
    dropdown_html1 = ""
    for option in dropdown_options_distrito:
        dropdown_html1 += f"<option value='{option}'>{option}</option>"
    
    dropdown_html2 = ""
    for option in dropdown_options_tricks:
        dropdown_html2 += f"<option value='{option}'>{option}</option>"
    
    dropdown_html3 = ""
    for option in dropdown_options_casa:
        dropdown_html3 += f"<option value='{option}'>{option}</option>"
        
    dropdown_html4 = ""
    for option in dropdown_options_actfis:
        dropdown_html4 += f"<option value='{option}'>{option}</option>"
    
    return HTMLResponse(content=f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Furrends Elección</title>
    </head>
    <body>
        <h1>Elige:</h1>
        <form action="/process_dropdown/" method="post">
            <div style="margin-bottom: 10px;">
                <label for="dropdown1">¿En qué distrito de Madrid vives?</label>
                <select name="dropdown_input1" id="dropdown1">
                    {dropdown_html1}
                </select>
            </div>
            <div style="margin-bottom: 10px;">
                <label for="dropdown2">¿Quieres tu futuro perro haga trucos?</label>
                <select name="dropdown_input2" id="dropdown2">
                    {dropdown_html2}
                </select>
            </div>
            <div style="margin-bottom: 10px;">
                <label for="dropdown2">¿En qué tipo de casa vives?</label>
                <select name="dropdown_input3" id="dropdown3">
                    {dropdown_html3}
                </select>
            </div>
            <div style="margin-bottom: 10px;">
                <label for="dropdown2">¿Dirías que haces mucho ejercicio?</label>
                <select name="dropdown_input4" id="dropdown4">
                    {dropdown_html4}
                </select>
            </div>
            <button type="submit">Submit</button>
        </form>
    </body>
    </html>
    """)



if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)