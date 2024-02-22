import pandas as pd
import numpy as np
dropdown_options_distrito = ['Fuencarral-El Pardo','Moncloa-Aravaca','Latina','Carabanchel','Usera','Villaverde','Puente de Vallecas','Villa de Vallecas','Moratalaz','Vicálvaro','Ciudad Lineal','San Blas-Canillejas','Hortaleza','Barajas','Tetuán','Chamartín','Chamberí','Centro','Retiro','Arganzuela','Salamanca']
dropdown_options_tricks = ['Sí', 'No']
dropdown_options_casa = ['Apartamento pequeño', 'Apartamento mediano', 'Casa pequeña', 'Casa con jardín']
dropdown_options_actfis = ['No', 'Rara vez', 'Un par de días a la semana', 'Siempre que puedo']

df_distritos = pd.DataFrame({'Distritos': dropdown_options_distrito})
df_tricks = pd.DataFrame({'Trucos': dropdown_options_tricks})
df_casa_actfis = pd.DataFrame({'Tipo de casa': dropdown_options_casa, 'Actividad Física':dropdown_options_actfis})

df_distritos.to_csv('../Datasets/Inputs/input_distritos.csv')
df_tricks.to_csv('../Datasets/Inputs/input_tricks.csv')
df_casa_actfis.to_csv('../Datasets/Inputs/input_casa_actfis.csv')