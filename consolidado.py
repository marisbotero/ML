import pandas as pd
import os

PATH = os.path.dirname(os.path.abspath(__file__))
data2018 = pd.read_excel(PATH+'/data/master_med100p_2018.xlsx')
data2019 = pd.read_excel(PATH+'/data/Master_med100p_2019.xlsx')
data20192 = pd.read_excel(PATH+'/data/MED100p_2019.xlsx')

data2018 = data2018.rename(columns={'comuna': 'region'})
data2018['comuna'] = 'Sin dato'
data2018['barrio'] = 'Sin dato'
data2018['anio'] = 2018

data20192 = data20192[['barrio', 'idc']]

data2019 = pd.merge(data2019, data20192, how='left', on='idc')
data2019['anio'] = 2019

consolidado = pd.concat([data2018, data2019])
consolidado.to_excel(PATH+'/data/consolidado_2018_2019.xlsx')