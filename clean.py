import pandas as pd
import os

EDAD_BINS = (0, 5, 11, 13, 17, 24, 28, 32, 38, 45, 52, 59, 66, 98, 160)
EDAD_LABELS = (
    '0-5', '6-11', '12-13', '14-17', '18-24', '25-28', '29-32', '33-38', '39-45', '46-52', '53-59', '60-66',
    'mayor-67', 'Sin dato'
)

PATH = os.path.dirname(os.path.abspath(__file__))
data = pd.read_excel(PATH+'/data/consolidado_2018_2019.xlsx')

data = data[data.pais == 'Colombia']
data.region = data.region.replace('San Antonio de Prado', 'Medellín')

data = data[~data.region.isin([9, 8, 3, 6])]
data = data.fillna('Sin dato')
data['rango_edad'] = pd.cut(data['edad'], EDAD_BINS, labels=EDAD_LABELS)
#data = data[data.region=='Sin dato']
data.drop('Unnamed: 0', axis=1, inplace=True)
data.to_excel(PATH+'/data/consolidado_2018_2019_clean.xlsx', index=False)

"""
obtener frecuencia de edades
data2= data.drop_duplicates('idu')
data2.edad.value_counts().to_excel('edades.xlsx')

"""
