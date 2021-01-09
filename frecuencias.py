import pandas as pd
from collections import Counter
import ast
import os

PATH = os.path.dirname(os.path.abspath(__file__))

data = pd.read_excel(PATH+'/data/data_consolidado_tokens.xlsx')

data.token = data.token.apply(lambda x: ast.literal_eval(x))
unicos = list()
for item in data.token:
    for value in item:
        unicos.append(value)
counts = Counter(unicos)
frecuencia = pd.DataFrame.from_dict(counts, orient='index')
frecuencia.sort_values(0, ascending=False, inplace=True)
frecuencia.to_excel(PATH+'/data/frecuencia_palabras.xlsx')