import pandas as pd 
import numpy as np 

#importar las librerias

#regresior que utiliza arboles de decision para tratar de predecir una variable continua
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import (cross_val_score, KFold)


if __name__ == "__main__":
    #datos de la felicidad
    dataset = pd.read_csv('./data/felicidad.csv')
    #partimos los datos
    X = dataset.drop(['country','score'], axis =1) #ajustarlo al eje de las columnas
    y = dataset['score'] #variable a predeccir
    
    #definición del modelo
    model = DecisionTreeRegressor()

    #vamos a utilizar cross validation , metodo: neg_mean_squared_error --> valor medio cuadrado
    score = cross_val_score(model, X, y, cv =3, scoring='neg_mean_squared_error')
    print(np.abs(np.mean(score)))
    #El score es un arreglo de errores negativos medios cuadrados (es decir, cuanto mas pequeño en valor absoluto, mejor se ajusta el modelo a los datos) como salida del coss_val_score, este resultado se da ya que el modelo fue separado cv veces (en este caso 5 al principio y luego 3) en set de datos de entrenamiento y prueba, en lo que se puede notar que particiones fueron mas satisfactorias.
    #Ahora al aplicar el promedio y el valor absoluto, puedes observar el error medio cuadrado promedio calculado a partir de las salidas score que evalúan la adaptación promedio del modelo a los datos.

    #Kfold
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    for train, test in kf.split(dataset):
        print(train)
        print(test)
