import pandas as pd
import sklearn 

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


if __name__ == "__main__":
    dataset = pd.read_csv('./data/felicidad.csv')
    print(dataset.describe())

    X = dataset[['gdp','family', 'lifexp', 'freedom','corruption','generosity', 'dystopia']]
    y = dataset[['score']]

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25)

    #Defición de los regresores

    modelLinear = LinearRegression().fit(X_train, y_train)
    #calcular la prediccoón del modelo de los test
    y_predict_linear =  modelLinear.predict(X_test)

    #Regresión Lasso
    #alpha nos permite configurar la penalización de nuestros features
    modelLasso = Lasso(alpha=0.02).fit(X_train, y_train)
    #Realizar una predicción para ver si mejora en comparación del modelo lineal
    y_predict_lasso = modelLasso.predict(X_test)

    #Ridge
    modelRidge = Ridge(alpha=1).fit(X_train, y_train)
    y_predict_ridge = modelRidge.predict(X_test)


    # calcular la perdida con el error medio cuadratico 
    
    linear_loss = mean_squared_error(y_test, y_predict_linear)
    print("Linear Loss:", linear_loss)

    lasso_loss = mean_squared_error(y_test, y_predict_lasso)
    print("Lasso Loss: ", lasso_loss)

    ridge_loss = mean_squared_error(y_test, y_predict_ridge)
    print("Ridge Loss: ", ridge_loss)

    print("="*32)
    print("Coef LASSO")
    print(modelLasso.coef_)
    
    print("="*32)
    print("Coef RIDGE")
    print(modelRidge.coef_)


    print(X.shape)
    print(y.shape)

    