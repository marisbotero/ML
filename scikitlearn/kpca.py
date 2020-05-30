import pandas as pd 
import sklearn
import matplotlib.pyplot as plt



from sklearn.decomposition import KernelPCA

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Dataset de pacientes con riesgo de una enfermadad cardiacas
#utilizando ciertas variables de los pacientes, intentar hacer una clasificacion binaria
#si el paciente tiene o no una enfermedad cardiaca

if __name__ == "__main__":
    dt_heart = pd.read_csv('./data/heart.csv')
    print(dt_heart.head(5))
    
    dt_features = dt_heart.drop(['target'], axis=1)
    dt_target = dt_heart['target']

#PCA normalizar los datos

    dt_features = StandardScaler().fit_transform(dt_features)

    #Distribución de variables:

    X_train, X_test, y_train, y_test = train_test_split(dt_features, dt_target, test_size=0.3, random_state=42)


    kpca = KernelPCA(n_components=4, kernel='poly' )
    kpca.fit(X_train)

    dt_train = kpca.transform(X_train)
    dt_test = kpca.transform(X_test)

    logistic = LogisticRegression(solver='lbfgs')

    logistic.fit(dt_train, y_train)
    print("SCORE KPCA: ", logistic.score(dt_test, y_test)) 