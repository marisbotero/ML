import pandas as pd 
import numpy as np 

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


if __name__ == "__main__":
    
    dt_heart = pd.read_csv('./data/heart.csv')

    print(dt_heart['target'].describe())

    X = dt_heart.drop(['target'], axis=1)
    y = dt_heart['target']

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

    boost = GradientBoostingClassifier(n_estimators=50).fit(X_train, y_train)
    boost_pred = boost.predict(X_test)
    print("="*64)
    print(accuracy_score(boost_pred, y_test))

#implementacion_boosting