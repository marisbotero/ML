import pandas as pd 
import sklearn
import matplotlib.pyplot as plt


from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    dt_heart = pd.read_csv('./data/heart.csv')
    print(dt_heart.head(5))

    dt_features = dt_heart.drop(['target'], axis=1)
    dt_target = dt_heart['target']

    df_features = StandardScaler().fit_transform(dt_features)
    
