# Código de Entrenamiento - Modelo de Riesgo de Default en un empresa de Telecomunicaciones
############################################################################

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
import pickle
import os


# Cargar la tabla transformada
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/processed', filename))
    df = df.iloc[:,1:]
    X_train = df.drop(['Churn','customerID'],axis=1)
    y_train = df[['Churn']]
    print(filename, ' cargado correctamente')
    
    # Entrenamos el modelo con toda la muestra
    ros = RandomOverSampler(random_state=2022)

    # fit predictor and target variablex_ros, 
    x_ros, y_ros = ros.fit_resample(X_train, y_train)

    # Entrenamos el modelo con toda la muestra
    rf = RandomForestClassifier(random_state=2022)
    rf.fit(x_ros, y_ros) # Entrenando un algoritmo
    print('Modelo entrenado')
    

    # Guardamos el modelo entrenado para usarlo en produccion
    package = '../models/best_model.pkl'
    pickle.dump(rf, open(package, 'wb'))
    print('Modelo exportado correctamente en la carpeta models')


# Entrenamiento completo
def main():
    read_file_csv('Data_train.csv')
    print('Finalizó el entrenamiento del Modelo')


if __name__ == "__main__":
    main()
