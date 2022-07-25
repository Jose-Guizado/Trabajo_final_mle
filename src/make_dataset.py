
# Script de Preparación de Datos
###################################

import pandas as pd
import numpy as np
from sklearn import preprocessing
import os


# Leemos los archivos csv
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/raw/', filename))
    print(filename, ' cargado correctamente')
    return df


# Realizamos la transformación de datos
def data_preparation(dataset):
    
    # Convertimos la variale TotalCharges
    dataset[['TotalCharges']] = dataset[['TotalCharges']].apply(pd.to_numeric, errors='coerce')
    
    ## selection of category variables
    target = 'Churn'
    exclude = ['customerID','Churn']

    cols = [x for x in dataset.columns if x not in exclude + [target]]
    cols_cat = dataset[cols].select_dtypes(['object']).columns.tolist()
    index_categorical=[cols.index(x) for x in cols_cat]

    ## For Training
    for i in cols_cat:
        le = preprocessing.LabelEncoder()
        le.fit(list(dataset[i].dropna()))
        dataset.loc[~dataset[i].isnull(),i]=le.transform(dataset[i].dropna())
        
    # Eliminamos los valores faltantes
    dataset = dataset.dropna(axis = 0)
    
    print('Transformación de datos completa')
    return dataset


# Exportamos la matriz de datos con las columnas seleccionadas
def data_exporting(df, features, filename):
    dfp = df[features]
    dfp.to_csv(os.path.join('../data/processed/', filename))
    print(filename, 'exportado correctamente en la carpeta processed')


# Generamos las matrices de datos que se necesitan para la implementación

def main():
    
    # Matriz de Entrenamiento
    df1 = read_file_csv('Data.csv')
    tdf1 = data_preparation(df1)
    data_exporting(tdf1, ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents','tenure', 'PhoneService', 'MultipleLines', 'InternetService','OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport','StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling','PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'],'Data_train.csv')
    
    # Matriz de Validación
    df2 = read_file_csv('Data_new.csv')
    tdf2 = data_preparation(df2)
    data_exporting(tdf2, ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents','tenure', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport','StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling','PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'],'Data_val.csv')
    
    # Matriz de Scoring
    df3 = read_file_csv('Data_score.csv')
    tdf3 = data_preparation(df3)
    data_exporting(tdf3, ['gender', 'SeniorCitizen', 'Partner', 'Dependents','tenure', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport','StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling','PaymentMethod', 'MonthlyCharges', 'TotalCharges'],'Data_score.csv')
    
if __name__ == "__main__":
    main()
