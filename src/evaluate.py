# Código de Evaluación - Modelo de Riesgo de Default en un empresa de Telecomunicaciones
############################################################################

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import *
import os


# Cargar la tabla transformada
def eval_model(filename):
    df = pd.read_csv(os.path.join('../data/processed', filename))
    df = df.iloc[:,1:]
    print(filename, ' cargado correctamente')
    # Leemos el modelo entrenado para usarlo
    package = '../models/best_model.pkl'
    model = pickle.load(open(package, 'rb'))
    print('Modelo importado correctamente')
    # Predecimos sobre el set de datos de validación 
    X_test = df.drop(['Churn','customerID'],axis=1)
    y_test = df[['Churn']]
    y_pred_test=model.predict(X_test)
    # Generamos métricas de diagnóstico
    cm_test = confusion_matrix(y_test,y_pred_test)
    print("Matriz de confusion: ")
    print(cm_test)
    accuracy_test=accuracy_score(y_test,y_pred_test)
    print("Accuracy: ", accuracy_test)

# Validación desde el inicio
def main():
    df = eval_model('Data_val.csv')
    print('Finalizó la validación del Modelo')


if __name__ == "__main__":
    main()