import pandas as pd
import numpy as np
dataset = pd.read_csv('Base de datos Nutrientes.csv')
datos_consumo = dataset.iloc[:,1:8]

print(datos_consumo)

estadistica_descriptiva = datos_consumo.describe()

print(" ")
print("Estadística descriptiva")
print(estadistica_descriptiva)

datos_consumo.isnull().any()
dataset = datos_consumo.fillna(method='ffill') 

X = dataset[['Protein', 'Fat', 'Fiber', 'Carbs', 'Sat.Fat']].values
y = dataset['Calories'].values 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
modelo_regresion = LinearRegression()
modelo_regresion.fit(X_train, y_train) 
x_columns = ['Protein', 'Fat', 'Fiber', 'Carbs','Sat.Fat']
coeff_df = pd.DataFrame(modelo_regresion.coef_, x_columns, columns=['Coeficientes'])

print(" ")
print(coeff_df) 

y_pred = modelo_regresion.predict(X_test)
validacion = pd.DataFrame({'Actual': y_test, 'Predicción': y_pred})
muestra_validacion = validacion.head(25)

print(" ")
print(muestra_validacion)

from sklearn import metrics

print(" ")
print("Raíz de la desviación media al cuadrado:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

import matplotlib.pyplot as plt
muestra_validacion.plot.bar(rot=0)
plt.title("Comparación de calorías actuales y de predicción")
plt.xlabel("Muestra de alimentos")
plt.ylabel("Cantidad de calorías") 
plt.savefig("Plot1.png")
