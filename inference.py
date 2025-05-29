import joblib
import pandas as pd

# Definimos un nuevo caso de prueba 
nuevo_caso = pd.DataFrame({
    'Length1': [23],  
    'Length2': [26],  
    'Length3': [30],  
    'Height': [11], 
    'Width': [4],  
    'Species_Bream': [1],  
    'Species_Parkki': [0],  
    'Species_Perch': [0],  
    'Species_Pike': [0],   
    'Species_Roach': [0],
    'Species_Smelt': [0],
    'Species_Whitefish': [0]
})

#Cargamos el modelo
modelo=joblib.load("regresion_model.pkl")

# Realizamos la predicción con el nuevo caso de prueba
prediccion = modelo.predict(nuevo_caso)

# Imprimimos el resultado de la predicción
print(f'El nuevo pez tiene un peso de: {prediccion[0]:.2f} gramos.')
