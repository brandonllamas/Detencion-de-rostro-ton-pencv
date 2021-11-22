import cv2
import numpy as np
import os
import imutils
# para las fechas
from datetime import datetime
# tiempo que se ejecuta
from time import time

# print(datetime.today().strftime('%Y-%m-%d'))

nombreEntrenamiento ="Entrenamiento"
# ruta de los rostros
ruta = "rostros"
# lista de rostros
listaData = os.listdir(ruta)
# print("data", listaData)

ids=[]
rostrosData=[]
id = 0

tiempoInicial = time()
# recorremos cada una
for fila in listaData:
    ruta_completa=ruta+"/"+fila    
    # recorremos los rostros dentro de la carpeta
    for rostro in os.listdir(ruta_completa):
        
        print("Imagenes: ",fila+"/"+rostro)
        # funcion para manejar arhcihvos
        ids.append(id)
        # cogemos el dato y lo convertimos a gris
        rostrosData.append(cv2.imread(ruta_completa+"/"+rostro,0))
        # imagenes = cv2.imread(ruta_completa+"/"+rostro,0)
     
    id=id+1
    tiempoTotalFinalLecuta = time()
    tiempofinalEntrenamiento = tiempoTotalFinalLecuta- tiempoInicial
    print("Timepo total Lectura:",tiempofinalEntrenamiento)
# print(dir (cv2.face))
entrenamientoModelo = cv2.face.EigenFaceRecognizer_create()
print("iniciando el entrenamiento")
entrenamientoModelo.train(rostrosData,np.array(ids))
tiempofinalEntrenamiento = time()
tiempoFinalEntrenamientoFInal = tiempofinalEntrenamiento - tiempofinalEntrenamiento
print("tiempo entrenamiento total : ",tiempoFinalEntrenamientoFInal)
entrenamientoModelo.write('entrenadores/'+nombreEntrenamiento+".xml")
print("entrenamiento concluido")