import cv2 as cv
import numpy as np
import os
import imutils
# nombre del entrenamiento
nombreEntrenamiento= "Entrenamiento.xml"
# creamos un entrenador
entrenamientoDeCara = cv.face.EigenFaceRecognizer_create()
# llamamos el archivo 
entrenamientoDeCara.read('entrenadores/'+nombreEntrenamiento)
print("entrenamiento")
print(entrenamientoDeCara)
# obtenemos el ruido
ruido = cv.CascadeClassifier(cv.data.haarcascades +'haarcascade_frontalface_alt2.xml')
# camara

capturevideo = cv.VideoCapture(0)
# capturevideo = cv.VideoCapture('videos/videoauron.mp4')

rutasrostros = "rostros"
listDatasrostros = os.listdir(rutasrostros)

if not capturevideo.isOpened():
    print("Error en camara")
    exit()

while True:
    tipocamara,camara = capturevideo.read()
    if tipocamara==False:break
    # cmabiamos resolucion
    camara=imutils.resize(camara,width=640)
    # escala de gris
    gray = cv.cvtColor(camara ,cv.COLOR_BGR2GRAY)
    # copio la imagen
    idcaptura= gray.copy()
    # asi detecto las caras
    caras = ruido.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in caras:
         rostrocapturado = idcaptura[y:y+h,x:x+w]
         rostrocapturado=cv.resize(rostrocapturado, (160,160),interpolation=cv.INTER_CUBIC)
        #  voy a comparar lo capturado con lo entrenado
        # esto me devuelve una prediccion
         resultados = entrenamientoDeCara.predict(rostrocapturado)
        #  el put text sirve para hacer un texto en la imagen
        # el {}.format sirve para meter en el string alguna variable
        # le restamos -5 a la y para poner el texto osea para que el texto este arriba
         cv.putText(camara,'{}'.format(resultados),(x,y-5),1,1.3,(0,255,0),1,cv.LINE_AA)
        #  osea si la predicicon es de 8200 si entra en esta caga
         if resultados[1] < 5700:
            cv.putText(camara,'{}'.format(listDatasrostros[resultados[0]]),(x,y-20),2,1.3,(0,255,0),1,cv.LINE_AA)
            # pinto el rectangulo
            cv.rectangle(camara, (x,y) ,(x+w,y+h),(0,255,0),2) 
         else:
            cv.putText(camara,"No encontrado",(x,y-20),2,1.3,(0,0,255),1,cv.LINE_AA)
            # pinto el rectangulo
            cv.rectangle(camara, (x,y) ,(x+w,y+h),(0,255,0),2)
    # lo mostramos
    cv.imshow('Video',camara)
    if cv.waitKey(1) & 0xFF == ord('q') :
        break;
# detenemos video  
capturevideo.release()
cv.destroyAllWindows()