import cv2
import numpy as np
# para almacenar y mandejar archivos
import os
# para renderizar si laimagen es muy alta cabron
import imutils
# cambiar nombre al modelo
nombreModelo = "Brandon"
ruta = "rostros"
# validamos si la ruta existe
if not os.path.exists(ruta):
      os.makedirs(ruta)
      print("Ruta ruta")
      
if not os.path.exists(ruta+"/"+nombreModelo):
     os.makedirs(ruta+"/"+nombreModelo)
     print("Ruta modelo creada")
      
# habilitamos el video
capturevideo = cv2.VideoCapture(0)
# creamos calificador con los rostros de opencv
ruido = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_alt2.xml')

# id= 0
id= 0
if not capturevideo.isOpened():
     print("No hay camara cabron")
     exit()
     
while True:
    # obtenemos la camara
    tipocamara,camara = capturevideo.read()
    # si no responde la camara lo sacamos
    if tipocamara==False:break
    # cmabiamos resolucion
    camara=imutils.resize(camara,width=640)
    # escala de gris
    gray = cv2.cvtColor(camara ,cv2.COLOR_BGR2GRAY)
    # copio la imagen
    idcaptura= camara.copy()
    # asi detecto las caras
    caras = ruido.detectMultiScale(gray,1.3,5)
      # recorro la cara
    for (x,y,w,h) in caras:
        # pinto el rectangulo
         cv2.rectangle(camara, (x,y) ,(x+w,y+h),(0,255,0),2)
         rostrocapturado = idcaptura[y:y+h,x:x+w]
         rostrocapturado=cv2.resize(rostrocapturado,(160,160),interpolation=cv2.INTER_CUBIC)
         cv2.imwrite(ruta+"/"+nombreModelo+"/imagen_{}.jpg".format(id),rostrocapturado)
         id=id+1
    # lo mostramos
    cv2.imshow('Video',camara)
    if id==350:
         break;

    if cv2.waitKey(1) & 0xFF == ord('q') :
        break;
# detenemos video  
capturevideo.release()
cv2.destroyAllWindows()
