# importacion para la webcam
import cv2
# importacion para detectar las manos
from cvzone.HandTrackingModule import HandDetector
# importacion de numpy para crear un overlay para la captura de la mano
import numpy as np
import math
# importacion para nombre de las imagenes que guardamos
import time

# codigo para abrir la camara
cap = cv2.VideoCapture(0)
# codigo para detectar las manos
detector = HandDetector(maxHands=1)
# codigo para abrir la camara

# variables importantes
# tamaño extra de la caputura de imagen de la mano
offset = 40
imgSize = 300

# variables para guardar imagenes de las letras
folder = "Data/F"
counter = 0


while True:
    try:
        # codigo para lo que detecta la camara
        # lo que captura la camara
        # success true si detecta imagen
        # img para la informacion que captura
        success, img = cap.read()
        # codigo para detectar las manos
        # hands para la informacion de la mano
        # img para la informacion que detecta de la imagen de la mano
        hands, img = detector.findHands(img)
        # si hay una imagen
        if hands:
            # obtenemos la informaicon de la primera mano que detecta
            hand = hands[0]
            # obtenemos las cordenadas de los 4 puntos importantes de la captura de la imagen de la mano
            # X =cordenada x en la imagen capturada de imagen totoal
            # y =cordenada y en la imagen capturada de imagen totoal
            # w = ancho de la mano
            # h = alto de la imagen
            x, y, w, h = hand['bbox']
            # Con la ayuda de np crearemos un overlay para la captura de la imagen de nuestra mano
            # nos ayuda a crear un overlay blanco donde insertaremos la imagen de la mano
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            # definimos el tamaño de la nueva imagen con espacio extra para mejor visualizacion
            y1 = y - offset - 30
            y2 = y + h + offset
            x1 = x - offset
            x2 = x + w + offset
            imgCrop = img[(y1 if y1 >= 0 else 0): (y2 if y2 >= 2 else 0), x1 if x1 >= 0 else 0: x2 if x2 >= 0 else 0]
            # codigo para el overlay, insertamos la imagen de la captura de la mano en nuestro overlay blanco
            imgCropShape = imgCrop.shape

            # usamos un pequeño algoritmo para redimencionar la imagen obtenida de la mano
            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # mostramos la captura de imagen de la mano
            cv2.imshow("hand1", imgCrop)
            # np
            cv2.imshow("ImageWhite", imgWhite)
        # mostramos la capura de imagen de la webcam
        cv2.imshow("image", img)
        key = cv2.waitKey(1)
        if key == ord("s"):
            counter-=-1
            cv2.imwrite(f"{folder}/Image_{time.time()}.png", imgWhite)
            print(counter)
    except Exception as e:
        pass
