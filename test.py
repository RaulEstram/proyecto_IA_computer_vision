# Importaciones
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Captura de la imagen
cap = cv2.VideoCapture(0)
# Desimos cuantas manos queremos capturar
detector = HandDetector(maxHands=1)
# carga del modelo
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# Varaibles para la imagen
offset = 20
imgSize = 300

# Lista de las posibles letras
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

# empieza el codigo realmente
while True:
    # leemos o campuramos la informaicon de lo que captura la camara
    success, img = cap.read()
    # guardamos en una variable la inforamcion de la imagen capturada por la camara
    imgOutput = img.copy()
    # buscamos manos y guardamos la informacion
    hands, img = detector.findHands(img)
    # usamos un try excpet por si hay errores
    try:
        # comprobamos si la camapara detecto una mano
        if hands:
            # obtenemos la inforamcion de la mano
            hand = hands[0]
            # obtenemos las cordenadas y el ancho y largo de la informacion de la mano
            x, y, w, h = hand['bbox']

            # codigo para crear el overlay de la iamgen capturada de la mano
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCropa = img[y - offset:y + h + offset, x - offset:x + w + offset]
            imgCropShape = imgCropa.shape

            # calculamos el aspectRatio de la imagen de la mano para saber como redimencionar la imagen de la mano
            aspectRatio = h / w

            # redimencionamos la mano
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCropa, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
                # realizamos la clasificacion de la imagen de la mano capturada
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCropa, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                # realizamos la clasificacion de la imagen de la mano capturada
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

            # agregamos la informacion de la mano a la captura de la camara que le mostraremos al usuario
            cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                          (x - offset+90, y - offset-50+50), (0, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x-offset, y-offset),
                          (x + w+offset, y + h+offset), (0, 0, 255), 4)

        # le mostramos la informacion al usuario
        cv2.imshow("Image", imgOutput)
        cv2.waitKey(1)
    except:
        pass