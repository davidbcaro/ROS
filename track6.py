import cv2
from markerfunctions import *
from imutils.video import VideoStream
from scipy.spatial import distance as dist
import numpy as np
import datetime
import argparse
import imutils
import time
import os
import math

########################################################################

# inicializar variables
SHAPE_RESIZE = 100.0
BLACK_THRESHOLD = 100
WHITE_THRESHOLD = 155
LM = 19.5
#PM = 82
PATTERN = [0, 1, 0, 1, 0, 0, 0, 1, 1] 	# marca 1 
# PATTERN = [0, 1, 0, 1, 1, 1, 1, 0, 1] # marca 2 Y
# PATTERN = [1, 0, 1, 0, 1, 0, 1, 0, 1] # marca 3 X 
# PATTERN = [0, 1, 0, 1, 1, 1, 0, 1, 0] # marca 4 + 

# constructruir los argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())

# iniciar el video y evitar que el sensor de la camara se caliente
# (1024, 768) (2048, 1536) (2592, 1952) (30280, 2464)
vs = VideoStream(usePiCamera=args["picamera"] > 0, resolution=(1024, 768)).start()
time.sleep(2.0)

 
while True:
	
	# tiempo inicial
	to = time.time()
	# leer la imagen de la camara
	image = vs.read()
	image = imutils.resize(image, width=800)
	# convertir a escala de grises, suavizar y detectar bordes
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5,5), 0)
	edges = cv2.Canny(gray, 100, 200)
	# cv2.imshow('Edges', edges)
	
	# encontrar los contornos
	(_, cnts, _) = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	# seleccionar diez contornos por el tamano de area 
	contours = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
	
	# bucle sobre los contornos para encontrar la marca o patron
	for contour in contours:
		
		# aproximar la forma de los objetos encontrados y determinar puntos o vertices 
		perimeter = cv2.arcLength(contour, True)
		approx = cv2.approxPolyDP(contour, 0.01*perimeter, True)
		
		# asegurarse que el contorno sea de forma cuadrilatera y descartar el resto
		# if len(approx) >= 4 and len(approx) <= 6:       
		if len(approx) == 4:
			
			# corregir la perspectiva del objeto cuadrilatero encontrado
			topdown_quad = get_topdown_quad(gray, approx.reshape(4, 2))
			# redimensionar el objeto a 100 pixeles
			#resized_shape = resize_image(topdown_quad, SHAPE_RESIZE)
			# cv2.imshow('Topdown', topdown_quad)
			
			# comprobar si dentro del borde del objeto hay un pixel [5, 5] negro
			# if resized_shape[5, 5] > BLACK_THRESHOLD: continue
			if topdown_quad[(topdown_quad.shape[0]/100.0)*5,
							(topdown_quad.shape[1]/100.0)*5] > BLACK_THRESHOLD: continue
							
			#glyph_found = False
			glyph_found = False
			
			#for i in range(4):
			try:
				# comprobar cada celda del glifo, si el pixel es blanco o negro
				glyph_pattern = get_glyph_pattern(topdown_quad, BLACK_THRESHOLD, WHITE_THRESHOLD)
			except:
				continue
				# comprobar si el patron encontrado coincide con la variable
			#if glyph_pattern == PATTERN:
				#glyph_found = True
				#break
				# rotar la imagen 90 grados
			if not glyph_pattern: continue
			
			if glyph_pattern == PATTERN:
				glyph_found = True
				break
			topdown_quad = rotate_image(topdown_quad, 90)
			
			if glyph_found:
				# dibujar el contorno de la marca
				cv2.drawContours(image, [approx], -1, (0, 0, 255), 4)
                # calcular el centro de la marca y dibujar 
				M = cv2.moments(approx)
				(cX, cY) = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
				cv2.line(image, (cX, cY), (cX, cY), (255, 0, 0), 3)
				# dibujar el centro del frame y dibujar una linea respecto al objetivo 
				width = 800
				Co = width/2
				cv2.line(image, (Co, Co), (Co, Co), (255, 0, 0), 3)
				cv2.line(image, (Co, Co), (Co, Co), (255, 0, 0), 3)
				cv2.line(image, (Co, Co), (cX, cY), (0, 255, 0), 3)
				#
				(x, y, w, h) = cv2.boundingRect(approx)
				#aspectRatio = w
				print('El ancho en px es:'+ str(w))
				print('El ancho en px es:'+ str(h))
				
				# calcular la distancia euclidiana (px)
				Dpx = dist.euclidean((cX, cY), (Co, Co))
				print('La distancia en px es:'+ str(Dpx))
				# calcular la distancia (cm) de error
				PM = w
				Dcm = LM*Dpx/PM
				#cv2.putText(image, str(Dcm), (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
				print('La distancia en cm es:'+ str(Dcm))
				# calcular el angulo del vector
				angle_rad = math.atan2(cY, cX)
				angle_deg = math.degrees(angle_rad)
				#print('Angulo en grados: ' + str(angle_deg) + ' ' + 'Cuadrante I')
				# tiempo final
				tf = time.time()
				# tiempo total de ejecucion
				te = tf - to
				print('El tiempo de ejecucion fue:'+ str(te))
				# calcular la velocidad lineal de error
				Vt = Dcm / te
				Vx = Vt * math.sin(angle_rad)
				Vy = Vt * math.cos(angle_rad)
				Vz = 0
				print('La velocidad total:'+ str(Vt))
				print('Velocidad lineal')
				print('Vx: ' + str(Vx))
				print('Vx: ' + str(Vy))
				print('Vx: ' + str(Vz))
				# calcular la velocidad angular
				Wx = 0
				Wy = 0
				Wz = angle_rad
				print('Velocidad angular')
				print('Wx: ' + str(Wx))
				print('Wx: ' + str(Wy))
				print('Wx: ' + str(Wz))
				
				twist = Twist()
				twist.linear.x = Vx
				twist.linear.y = Vy
				twist.linear.z = Vz
				twist.angular.x = Wx
				twist.angular.y = Wy
				twist.angular.z = Wz
				#break
				
	cv2.imshow('Seguimiento de la marca', image)
	key = cv2.waitKey(1) & 0xFF
				
	# si la tecla `q` es presionada, salir del bucle
	if key == ord("q"):
		#os.system('clear')
		#cv2.destroyAllWindows()
		#vs.stop()
		break
					
# limpiar espacio de trabajo
cv2.destroyAllWindows()
vs.stop()
