#Function dedicated to detect the position of the face in the image if there
#is one, the landmarks and get important distances betwen certain points
#to calculate the golden ratio proportions
#later: https://www.learnopencv.com/average-face-opencv-c-python-tutorial/
#https://www.learnopencv.com/face-morph-using-opencv-cpp-python/
import numpy as np
import cv2
import dlib
from imutils.face_utils import shape_to_np
import math

class FaceAnalysis:
    def __init__(self,shape_predictor):
        self.shape_predictor = shape_predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.shape_predictor)
        #Se asume que roixy y el landmark estan con respecto al marco de referencia
        #antterior, este puede ser por ejepmlo una imagen. Esta función en ese caso se ocuparía para
        #Obtener un crop del roi, los puntos del landmark respecto al roixy
        #self.transform_landmarks_reference_frame_to_roi = lambda landmarks,roixy:np.array(landmarks)-np.array(roixy)
        pass

    def face_landmarks_enlisted(self,img):
        """Get landmarks with dlib, return list of faces and a list of faces and rectangle of the face"""
        #https:https://github.com/1adrianb/face-alignment, this is better
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #Detect faces
        rects = self.detector(gray,1)
        face_landmarks_list =[]
        rect_list = []
        if(len(rects)>0):
            for rect in rects:
                shape = shape_to_np(self.predictor(gray, rect))
                face_landmarks_list.append(shape)
                rect_list.append(rect)
            return face_landmarks_list,rect_list
        return [False,False]

    def face_landmarks_to_roi(self,face):
        """
        ESTA FUNCIÓN SE DEBERÍA OCUPAR CON LA INCLINACIÓN DE LA CARA YA CORREGIDA
        Retorna un roi por cada landmark de la lista
        Los landmark que acepta son como arrays de puntos sin etiquetas, o sea,
        como la salida de face_landmarks_enlisted, retorna el roi como (x,y,w,h)
        """
        x = np.min([p[0] for p in face])
        y = np.min([p[1] for p in face])
        w = np.max([p[0] for p in face]) - x
        h = np.max([p[1] for p in face]) - y
        return (x,y,w,h)

    def draw_points(self,point_list,img,color,ancho):
        for p in point_list:
            img = cv2.circle(img,tuple(p), ancho, color, -1)
        return img

    def rect_to_bb(self,rect):
    	# Bounding predicted by dlib and convert it
    	# to the format (x, y, w, h), por convinience
    	x = rect.left() ; y = rect.top()
    	w = rect.right()-x ; h = rect.bottom()-y
    	return (x, y, w, h)

    def get_face_roll(self,face_landmarks_list,left_eye_corner_ix=36,right_eye_corner_ix=45):
        #Using eye corner points, get face roll, accepts,
        #Inclinaciones a la derecha son negativas, a la izquierda positivas
        p0 = face_landmarks_list[left_eye_corner_ix]
        p1 = face_landmarks_list[right_eye_corner_ix]
        d_y = p1[1]-p0[1] ; d_x = p1[0]-p0[0]
        if(d_y!=0):
            return -math.atan2(d_y,d_x)
        return 0

    def rotate(self, origin, point, angle):
        """Rotate a point counterclockwise by a given angle around a given origin.
        The angle should be given in degres"""
        ox, oy = origin ; px, py = point
        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return (int(qx), int(qy))

    def correct_face_landmarks_roll_and_align(self,face_landmarks_list):
        #Just if yaw and pitch are mostly aligned
        #Primero traslada los puntos respecto al tamaño del roi
        x,y,w,h = self.face_landmarks_to_roi(face_landmarks_list)
        traslated_landmarks = np.array(face_landmarks_list)+(w,h)
        #Luego rota
        punto_de_rotacion = traslated_landmarks[8]
        angle = self.get_face_roll(face_landmarks_list)
        rotated_landmarks_list = []
        for keypoint in traslated_landmarks:
            rotated_landmarks_list.append(self.rotate(punto_de_rotacion,keypoint,angle))
        #Nuevo marco de referencia, respecto a nuevo roi
        nx,ny,nw,nh = self.face_landmarks_to_roi(rotated_landmarks_list)
        #Cambia marco de referencia
        landmarks_respect_of_roi = np.array(rotated_landmarks_list)-(nx,ny)
        #Retorna resultado y (w,h) por conveniencia, en el caso de querer saber el ancho al escalar los landmarks
        return landmarks_respect_of_roi,(x,y,w,h)

    def mask_face(self,faceAligned,landmarks,kernel=(3,3)):
        kernel = np.ones(kernel,np.uint8)
        size = faceAligned.shape[0:2]
        #Puntos límite de la cara
        p1 = landmarks[0] ; p2 = landmarks[16]
        #Proyección de ancho de la cara sobre punts límite
        meanxy = (p1[0]+p2[0])//2,(p1[1]+p2[1])//2
        p1 =(p1[0],p1[1]-meanxy[1])
        p2 = (p2[0],p2[1]-meanxy[1])
        mask = np.zeros(tuple(size),np.uint8)
        #Puntos mandibula del 0-16
        pts = [p1]
        for p in landmarks[0:17]:
            pts.append(tuple(p))
        pts.append(p2)
        cv2.fillPoly(mask, pts = [np.array(pts)], color=(255,255,255))
        mask = cv2.erode(mask,kernel,iterations = 1)
        return cv2.bitwise_and(faceAligned,faceAligned,mask=mask)

    def is_face_aligned_estimation(self,frame,landmarks,threshold=45):
        """Antes acegurate de que la los landmarks esten en una roi, solo acepta una cara a la vez,
        Esta función acepta (68) landmarks, junto con la imagen con la escala correcta,  hace una
        estimación bruta de si la cara está derecha alineada o no, Si lo está aún puede requerir ajustar
        el roll.
        referencia: https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
        """
        size = frame.shape
        #2D image points. If you change the image, you need to change vector
        #Nose tip,Chin,Left eye left corner,Right eye right corner,Left Mouth corner,Right mouth corner
        image_points = np.array([(landmarks[33, :]),(landmarks[8,  :]),(landmarks[36, :]),(landmarks[45, :]),
                                (landmarks[48, :]),(landmarks[54, :])], dtype="double")
        # 3D model points.
        #Nose tip,Chin,Left eye left corner,Right eye right corner,Left Mouth corner,Right mouth corner
        model_points = np.array([(0.0, 0.0, 0.0),(0.0, -330.0, -65.0),(-165.0, 170.0, -135.0),
                                (165.0, 170.0, -135.0),(-150.0, -150.0, -125.0),(150.0, -150.0, -125.0)])
        # Camera internals
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array([[focal_length, 0, center[0]],
                                 [0, focal_length, center[1]],
                                 [0, 0, 1]], dtype = "double")
        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        #print ("Rotation Vector:\n {0}".format(rotation_vector))
        #print ("Translation Vector:\n {0}".format(translation_vector))

        # Project a 3D point (0, 0, 1000.0) onto the image plane.
        # We use this to draw a line sticking out of the nose
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        p1 = ( int(image_points[0][0]), int(image_points[0][1]))
        p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        #calculate an aproximation of the 2d tilt of the reference line,
        ang = abs(math.atan2(p1[1]-p2[1],p1[0]-p2[0])*180/math.pi+90)%180
        if( ang > threshold):
            return False
        else:
            return True
