# -*- coding: utf-8 -*-
#Local imports
from FaceAnalysis.FaceAnalysis import FaceAnalysis
from FaceAnalysis.FaceAligner import FaceAligner
from Bot.Bot import Bot
#Other imports
import os ; import sys
import pandas as pd
import numpy as np
import cv2

user_agent=''
client_id=''
client_secret=''
username=''
password=''

#https://www.reddit.com/prefs/apps
#app's client secret is: PNj8wSpR63s5VeJD4ZyDKnvJM5Y
#Id: Ra4-_6k5kferPQ
#Perfil: martinn_castillo
#Data tags: id,landmark,image,gender,age,score
#Note: You can access to the submission's instance with the submission's id
#with the url like: https://www.reddit.com/r/subreddit/comments/id

def imprimir_conteo_posts_guardados(conteo1,conteo2,conteo_g):
    mensaje = "[Saved faces]: {} / [Added faces]: {} / [Unknown tagged]: {}"
    formato = mensaje.format(conteo1,conteo2,conteo_g)
    backtrack = '\b'*(len(str(conteo1)+str(conteo2)+str(conteo_g)+mensaje))
    sys.stdout.write(backtrack)
    sys.stdout.write(backtrack + formato) ; sys.stdout.flush()

def post_scraping(bot,data_storage_src,subreddit_dir,limit_posts=None,limite_comentarios = 25,face_image_width = 224):
    face_anlysis = FaceAnalysis(shape_predictor = "FaceAnalysis/shape_predictor_68_face_landmarks.dat")
    #Iterate over subreddit's post and saves data like id, score, landmarks, etc
    #Get saved ids to check if there are already saved posts with te same id
    df = pd.read_pickle(data_storage_src)
    saved_ids = df['id'].to_numpy()
    numero_de_datos = df.shape[0]
    fa = FaceAligner(desiredFaceWidth=face_image_width)
    subreddit = bot.subreddit(subreddit_dir)
    #Loop sobre posts
    for submission in subreddit.hot(limit=limit_posts):
        #Busca si ya está en los posts guardados en data
        if(bot.post_contains_image(submission) and not(submission.id in saved_ids)):
            #Si post contiene imagen, obtiene la información
            average_score_in_coments = bot.get_average_score_in_coments(submission,limite_comentarios)
            if(average_score_in_coments):
                (op_gender,op_age) = bot.get_op_gender_and_age(submission)
                image = bot.get_submissions_image_array(submission)
                #Obtine landmarks sucios/crudos, y los bounding box de la cara,luego chequea que no esté vacio
                #El indice 0 son los landmarks el 1 los bounding box
                landmarks_list,rect_list = face_anlysis.face_landmarks_enlisted(image)
                if(landmarks_list):
                    for ix,landmark in enumerate(landmarks_list):
                        """Cuidado, puede que is_face_aligned_estimation este muy rota, para cosas raras como imagenes con muchas caras"""
                        #revisa si la cara está lo suficientemente derecha
                        if(face_anlysis.is_face_aligned_estimation(image,landmark,threshold=40)):
                            #Obtine cara alineada,la enmascara cara, sacando el fondo, y la pasa a escala de grises
                            faceAligned = fa.alignFace(image, landmark, rect_list[ix])
                            #landmarks_nuevos,rects_nuevos = face_anlysis.face_landmarks_enlisted(faceAligned)
                            #Este lio de comentarios es por omitir temporalmente la mascara de cara, y cosas como escala de grises
                            landmarks_nuevos=True
                            if(landmarks_nuevos):
                                res = faceAligned
                                #res = face_anlysis.mask_face(faceAligned,landmarks_nuevos[0])
                                #res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
                                #Alinea landmark
                                clean_landmark , w_h = face_anlysis.correct_face_landmarks_roll_and_align(landmark)
                                #Save data and print info
                                df = df.append({'id':submission.id,
                                                'landmark':clean_landmark,
                                                'image':res,
                                                'gender':op_gender,
                                                'age':op_age,
                                                'score':average_score_in_coments/10
                                                }, ignore_index = True)
                                df.to_pickle(data_storage_src)
                                imprimir_conteo_posts_guardados(df.shape[0],df.shape[0]-numero_de_datos,
                                    df[df['gender']=="unknown"].shape[0])
    return 1

def post_predicting(bot,subreddit_dir,face_image_width = 224):
    from face_score_model.face_score_model import ScoreImage
    face_anlysis = FaceAnalysis(shape_predictor = "FaceAnalysis/shape_predictor_68_face_landmarks.dat")
    fa = FaceAligner(desiredFaceWidth=face_image_width)
    image_rater = ScoreImage(image_width = 224,trained_model_dir="face_score_model/model_saved_faces.h5")
    subreddit = bot.subreddit(subreddit_dir)
    for submission in subreddit.hot(limit=None):
        if(bot.post_contains_image(submission)):
            image = bot.get_submissions_image_array(submission)
            landmarks_list,rect_list = face_anlysis.face_landmarks_enlisted(image)
            if(landmarks_list):
                for ix,landmark in enumerate(landmarks_list):
                    if(face_anlysis.is_face_aligned_estimation(image,landmark,threshold=30)):
                        faceAligned = fa.alignFace(image, landmark, rect_list[ix])
                        res = cv2.cvtColor(faceAligned, cv2.COLOR_BGR2GRAY)
                        score = round(image_rater.score_image(res)*10,1)
                        print("score:{} / url: https://www.reddit.com/r/{}/comments/{}".format(score,subreddit,submission.id))
    return 1

if(__name__=='__main__'):
    try:
        bot = Bot(user_agent=user_agent, client_id=client_id,
                client_secret=client_secret,username=username,
                password=password,coment_score_template='Score< {}')
        post_predicting(bot,"rateme",face_image_width = 224)
    except KeyboardInterrupt:
        print("\nCerrando...")
        sys.exit(0)
