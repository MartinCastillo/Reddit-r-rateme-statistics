#This code contains all the functions required to interact with redit,
#Functions that control the bot and get information from the posts
#E.g: get_submissions_image_array get the array of the image of a post
from praw import Reddit
from praw.models import MoreComments
import requests
import cv2
import numpy as np
import re
from pdb import set_trace

class Bot(Reddit):
    def __init__(self,user_agent,client_id,client_secret,username,password,coment_score_template):
        self.user_agent = user_agent
        self.client_id = client_id
        self.client_secret = client_secret
        self.username = username
        self.password = password
        self.coment_score_template = coment_score_template
        #Get the Reddit's class methods to have everithing related to reddit
        #interface in a single class
        Reddit.__init__(self,user_agent=self.user_agent,
                             client_id=self.client_id, client_secret=self.client_secret,
                             username=self.username, password=self.password)

    def coment_score(self,submission,score):
        submission.reply(self.coment_score_template.format(score))

    def post_contains_image(self,submission):
        #Hubo un bug con el anterior submission.preview
        if(not(submission.is_self)):
            if(re.search(r"\.(jpg|png|jpeg)$",submission.url)):
                return True
        return False

    def get_submissions_image_url(self,submission):
        if(self.post_contains_image(submission)):
            return submission.url
        return False

    def get_image_array_from_url(self,url):
        resp = requests.get(url, stream=True).raw
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image

    def get_submissions_image_array(self,submission):
        #Get and decode the image of the post using the url from get_submissions_image_url
        url = self.get_submissions_image_url(submission)
        if(url):
            return self.get_image_array_from_url(url)
        return 0

    def get_op_gender_and_age(self,submission):
        #If there is something like [M22], [F23], [12F], etc in the title
        #- Guión con slash o sin 12-M 14M 22F 22/F, y al revés M-23 F12 M/40
        #- Letra mayúscula y número seguídos de espacio, ej: M22 ,34F , 15M ,etc
        #- Letra con o sin número entre () o [] ej: [M22], [M], (F13)
        gender_age_tag = re.search(r"(\d+[/-]?[MFmf]|[MFmf][/-]?\d+|[MF][\d+]? |[\[\(][\d+]?[MFmf][\d+]?[\]\)])",submission.title)
        if(gender_age_tag):
            age = re.search(r"\d+",gender_age_tag.group())
            age = int(age.group()) if age else "unknown"
            return re.search(r"[mf]",gender_age_tag.group().lower()).group(),age
        return "unknown","unknown"

    def get_average_score_in_coments(self,submission,limit):
        global MoreComments
        #Read coments and sreach in the for ranks in a format of 0 to 10 float number
        #When gets enought, average them and return the score, limit is the maximum number of
        #coments to checke
        pila_score_post = 0 ; pila_score_post_counter = 0
        for (n_comment,top_level_comment) in enumerate(submission.comments):
            if isinstance(top_level_comment, MoreComments):
                continue
            """Comon context of rates:
            ° x/10 e.g: 5/10 or 6/10, alwas 0 to 10,
            ° x1-x2 e.g: 6.5-7 means from 6.5 to 7 out of 10
            ° just a float or integer number e.g: 6.5, means 6.5 out of 10
            PRECAUCIÓN:
            Sometimes someone can say ages like 'you look like you are 13'
            Promedia los puntajes en un solo comentario
            """
            matchs = re.findall(r"(\d*\.\d+|\d+)[-/](\d*\.\d+|\d+)",top_level_comment.body)
            pila_comentario = 0
            for numbers in matchs:
                n1 = float(numbers[0]) ; n2 = float(numbers[1])
                #De primeras el valor es el promeio
                score = (n1+n2)/2
                #Pero si uno de los 2 vale 10, o ambos, solamente elige el menor de ambos
                if((n1==10)or(n2==10)):
                    score = n1 if n1<=n2 else n2
                pila_comentario += score
            if(len(matchs)>0):
                pila_comentario = pila_comentario/(len(matchs))
                if(pila_comentario<=10):
                    pila_score_post += pila_comentario
                    pila_score_post_counter += 1
            else:
                continue
            n_comment+=1
            if(n_comment>limit):
                break
        if(pila_score_post_counter!=0):
            #Retorna preomedio de scores en comentarios, si ha recopilado algún score
            return pila_score_post/pila_score_post_counter
        return False
