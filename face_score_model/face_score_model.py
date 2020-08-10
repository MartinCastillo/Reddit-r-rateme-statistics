from keras.layers import Activation, Dropout, Flatten, Dense, MaxPooling2D, Conv2D
from keras.models import Sequential
from keras import backend as K
import numpy as np
import cv2

class ScoreImage:
    def __init__(self,image_width = 224,trained_model_dir = 'model_saved_faces.h5'):
        self.trained_model_dir = trained_model_dir
        self.image_width = image_width
        self.model = Sequential()
        self.model.add(Conv2D(32, (2, 2), input_shape=(self.image_width,self.image_width,3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(32, (2, 2)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(64, (2, 2)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(64))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))
        self.model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=["mean_absolute_percentage_error"])
        self.model.load_weights(self.trained_model_dir)

    def score_image(self,image):
        #Toma una imagen como array en grayscale, con 1 canal, con valores de 0 a 255
        #Con la cara centrada en la imgen y con ángulo despreciable, retorna score predicho
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = cv2.resize(image,(self.image_width,self.image_width))
        return self.model.predict(np.array([image/255]))[0][0]

if (__name__=="__main__"):
    pass
