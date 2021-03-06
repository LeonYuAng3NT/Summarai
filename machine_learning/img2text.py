
import cv2
import os
import glob
import pandas as pd
import math
import pytesseract
import numpy as np
from skimage.filters import threshold_local

# importing libraries to read text from image
from PIL import Image
from scipy.misc import imread, imsave, imresize
from natsort import natsorted
from labelConv import label2int, int2label
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2D

def get_contour_precedence_two(contour, cols):
    tolerance = 10
    origin_point = cv2.boundingRect(contour)
    return ((origin_point[0] // tolerance) * tolerance) * cols + origin_point[1]
def get_contour_precedence(contour, cols):
    tolerance_factor = 10
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]

def captch_ex2(file_name, index):
    img = cv2.imread(file_name)
    img = cv2.resize(img,None,fx=1, fy=1, interpolation = cv2.INTER_CUBIC)
        
    img_final = cv2.imread(file_name)
    img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 180, 255, cv2.THRESH_BINARY)
    image_final = cv2.bitwise_and(img2gray, img2gray, mask=mask)
    ret, new_img = cv2.threshold(image_final, 180, 255, cv2.THRESH_BINARY)  # for black text , cv.THRESH_BINARY_INV
    # Remove noisy portion 

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1,
                                                         1))  # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
    dilated = cv2.dilate(new_img, kernel, iterations=0)  # dilate , more the iteration more the dilation

    # for cv2.x.x

    _, contours,hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # get contours
    contours.sort(key=lambda x:get_contour_precedence_two(x, img.shape[1]))
    index2 = 1
    for contour in contours:
        # get rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(contour)

        # draw rectangle around contour on original image
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 1)

        
        #you can crop image and send to OCR  , false detected will return no text :)
        cropped = img[y :y +  h , x : x + w]
        s = 'data/test/crop_' + str(index) + '_' + str(index2) + '.jpg' 
        #print(s)
        cv2.imwrite(s , cropped)
        #cv2.imshow('captcha_result', cropped)
        #cv2.waitKey()
        index2 = index2 + 1

    return index2-1

def extract_text_img(file_name):
        img = cv2.imread(file_path)
        ratio = img.shape[0]/500.0
        original_img = img.copy()

        # converting image into grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # blurring and finding edges of the image
        blurred = cv2.GaussianBlur(gray, (5,5) ,0)
        edged = cv2.Canny(gray, 75, 200)

        # applying threshold to grayscale image
        thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]

        # finding contours
        (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # draw contours on image 
        cv2.drawContours(img, cnts, -1, (240, 0, 159), 3)

        H,W = img.shape[:2]
        for cnt in cnts:
            x,y,w,h = cv2.boundingRect(cnt)
            if cv2.contourArea(cnt) > 100 and (0.7 < w/h < 1.3) and (W/4 < x + w//2 < W*3/4) and (H/4 < y + h//2 < H*3/4):
                break

        # creating mask and performing bitwise-op
        mask = np.zeros(img.shape[:2],np.uint8)
        cv2.drawContours(mask, [cnt],-1, 255, -1)
        dst = cv2.bitwise_and(img, img, mask=mask)

        # displaying image and saving in the directory
        gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        cv2.imwrite(file_name, dst)


        # fetching text from the image and storing it into a text file
        detailed_text = pytesseract.image_to_string(Image.open(file_name))
        return detailed_text
def generate_model():
    img_rows, img_cols = 20, 20
    batch_size = 128
    nb_classes = 62 
    nb_epoch = 500
    model = Sequential()

    model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu', input_shape=(img_rows, img_cols, 1)))
    model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, kernel_initializer='he_normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, kernel_initializer='he_normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, kernel_initializer='he_normal', activation='softmax'))
    model.load_weights("best.kerasModelWeights")
    return model

def CNN(model, file, records):
    X = np.load(file)
    Y = model.predict_classes(X)
    
    # Translate integers to character labels
    vInt2label = np.vectorize(int2label)
    Y = vInt2label(Y)
    
    list_of_words = []
    k = 0
    for i in range(len(records)):
        s = ''
        for j in range(records[i][1]):
            c = Y[k].lower()
            if(c == 'j'):
                s += ''
            else:
                s += c
            k += 1
            list_of_words.append(s.lower())
    
    return list_of_words
if __name__ == "__main__":
    model = generate_model()
    for file in all_files:
        list_of_words = CNN(model, file, records)
        detailed_text = extract_text_img()
    