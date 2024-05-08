import joblib
import cv2 as cv

import os
import pygame
from gtts import gTTS




def text_to_speech(text, language='en', filename='output.mp3'):
    tts = gTTS(text=text, lang=language, slow=False)
    tts.save(filename)
    os.system("mpg321 " + filename) 
    play_audio('output.mp3')

def play_audio(filename):
    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

loaded_model = joblib.load('your_model_file.pkl')
   
cam=cv.VideoCapture(0)
t=0
while True:
    _,img=cam.read()  
    img = cv.flip(img,1)
    img_g=cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    img_a=cv.Canny(img,50,100)
    cv.imshow("Input Live",img)
    cv.imshow("Gray_Scale Live",img_g)
    cv.imshow("Feture Extraction",img_a)
    if cv.waitKey(20)==27:
        cam.release()
        cv.destroyAllWindows()
        break
    t+=1
    if t==100:
        dic_of_result=dict(loaded_model.predict(img,confidence=40, overlap=30).json())
        text_to_speech(dic_of_result['predictions'][0]['class']+'Rupee Note')
        
        
        
   
        

# Load the model