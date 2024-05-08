import joblib
from gtts import gTTS
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
dic_of_result=dict(loaded_model.predict('OIP (1).jpg',confidence=40, overlap=30).json())
text_to_speech(dic_of_result['predictions'][0]['class']+'Rupee Note')