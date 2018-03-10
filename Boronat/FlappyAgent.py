from keras.models import load_model
from collections import deque
import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray

global stackedframes

def process_screen(x):
    return 255*resize(rgb2gray(x[60:, 25:310,:]),(80,80))

model = load_model("FlappyModel_900000")
list_actions = [119,None]
stackedframes = deque([np.zeros((80,80)),np.zeros((80,80)),np.zeros((80,80)),np.zeros((80,80))], maxlen=4)

def FlappyPolicy(state,screen):
    
    screen = process_screen(screen)
    stackedframes.append(screen)
    frameStacked = np.stack(stackedframes, axis=-1)
    action = list_actions[np.argmax(model.predict(np.expand_dims(frameStacked,axis=0)))]
    return action
    


