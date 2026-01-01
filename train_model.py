import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
path = 'dataset'

# Create the recognizer (The AI Brain)
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml");

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
    
    for imagePath in imagePaths:
        # Ignore hidden files (like .DS_Store on Mac)
        if os.path.split(imagePath)[-1].startswith("."):
            continue

        # Convert image to grayscale
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img,'uint8')
        
        # Get the ID from the filename (User.101.1.jpg -> ID is 101)
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")

faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Save the model into the 'models' folder
if not os.path.exists('models'):
    os.makedirs('models')

recognizer.write('models/trainer.yml') 

print(f"\n [INFO] {len(np.unique(ids))} faces trained. Exiting Program")