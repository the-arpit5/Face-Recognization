import cv2
import os
import numpy as np
from PIL import Image

def train_data():
    data_dir = "data"
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    faces = []
    ids = []
    
    for image in path:
        img = Image.open(image).convert('L') # Gray scale
        image_np = np.array(img, 'uint8')
        id = int(os.path.split(image)[-1].split('.')[1])
        
        faces.append(image_np)
        ids.append(id)
        
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, np.array(ids))
    clf.write("classifier.xml")
    print("Training Complete! 'classifier.xml' file ban gayi hai.")

if __name__ == "__main__":
    train_data()