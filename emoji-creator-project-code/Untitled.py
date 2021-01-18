#!/usr/bin/env python
# coding: utf-8

# In[10]:


import tkinter as tk
from PIL import Image, ImageTk
import cv2

import numpy as np
import cv2
import sys

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPool2D
from keras.preprocessing.image import ImageDataGenerator


# In[14]:


model = Sequential()

model.add(Conv2D(input_shape=(48,48,1),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Flatten())
model.add(Dense(units=1024,activation="relu"))
model.add(Dense(units=1024,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(7, activation="softmax"))

model.load_weights('model.h5')


cv2.ocl.setUseOpenCL(False)

emoji_dist={0:"emojis/angry.png",2:"emojis/disgusted.png",2:"emojis/fearful.png",3:"emojis/happy.png",4:"emojis/neutral.png",5:"emojis/sad.png",6:"emojis/surpriced.png"}


file_path = "emojis/neutral.png"


# In[15]:


class MainWindow():
    def __init__(self, window, cap):
        self.window = window
        self.cap = cap
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.interval = 20
        self.canvas = tk.Canvas(self.window, width=1000, height=300)
        self.canvas.grid(row=0, column=0)
        self.update_image()
        
    def update_image(self):
        global file_path
        
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        bounding_box = cv2.CascadeClassifier("/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml")
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in num_faces:
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            
            maxindex = int(np.argmax(prediction))
            file_path = emoji_dist[maxindex]
    
        
        image1 = cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2RGB)
        image1 = Image.fromarray(image1)
        image1 = image1.resize((500, 300))

        image2 = Image.open(file_path)
        image2 =image2.resize((500, 300))
        
        image1_size = image1.size
        image2_size = image2.size

        
        new_image = Image.new('RGB',(2*image1_size[0], image1_size[1]), (250,250,250))
        new_image.paste(image1,(0,0))
        new_image.paste(image2,(image1_size[0],0))
        
        self.image = new_image
        
        self.image = ImageTk.PhotoImage(self.image)
        
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)
        self.window.after(self.interval, self.update_image)


# In[16]:


root = tk.Tk()
MainWindow(root, cv2.VideoCapture(0))
root.mainloop()


# In[ ]:





# In[ ]:




