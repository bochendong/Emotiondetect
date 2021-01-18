import numpy as np
import cv2
import sys

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

'''
train_dir = 'data/train'
val_dir = 'data/test'
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 64

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size= batch_size,
        color_mode="grayscale",
        class_mode='categorical',
        shuffle=True)

validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size= batch_size,
        color_mode="grayscale",
        class_mode='categorical',
        shuffle=False)



emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',  input_shape=(48,48,1), padding = 'same'))
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding = 'same'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding = 'same'))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding = 'same'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding = 'same'))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding = 'same'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(64, activation='relu'))
emotion_model.add(Dense(64, activation='relu'))
emotion_model.add(Dense(7, activation='relu'))
emotion_model.add(Dense(7, activation='softmax'))

print(emotion_model.summary())
emotion_model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

emotion_model.fit_generator(train_generator, steps_per_epoch=28709 // 64,
        epochs = 15,validation_data=validation_generator, validation_steps=7178 // 64)
emotion_model.save_weights('emotion_model.h5')
'''

def CatchPICFromVideo(camera_idx):
    cap = cv2.VideoCapture(camera_idx)
    
    while True:
    # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        classfier = cv2.CascadeClassifier("/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml")
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        
        Muti_faces = classfier.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in Muti_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame,"Dong Bochen", (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('emotion', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s camera_id face_num_max path_name\r\n" % (sys.argv[0]))
    else:
        CatchPICFromVideo(0)


