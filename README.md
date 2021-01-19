# Emotiondetect
Detect human face emotion using OpenCV and tensorflow
* [Data preparation](https://github.com/bochendong/emotionDetect/blob/master/README.md#data-preparation)
* [Build and train Model](https://github.com/bochendong/emotionDetect/blob/master/README.md#build-and-train-model)
* [GUI](https://github.com/bochendong/emotionDetect/blob/master/README.md#gui)
* [Demo](https://github.com/bochendong/emotionDetect/blob/master/README.md#demo)

## Data preparation
```Python
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
```     

## Build and train Model:
```Python
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

emotion_model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
emotion_model.fit_generator(train_generator, steps_per_epoch=28709 // 64,
        epochs = 15,validation_data=validation_generator, validation_steps=7178 // 64)
emotion_model.save_weights('emotion_model.h5')
```

## GUI:
```Python
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
```
## Demo:
<p align="center">
	<img src="https://github.com/bochendong/emotionDetect/blob/master/imgs/IMG_6936.JPG"
        width="1400" height="500">
	<p align="center">
</p>
