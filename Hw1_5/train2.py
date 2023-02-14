import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from keras.applications.vgg19 import VGG19
from keras.layers import  Dense
from keras.models import Model
import numpy as np
from keras.utils import to_categorical
from keras.datasets import cifar10
import tensorflow as tf

def resize_img(img):
    Image = img.shape[0]
    resize_img = np.zeros((Image, 32,32,3))
    for i in range(Image):
        resize_img[i] = cv.resize(img[i,:,:,:],(32,32))
    return resize_img

def build_model(num_classes=10):
    pred_model = VGG19(include_top=False, weights='imagenet',
                              input_shape=(32, 32, 3),
                              pooling='max', classifier_activation='softmax')
    output_layer = Dense(num_classes, activation="softmax", name="output_layer")

    model =Model(pred_model.inputs, output_layer(pred_model.output))

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

    return model

# set image data
(x_train,y_train),(x_test,y_test) = cifar10.load_data()
x_train = resize_img(x_train)
x_test = resize_img(x_test)
y_train = to_categorical(y_train,num_classes=10)
y_test = to_categorical(y_test,num_classes=10)

# Model_Set
model = build_model()
model.summary()

# compile model
model.compile(optimizer="Adam",loss="categorical_crossentropy",metrics=["accuracy"])

#Train.
Hist = model.fit(x_train,y_train,validation_split=0.15,epochs=60,batch_size=1200)
model.save('Q1_5_2.h5')

plt.subplots(figsize=(6,4))
plt.plot(Hist.epoch,Hist.history["loss"],color="red",label="Training Loss")
plt.plot(Hist.epoch,Hist.history["val_loss"],color="blue",label="Testing Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss")
plt.savefig("Loss.png")
plt.show()

plt.subplots(figsize=(6,4))
plt.plot(Hist.epoch,Hist.history["accuracy"],color="red",label="Training Accuracy")
plt.plot(Hist.epoch,Hist.history["val_accuracy"],color="blue",label="Testing Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy")
plt.savefig("ACCURACY.png")
plt.show()