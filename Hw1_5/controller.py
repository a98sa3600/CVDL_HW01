from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from UI import Ui_MainWindow
from keras.utils import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.applications.imagenet_utils import decode_predictions
from keras.applications import VGG19
from keras.utils.image_utils import load_img
from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input

label_dict={0:"airplain",1:"automobile",2:"bird",3:"cat",4:"deer",5:"dog",
            6:"frog",7:"horse",8:"ship",9:"truck"}  

class MainWindow_controller(QtWidgets.QMainWindow):
    
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        self.ui.LoadButton.clicked.connect(self.load_image) 
        self.ui.TrainButton.clicked.connect(self.TrainImage)
        self.ui.ModelButton.clicked.connect(self.Model_Structure)
        self.ui.DataButton.clicked.connect(self.Data_Augmentation)        
        self.ui.AccuaryButton.clicked.connect(self.Accuary)
        self.ui.InferenceButton.clicked.connect(self.Inference)

        
        
    def load_image(self):
        self.filepath,_= QFileDialog.getOpenFileName(self,filter='Image Files (*.png *.jpg *.jpeg *.bmp)')# start path
        if len(self.filepath) == 0: 
            print("\n Please input correct picture")
                
        self.img = cv.imread(self.filepath)
        height, width, channel = self.img.shape
        bytesPerline = 3 * width
        img = QImage(self.img.tobytes(), width, height, bytesPerline, QImage.Format_RGB888)
        qmap = QPixmap.fromImage(img)
        qheight = qmap.height() +300
        self.qmap = qmap.scaledToHeight(qheight)
        self.ui.label_1.setPixmap(self.qmap)
    
    def TrainImage(self):
        (x_train_image, y_train_label),(x_test_image, y_test_label)=cifar10.load_data() 
        num=9
        idx=0
        fig=plt.gcf()                                           
        for i in range(0, num):                            
            ax=plt.subplot(3, 3, i+1)                        
            ax.imshow(x_train_image[idx], cmap='binary')           
            title= label_dict[y_train_label[idx][0]] 
            ax.set_title(title, fontsize=10)                  
            ax.set_xticks([]);                                
            ax.set_yticks([]);                                
            idx += 1                                              
        plt.show()                                                

    def Model_Structure(self):
        (x_train,y_train),(x_test,y_test) = cifar10.load_data()
        vgg19 = VGG19(include_top=True,weights='imagenet')
        vgg19.summary()    

    def Data_Augmentation(self):
        datagen=ImageDataGenerator(rotation_range=85 , 
                             width_shift_range=0.8 , 
                             zoom_range=0.2 )
        self.qmap = np.uint8(self.img)
        x = np.expand_dims(self.img,axis=0)
        gen = datagen.flow(x,batch_size=1)
        fig = plt.figure(figsize=(5,5))
        for i in range(1,4):
            plt.subplot(1,3,i)
            x_batch = next(gen)
            plt.imshow(x_batch[0]/256)
            plt.axis('off')
        plt.show()
          
    def Accuary(self):
        image1path = "ACCURACY.png"
        image1 = cv.imread(image1path)
        image2path = "Loss.png"
        image2 = cv.imread(image2path)
        cv.imshow('ACCURACY',image1)
        cv.imshow('Loss',image2)
        
    
    def Inference(self):
        image1 = load_img(self.filepath, target_size=(32,32))
        image1 = img_to_array(image1)
        image1 = np.expand_dims(image1, axis=0)
        
        # model = load_model('Q1_5.h5')
        # pred = model.predict(image1)
        # image1 = preprocess_input(image1)
        # top_predict = decode_predictions(pred, top=1)
        # t1 = top_predict[0][0][0]
        # t2 = top_predict[0][0][1]
        # t3 = top_predict[0][0][2]
        # t3 = int(t3*100)
        
        model = VGG19(weights='imagenet', include_top=False,input_shape=(32,32,3)) 
        # Input：要辨識的影像
        #img_path = 'elephant.jpg'
        img = load_img(self.filepath, target_size=(32,32))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # 預測，取得features，維度為 (1,7,7,512)
        features = model.predict(x)
        # 取得前三個最可能的類別及機率
        print('Predicted:', decode_predictions(features, top=3)[0])
        top_predict = decode_predictions(features, top=1)
        t1 = top_predict[0][0][0]
        t2 = top_predict[0][0][1]
        t3 = top_predict[0][0][2]
        t3 = int(t3*100)
        print ('t123:',t1,t2,t3)
        
        print('top_predict:',top_predict)
        # print('pred:',pred)
        #text = "Cofidence="+str(t3)+"% \n"+"Prediction Label:"+ str(t2)
        


 



        