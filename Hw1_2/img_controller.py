from dis import show_code
from email.mime import image
from pickletools import floatnl
import numpy as np
import cv2 as cv
import os
import skimage.exposure as exposure

class img_controller(object):
    def __init__(self, img_path1):
        self.img_path1 = img_path1
        self.read_file_and_init() 
        
    def read_file_and_init(self):
        if self.img_path1 != '':
            self.img = cv.imread(self.img_path1)
        
    def set_path(self,img_path):
        self.img_path1 = img_path
        self.filename1= os.path.basename(self.img_path1) 
        self.read_file_and_init()
           
    def conv(self,image,filter) :
        height,width= image.shape[:2]
        filterHeight,filterWidth = filter.shape[:2]
        newImageWidth = width - filterWidth + 1
        newImageHeight = height - filterHeight + 1
        convImage=np.zeros(image.shape[:2],dtype="int32")
        for i in range( 0, newImageHeight):
            for j in range( 0,newImageWidth):
                for h in range(i,i + filterHeight):
                    for w in range(j,j + filterWidth):
                        convImage[i, j] += filter[h-i, w-j] * image[h, w]
                if convImage[i, j] > 255:
                    convImage[i, j]= 255
                elif convImage[i, j] < 0 :
                    convImage[i, j] = 0
        dst = cv.convertScaleAbs(convImage)
        return dst
    
    def Gaussian_Blur(self):
        img = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        y,x = np.mgrid[-1:2, -1:2]
        kernel = np.exp(-(x**2+y**2))
        kernel = kernel / kernel.sum()
        self.GuassianImg = self.conv(img,kernel)
        cv.imshow("Gaussian_Blur",self.GuassianImg)
    
    
    def Sobel_X(self):
        img=self.GuassianImg
        Sobel_XFilter = np.int32([ [-1,0,1], [-2,0,2], [-1,0,1] ])
        self.sobelx = self.conv(img ,Sobel_XFilter)
        self.sobelx = cv.convertScaleAbs(self.sobelx)
        cv.imshow("Sobel_X",self.sobelx)
        


    def Sobel_Y(self):
        img=self.GuassianImg
        Sobel_YFilter = np.int32([ [1,2,1], [0,0,0], [-1,-2,-1] ])
        self.sobely= self.conv(img ,Sobel_YFilter)
        self.sobely = cv.convertScaleAbs(self.sobely)
        cv.imshow("Sobel_Y",self.sobely)  

    def Magnitude(self):
        sx=self.sobelx
        sy=self.sobely
        sx = np.uint8(np.absolute(sx))
        sy = np.uint8(np.absolute(sy))
        # add together and take square root
        sobelCombined = (sx+sy)
        cv.imshow('Magnitude',sobelCombined)
        

    def Resize(self):
        image = self.img
        SizeMatrix = np.zeros((430,430,3), dtype='uint8')
        resizeimg = cv.resize(image,(215,215))
        SizeMatrix[0:215, 0:215] = resizeimg        
        self.resize = SizeMatrix
        #show and store image
        cv.imshow("Resize",SizeMatrix)
        cv.imwrite("Resize.png",SizeMatrix)
        
    def Translation(self):
        image = self.resize
        transmatrix = np.float32([[1,0,215],[0,1,215]])
        transimg = cv.warpAffine(image,transmatrix,(430,430))
        self.translation = cv.addWeighted(image,1,transimg,1,0)
        self.translation = self.translation.astype(np.uint8)
        #show and store image
        cv.imshow("Translation",self.translation)
        cv.imwrite("Translation.png",self.translation)        
    def Rotation(self):
        image= self.translation
        rotationMatrix = cv.getRotationMatrix2D((215,215),45,0.5)
        self.rotation = cv.warpAffine(image,rotationMatrix,(430,430))
        #show and store image  
        cv.imshow("Rotation",self.rotation)
        cv.imwrite("Rotation.png",self.rotation)      
        
    def Shearing(self):
        image = self.rotation
        location1 = np.float32([[50,50],[200,50],[50,200]])
        location2 = np.float32([[10,100],[100,50],[100,250]]) 
        Shearimg = cv.getAffineTransform(location1,location2)
        self.shearing = cv.warpAffine(image,Shearimg,(430,430))
        #show and store image
        cv.imshow("Shearing",self.shearing) 
        cv.imwrite("Shearing.png",self.shearing)     
        
        

