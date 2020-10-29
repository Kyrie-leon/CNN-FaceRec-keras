import os
import glob
import h5py
import keras
import numpy as np
from tkinter import *
from tkinter import ttk
from PIL import Image,ImageTk
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model

from Name import *


def main():
    #获取模型权重h5文件
    model = load_model('./logs/face1.h5')
    img_rows = 300 # 高
    img_cols = 300 # 宽

    for img_location in glob.glob('./data/test/*.png'): # 限制测试图像路径以及图像格式
        img = load_img(img_location)
        img = img_to_array(img)
        #图像处理
        img = preprocess_input(np.array(img).reshape(-1,img_cols,img_rows,3))
        # out = model.predict(img)
        # img = Image.open(img_location)
        # img = np.array(img)
        img_name = (img_location.strip("face\\")).rstrip(".png")
        # img = img.reshape(-1, img_rows, img_cols, 3)
        # img = img.astype('float64')
        # img /= 255
        # intermediate_layer_model = Model(inputs=model.input,
        #                                  outputs=model.get_layer('dense_1').output)
        # intermediate_output = intermediate_layer_model.predict(img)
        
        # print(intermediate_output)
        # print(type(intermediate_output))
        # file=open('feature.txt','w') 
        # file.write(str(intermediate_output));      # 打开test.txt   如果文件不存在，创建该文件。

        
        pre_name = model.predict_classes(img) # 返回预测的标签值
        print(pre_name)

        pre = model.predict(img)

        for i in pre_name:
            for j in pre:
                name = Name.get(i)
                #print(name)
                # if name != "Audrey_Landers":
                acc = np.max(j) * 100
                print("\nPicture name is [%s]\npPredicted as [%s] with [%f%c]\n" %(img_name, name, acc, '%'))
                MainFrame = Tk()
                MainFrame.title(img_name)
                MainFrame.geometry('300x300')
                img = Image.open(img_location)
                img = ImageTk.PhotoImage(img)
                label_img = ttk.Label(MainFrame, image = img)
                label_img.pack()
                MainFrame.mainloop()
    

if __name__ == '__main__':
    main()

