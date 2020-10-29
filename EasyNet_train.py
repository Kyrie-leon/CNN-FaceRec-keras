
import os
import glob
import h5py
import keras
import numpy as np
from Name import *
from PIL import Image
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.models import load_model
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.applications.imagenet_utils import preprocess_input
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from utils.utils import get_random_data

image_w = 300 #图片的宽
image_h = 300 #图片的高
num_class = 40 #标签数量

def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (0,0,0))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image


def generate_arrays_from_file(lines,batch_size,train):
    # 获取总长度
    n = len(lines)
    i = 0
    while 1:
        X_train = []    #300x300x3
        Y_train = []    #label
        # 获取一个batch_size大小的数据
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            label_name = (lines[i].split(';')[1]).strip('\n')
            file_name = str(Name.get(int(label_name)))
            # 从文件中读取图像
            img = Image.open(r".\data\face" +"\\"+ file_name +"\\"+ name)
            if train == True:
                img = np.array(get_random_data(img,[image_h,image_w]),dtype = np.float64)
            else:
                img = np.array(letterbox_image(img,[image_h,image_w]),dtype = np.float64)
            X_train.append(img)
            Y_train.append(label_name)
            # 读完一个周期后重新开始
            i = (i+1) % n
        # 处理图像
        X_train = preprocess_input(np.array(X_train).reshape(-1,image_h,image_w,3))
        #one-hot
        Y_train = np_utils.to_categorical(np.array(Y_train),num_classes= num_class)   
        yield (X_train, Y_train)
    
#2. 模型搭建
def MmNet(input_shape, output_shape):
    model = Sequential()  # 建立模型
    # 第一层
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 第二层
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # 第三层
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # 全连接层
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    # 全连接层
    model.add(Dense(output_shape, activation='softmax'))
    print("-----------模型摘要----------\n")  # 查看模型摘要
    model.summary()

    return model

#3. 训练模型
def train(model, batch_size):

    model = model   #读取模型
    #定义保存方式，每三代保存一次
    checkpoint_period1 = ModelCheckpoint(
        log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='acc',
        save_weights_only=False,
        period=3
    )
    #学习率下降方式，acc三次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(
        monitor='accuracy',
        patience=3,
        verbose=1
    )
    #val_loss一直不下降意味模型基本训练完毕，停止训练
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=1
    )
    #交叉熵
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])
    #Tebsorboard可视化
    tb_Tensorboard = TensorBoard(log_dir="./model", histogram_freq=1, write_grads=True)
    #开始训练
        # 开始训练
    history = model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size, True),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=generate_arrays_from_file(lines[num_train:], batch_size, False),
            validation_steps=max(1, num_val//batch_size),
            verbose = 1,
            epochs=3,
            initial_epoch=0,
            callbacks=[early_stopping, checkpoint_period1, reduce_lr])
    # history = model.fit(x_train, y_train,
    #           batch_size=64,
    #           epochs = 20,
    #           verbose=0,
    #           validation_data=(x_test,y_test),shuffle = True,
    #           callbacks=([early_stopping, checkpoint_period1, reduce_lr, Tensorboard]),
    # )
    return history, model

#4. 生成测试数据
def test_data(lines):
    # 获取测试集数据总长度
    n = len(lines)
    i = 0   #计数器
    # 获取测试集数据和标签列表
    x_test = []    #测试集数据维度
    y_test = []    #测试集标签维度
    # 遍历测试集数据
    for i in range(n):
        name = lines[i].split(';')[0]   #人脸图像名字xxx_xxx.png
        label_name = (lines[i].split(';')[1]).strip('\n')   #人脸数字标签0-39
        file_name = str(Name.get(int(label_name)))  #对应人名str
        # 从文件中读取图像
        img = Image.open(r".\data\face" +"\\"+ file_name +"\\"+ name)
        img = np.array(letterbox_image(img,[image_h,image_w]),dtype = np.float64)
        x_test.append(img)
        y_test.append(label_name)
        # 读完一个周期后重新开始
        i += 1  #计数器加1
    # 处理图像
    x_test = preprocess_input(np.array(x_test).reshape(-1,image_h,image_w,3))
    #转换为one-hot编码
    y_test = np_utils.to_categorical(np.array(y_test),num_classes= num_class)   
    return x_test, y_test


if __name__ == "__main__":
    #模型保存位置
    log_dir = "./logs/"
    #数据集路径
    path = "./data/face/"
    # 打开数据集的txt
    with open(r".\data\train.txt","r") as f:
        lines = f.readlines()
        
    
    #打乱数据集
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    # 80%用于训练，20%用于测试。
    num_val = int(len(lines)*0.2)   #
    num_train = len(lines) - num_val  #4715
    # #数据初始化
    # x_train, y_train, x_test, y_test = read_image(path, image_w, image_w) #读取数据

    # #数据归一化并转化为one-hot编码
    # x_train = x_train.reshape(x_train.shape[0], image_w, image_h, 3)
    # x_test = x_test.reshape(x_test.shape[0], image_w, image_h, 3)
    # x_train = x_train.astype('float32')
    # x_test = x_test.astype('float32')
    # x_train /= 255
    # x_test /= 255
    # y_train = keras.utils.to_categorical(y_train, num_class)
    # y_test = keras.utils.to_categorical(y_test, num_class)


    # print(x_train.shape) # 训练集数据维度
    # print(y_train.shape) # 训练集标签维度
    # print(x_test.shape) # 测试集数据维度
    # print(y_test.shape) # 测试集标签维度

    #定义模型参数
    input_shape = (image_w, image_h, 3) #输入
    output_shape = num_class    #输出
    #创建AlexNet模型
    model = MmNet(input_shape, output_shape)
    batch_size = 32
    try:
        model = load_model(log_dir + 'easy1.h5')
    except OSError:
        history, model = train(model,batch_size)
    else:
        history, model = train(model,batch_size)
    model.save(log_dir + 'easy1.h5')    #保存训练模型与权重
   
    #打印预测结果
    x_test, y_test = test_data(lines[num_train:])
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

