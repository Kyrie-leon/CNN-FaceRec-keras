import os

Name_label = [] #姓名标签
path = './data/face/'   #数据集文件路径
dir = os.listdir(path)  #列出所有人
label = 0   #设置计数器

#数据写入
with open('./data/train.txt','w') as f:
    for name in dir:
        Name_label.append(name)
        print(Name_label[label])
        after_generate = os.listdir(path +'\\'+ name)
        for image in after_generate:
            if image.endswith(".png"):
                f.write(image + ";" + str(label)+ "\n")
        label += 1
