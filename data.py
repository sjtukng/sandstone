import os
import numpy as np
import random
import cv2

from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
###################################################
# load data from images
###################################################

color_channel = 1

def load_data(width, height):
    data = np.empty((12302, width, height, color_channel), dtype="float32") ### 29831,964
    label = np.empty((12302,), dtype="uint8")
    traing_sets = os.listdir("./train_aug")
    num_of_class = len(traing_sets)
    k = 0
    for i in range(num_of_class):
        imgs = os.listdir("./train_aug/" + traing_sets[i])
        num_of_imgs = len(imgs)
        for j in range(num_of_imgs):
            img = Image.open("./train_aug/" + traing_sets[i] + "/" + imgs[j])
            gray_img = img.convert("L")
            new_img = gray_img.resize((width, height), Image.BILINEAR)
            arr = np.asarray(new_img, dtype="float32")
            data[k,:,:,0] = arr
            label[k] = traing_sets[i]
            k += 1
    print ("totally " + str(k) + " training images loaded.")
    return data,label


def aug_image():
    traing_sets = os.listdir("./train")
    num_of_class = len(traing_sets)

    for i in range(num_of_class):
        imgs = os.listdir("./train/" + traing_sets[i])
        num_of_imgs = len(imgs)
        for j in range(num_of_imgs):
            img = Image.open("./train/" + traing_sets[i] + "/" + imgs[j])

            datagen = ImageDataGenerator(
                rotation_range=0.1,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest')

            x = img_to_array(img)  # this is a Numpy array with shape (3, width, height)
            x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, width, height)
            k = 0
            for batch in datagen.flow(x,
                                      batch_size = 1,
                                      save_to_dir = 'train_aug/' + traing_sets[i],  # 生成后的图像保存路径
                                      save_prefix = imgs[j][0:imgs[j].rindex(".")],
                                      save_format = 'jpg'):
                k += 1
                if k > 30:
                    break

            print (str(j+1), "images done.")

# randomly sample fragments from the image of h * w pixels
def rand_sample(img, h, w, num):
    rows = img.shape[0]
    cols = img.shape[1]

    for i in range(num):
        r = random.randint(0, rows - h)
        c = random.randint(0, cols - w)
        ri =  img[r:r+h-1, c:c+w-1, :]
        cv2.imwrite('randimg/' + str(30000+i+1) + '.jpg', ri)

img = cv2.imread("data/45degree.bmp", cv2.IMREAD_COLOR)
rand_sample(img, 64, 64, 10000)

