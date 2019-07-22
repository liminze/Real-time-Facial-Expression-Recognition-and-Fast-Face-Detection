from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import cv2

data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True,
                        fill_mode='nearest')

img_path = '/home/prl/PycharmProjects/RealTime_FER/4064_1.png'
img = image.load_img(img_path)
# img = cv2.imread(img_path)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imwrite('1.png',img)
fig1 = plt.figure()
plt.imshow(img)
plt.xticks([])
plt.yticks([])

# 将图片转为数组
x = image.img_to_array(img)
# 扩充一个维度
x = np.expand_dims(x, axis=0)
# 生成图片
gen = data_generator.flow(x, batch_size=1)
# 显示生成的图片
fig = plt.figure()
for i in range(3):
    for j in range(3):
        x_batch = next(gen)
        idx = (3*i) + j
        plt.subplot(3, 3, idx+1)
        plt.imshow(x_batch[0]/256)
        plt.xticks([])
        plt.yticks([])

# fig.savefig('./models/pictures/data.png', dpi=300)
fig1.savefig('./models/pictures/data1.png', dpi=300)
plt.show()
# x_batch.shape
