import numpy as np


# 加载训练数据
training_images = np.load("training-images.npy")
training_labels = np.load("training-labels.npy")
training_data = [(np.reshape(x,(28*28,1)),y) for x,y in zip(training_images, training_labels)]
# 加载测试数据
test_images = np.load("test-images.npy")
test_labels = np.load("test-labels.npy")
test_data = [(np.reshape(x,(28*28,1)),y) for x,y in zip(test_images,test_labels)]
