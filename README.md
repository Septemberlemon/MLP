# MLP

这是一个用来实现MNIST手写数字识别的最简单的神经网络

依赖的第三方库是numpy和matplotlib，并没有使用深度学习框架。

要训练模型，自定义train.py中的参数，然后运行train.py即可。

要用训练后的模型做预测，运行predict.py即可，它会读取checkpoints中的一个已经训练好的模型，查看其在训练集和测试集上的正确率，然后随机从测试集中选取一个图片进行预测，预测结果将使用matplotlib展示。
