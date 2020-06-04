import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cnn_utils
import time


def train(model, learning_rate, num_epochs, mini_batches):
    # 优化器
    optimizer = tf.optimizers.Adam(learning_rate)
    losses = []
    accuracies = []
    
    for i in range(num_epochs):
        accuracy = 0
        avg_loss = 0
        for index, (X_mini_batch, Y_mini_batch) in enumerate(mini_batches):
            with tf.GradientTape() as gt:
                # 正向传播
                #pred = model(X_mini_batch)
                pred = model(X_mini_batch)
                loss = cnn_utils.loss(Y_mini_batch, pred)
                avg_loss += loss / len(mini_batches)
                accuracy += cnn_utils.accuracy(Y_mini_batch, pred)
            # 反向传播
            gradient = gt.gradient(loss, model.trainable_variables)
            # 更新参数
            optimizer.apply_gradients(zip(gradient, model.trainable_variables))
        
        if i % 10 == 0:
            losses.append(avg_loss)
            accuracy = accuracy / 1080
            accuracies.append(accuracy)
            print('epoch %d, loss: %.4f, accuracy: %.4f' %(i, avg_loss, accuracy))
    
    # 绘制损失函数
    plt.plot(losses)
    plt.title('loss function')
    plt.xlabel('per 50 times')
    plt.ylabel('loss')
    plt.show()
    
    # 绘制正确率函数
    plt.plot(accuracies)
    plt.title('accuracy function')
    plt.xlabel('per 50 times')
    plt.ylabel('accuracy')
    plt.show()
    
    return model


def test(model, test_x, test_y):
    pred = model.predict_proba(test_x)
    accuracy = cnn_utils.accuracy(test_y, pred)
    print('test dataset accuracy: %.4f' % (accuracy / test_y.shape[0]))
 

def test_own_images(model):
    for i in range(1, 6):
        url = 'datasets/fingers/%d.png' % i
        img = plt.imread(url)
        img = img.reshape(1, 64, 64, 3)
        prediction = np.squeeze(model.predict_classes(img))
        ans = '错误'
        if prediction == i:
            ans = '正确'
        print('预测结果: %d, %s' % (prediction, ans))
        
        
if __name__ == '__main__':
    # 加载数据集
    train_x, train_y, test_x, test_y = cnn_utils.load_dataset()
    # 对训练集分片
    train_mini_batches = cnn_utils.random_mini_batch(train_x, train_y, mini_batch_size=64)
    # 创建CNN模型
    model = cnn_utils.cnn()
    # 记录训练开始时间
    start_time = time.time()
    # 训练模型
    model = train(model, learning_rate=0.001, num_epochs=1000, mini_batches=train_mini_batches)
    # 记录训练结束时间
    end_time = time.time()
    exhaust_time = end_time - start_time
    print('训练用时 %.2f秒' % exhaust_time)
    # 测试模型
    test(model, test_x, test_y)
    # 测试自己的图片
    test_own_images(model)