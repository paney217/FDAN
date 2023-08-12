import cv2
import numpy as np
import matplotlib.pyplot as plt


def GMM(img):
    print(img.shape)
    # 将一个像素点的rgb值作为一个单元处理
    data = img.reshape((-1, 3))
    print(data.shape)
    # 转换数据类型
    data = np.float32(data)
    # 生成模型
    em = cv2.ml.EM_create()
    # 设置参数，将像素分成num个类别
    num = 3
    em.setClustersNumber(num)
    em.setCovarianceMatrixType(cv2.ml.EM_COV_MAT_GENERIC)  # 默认
    # 训练返回的第三个元素包含了预测分类标签
    best = em.trainEM(data)[2]
    # 筛选出面积最大的一个分类（即认为背景是面积最大的）
    index = 0
    length = len(data[best.ravel() == 0])
    for i in range(0, num):
        if len(data[best.ravel() == i]) > length:
            length = len(data[best.ravel() == i])
            index = i
    # 设置为绿色
    data[best.ravel() == index] = (0, 255, 0)
    # 将结果转换为图片需要的格式
    data = np.uint8(data)
    oi = data.reshape(img.shape)
    # show('img',img)
    # show('res',oi)
    cv2.imshow('img', img)
    cv2.imshow('res', oi)
    cv2.waitKey()


if __name__ == '__main__':
    img = cv2.imread("/home/cwh/data/cwhtest/baby.jpg")
    #cv2.imshow('img', img)
    #cv2.waitKey()
    GMM(img)
