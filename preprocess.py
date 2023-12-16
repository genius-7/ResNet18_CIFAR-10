from torchvision.datasets import cifar
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import PIL.Image as Image

def datasetDownload():
    cifar.CIFAR10(root='./data', train=True, download=True)

def loadCifar10Batch(folderPath, batchId = 1,mode = 'train'):
    if mode == 'test':
        filePath = os.path.join(folderPath, 'test_batch')
    else :
        filePath = os.path.join(folderPath, 'data_batch_' + str(batchId))
    with open(filePath, 'rb') as batchFile:
        batch = pickle.load(batchFile, encoding = 'latin1')
    imgs = batch['data'].reshape((len(batch['data']),3,32,32))
    labels = batch['labels']
    # print(imgs)
    return np.array(imgs, dtype='float32'), np.array(labels)


class CIFAR10Dataset(Dataset):
    def __init__(self, folderPath='./data/cifar-10-batches-py', mode='train' , transforms = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                        ])):
        if mode == 'train':
            # 加载batch1-batch4作为训练集
            self.imgs, self.labels = loadCifar10Batch(folderPath=folderPath, batchId=1, mode='train')
            for i in range(2, 5):
                imgs_batch, labels_batch = loadCifar10Batch(folderPath=folderPath, batchId=i, mode='train')
                self.imgs, self.labels = np.concatenate([self.imgs, imgs_batch]), np.concatenate(
                    [self.labels, labels_batch])
        elif mode == 'val':
            # 加载batch5作为验证集
            self.imgs, self.labels = loadCifar10Batch(folderPath=folderPath, batchId=5, mode='val')
        elif mode == 'test':
            # 加载测试集
            self.imgs, self.labels = loadCifar10Batch(folderPath=folderPath, mode='test')
        self.transform = transforms

    def __getitem__(self, idx):
        img, label = self.imgs[idx], self.labels[idx]
        img = img.transpose(1, 2, 0)
        img = Image.fromarray(np.uint8(img))
        img = self.transform(img)
        # print(img.shape)
        # print(label)
        return img, label

    def __len__(self):
        return len(self.imgs)
    


if __name__ == '__main__':
    pass
    # cifar数据集下载
    # datasetDownload()

    # cifar数据集加载测试
    # imgsBatch, labelsBatch = loadCifar10Batch(folderPath='./data/cifar-10-batches-py',
    #                                             batchId=1, mode='train')
    # # 打印一下每个batch中X和y的维度
    # print("batch of imgs shape: ", imgsBatch.shape, "batch of labels shape: ", labelsBatch.shape)
    # image,label = imgsBatch[1],labelsBatch[1]
    # plt.figure(figsize=(2, 2))
    # plt.imshow(image.transpose(1,2,0))
    # plt.show()

    # dataset读取
    # trainDataset = CIFAR10Dataset(folderPath='./data/cifar-10-batches-py')
    # valDataset = CIFAR10Dataset(folderPath='./data/cifar-10-batches-py',mode='val')
    # testDataset = CIFAR10Dataset(folderPath='./data/cifar-10-batches-py',mode='test')