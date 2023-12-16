import os
import torch.nn.functional as F
import torch.optim as opt
import torch.nn as nn
from model import ResNet18
import torch
from preprocess import CIFAR10Dataset
from tqdm import tqdm
import pandas as pd
import torchvision.transforms as transforms

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
    ])
    lr = 0.1
    batch_size = 64
    epoch = 100
    trainDataset = CIFAR10Dataset(folderPath='./data/cifar-10-batches-py', mode='train' , transforms=transform)
    valDataset = CIFAR10Dataset(folderPath='./data/cifar-10-batches-py', mode='val', transforms=transform)
    testDataset = CIFAR10Dataset(folderPath='./data/cifar-10-batches-py', mode='test')
    trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
    valLoader = torch.utils.data.DataLoader(valDataset, batch_size=batch_size, shuffle=False)
    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    net = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = opt.Adam(net.parameters(), lr=lr)
    # optimizer = opt.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4) #优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

    best_acc = 0.0
    print("start training...")

    training_info = pd.DataFrame(columns=['Epoch', 'Train Loss', 'Train Acc', 'Val Acc'])
    for epochId in range(epoch):
        if epochId <= 49 :
            lr = 0.1
        else :
            lr = 0.01
        # print("epoch: ", epochId + 1)
        running_loss = 0.0
        correct = 0.0
        total = 0.0
        net.train()

        train_loss = 0.0
        val_loss = 0.0
        train_acc = 0.0
        val_acc = 0.0 
        # Use tqdm for the training loop
        tqdm_train = tqdm(enumerate(trainLoader, 0), desc=f'Epoch {epochId + 1}/{epoch} - Train', total=len(trainLoader))

        for i, data in tqdm_train:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            labels = labels.long()
            # print(labels)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            # print(torch.max(outputs.data, 1))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # print(correct,'/',total)
            # Use tqdm.set_postfix to display the progress within the tqdm loop
            tqdm_train.set_postfix(train_loss=running_loss / (i + 1), train_acc=100 * correct / total)
            train_loss , train_acc = running_loss / (i + 1) , 100 * correct / total
        tqdm_train.close()

        print("start validating...")
        with torch.no_grad():
            correct = 0.0
            running_loss = 0.0
            total = 0.0
            tqdm_val = tqdm(enumerate(valLoader,0), desc=f'Epoch {epochId + 1}/{epoch} - Validation', total=len(valLoader))
            for i,data in tqdm_val:
                net.eval()
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                labels = labels.long()
                loss = criterion(outputs,labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                tqdm_val.set_postfix(val_loss=running_loss / (i + 1),val_acc=100 * correct / total)
                val_loss , val_acc = running_loss / (i + 1) , 100 * correct / total
            tqdm_val.close()

            # Append the information to the DataFrame
            training_info = training_info.append({'Epoch': epochId + 1,
                                                  'Train Loss': train_loss,
                                                  'Train Acc': train_acc,
                                                  'Val Loss': val_loss,
                                                  'Val Acc': val_acc}, ignore_index=True)
            if not os.path.exists('./model'):
                os.makedirs('./model')
            if (val_acc) > best_acc:
                best_acc = val_acc
                print("best acc: %.3f" % best_acc)
                torch.save(net.state_dict(), './model/ResNet18.pth')
            training_info.to_csv('training_info.csv', index=False)
    
    print("training finished!")
    

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)
    batch_size = 64
    testDataset = CIFAR10Dataset(folderPath='./data/cifar-10-batches-py',mode='test')
    testLoader = torch.utils.data.DataLoader(testDataset,batch_size=batch_size,shuffle=True)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship','truck')
    net = ResNet18().to(device)
    net.load_state_dict(torch.load('./model/ResNet18.pth'))
    with torch.no_grad():
        correct = 0
        total = 0
        net.eval()
        tqdm_test = tqdm(testLoader, desc='Testing', total=len(testLoader))
        for data in tqdm_test:
            inputs,labels = data
            inputs,labels = inputs.to(device),labels.to(device)
            outputs = net(inputs)
            _,predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            tqdm_test.set_postfix(test_acc=100 * correct / total)
        tqdm_test.close()
        print("test acc: %.3f" % (100 * correct / total))

if __name__ == '__main__':
    train()
    # test()