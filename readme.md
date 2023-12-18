# ResNet18 CIFAR-10数据集训练及测试

CIFAR-10数据集，共60000张图片，大小32*32。

选用40000张做训练集，10000张当验证集，10000张当测试集。

## 1、下载数据集

运行preprocess.py

```python
python preprocess.py
```

如果你想查看CIFAR-10中的图片数据具体是什么，可以使用loadCifar10Bathch进行测试。

## 2、训练

运行train.py。选用交叉熵作为损失函数，Adam作为优化器。

```python
python train.py
```

在train.py中已经写好了训练、验证和测试的过程。

每一轮训练结束之后会向training_info.csv文件中写入train_loss,val_loss,train_acc,val_acc等。方便训练之后进行查看。

## 3、测试

修改train.py

```python
if __name__ == '__main__':
    # train()
    test()
```

运行train.py

```python
python train.py
```

## 4、训练结果可视化

运行result_visualization.py

```python
python result_visualization.py
```

该py文件可以绘制train_loss,val_loss,train_acc,val_acc，并标记出的train_acc,val_acc最大值，train_loss,val_loss的最小值。
