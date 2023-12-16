import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('./training_info.csv')

# 提取数据列
epoch = df['Epoch']
train_loss = df['Train Loss']
train_acc = df['Train Acc']
val_loss = df['Val Loss']
val_acc = df['Val Acc']

# 绘制训练集和验证集的损失图
plt.figure(figsize=(10, 5))
plt.plot(epoch, train_loss, label='train_loss')
plt.plot(epoch, val_loss, label='val_loss')

# 标记最小值点
min_train_loss_idx = train_loss.idxmin()
min_train_loss = train_loss[min_train_loss_idx]
plt.text(epoch[min_train_loss_idx], min_train_loss, f'Min: {min_train_loss:.4f}',
         verticalalignment='top', horizontalalignment='right', color='blue')

min_val_loss_idx = val_loss.idxmin()
min_val_loss = val_loss[min_val_loss_idx]
plt.text(epoch[min_val_loss_idx], min_val_loss, f'Min: {min_val_loss:.4f}',
         verticalalignment='top', horizontalalignment='right', color='orange')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Val Loss')
plt.legend()
plt.grid(True)
plt.savefig('train_val_loss.png')
# plt.show()

# 绘制训练集和验证集的IOU图
plt.figure(figsize=(10, 5))
plt.plot(epoch, train_acc, label='train_acc')
plt.plot(epoch, val_acc, label='val_acc')

# 标记最大值点
max_train_acc_idx = train_acc.idxmax()
max_train_acc = train_acc[max_train_acc_idx]
plt.text(epoch[max_train_acc_idx], max_train_acc, f'Max: {max_train_acc:.4f}',
         verticalalignment='top', horizontalalignment='right', color='blue')

max_val_acc_idx = val_acc.idxmax()
max_val_acc = val_acc[max_val_acc_idx]
plt.text(epoch[max_val_acc_idx], max_val_acc, f'Max: {max_val_acc:.4f}',
         verticalalignment='bottom', horizontalalignment='right', color='orange')

plt.xlabel('Epoch')
plt.ylabel('IOU')
plt.title('Train and Val Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('train_val_acc.png')
plt.show()