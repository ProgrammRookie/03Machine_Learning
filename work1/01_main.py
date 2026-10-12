import os
import numpy as np
import pandas as pd
import torch
from My_model import Classifier
from torch.utils.data import Dataset, DataLoader
data_root='./Data/timit_11/'
train=np.load(data_root+'train_11.npy')
train_label=np.load(data_root+'train_label_11.npy')
test=np.load(data_root+'test_11.npy')
#创建数据集
class MyDataset(Dataset):
    def __init__(self, X, y=None):
        self.data=torch.from_numpy(X).float()#将numpy数组转换为torch张量
        if y is not None:
            y = y.astype(int)
            self.label=torch.LongTensor(y)#
            '''
            因为损失函数要求标签为LongTensor类型，
            而LongTensor没法直接从普通的数据类型解析接受，所以先将y转换为np.int类型,
            再转换为LongTensor。
            '''
        else:
            self.label=None
    def __getitem__(self, index):
        if self.label is not None:
            return self.data[index], self.label[index]
        else:
            return self.data[index]
    def __len__(self):
        return len(self.data)
# 划分训练集和验证集
# VAL_RATIO 表示验证集占总数据的比例，这里设为 0.2，表示 20% 作为验证集
VAL_RATIO = 0.2

# 计算训练集的样本数量：取前 80% 的样本作为训练集，后 20% 作为验证集
percent = int(train.shape[0] * (1 - VAL_RATIO))

#将数据按比例切分成训练集和验证集
train_x=train[:percent] #表示取前 percent 条样本作为训练集特征
train_y=train_label[:percent] #表示取前 percent 条样本对应的训练标签
val_x=train[percent:] #表示取后面剩余样本作为验证集特征
val_y=train_label[percent:] #表示取后面剩余样本对应的验证标签



# 创建数据加载器
# 这里的 DataLoader 会把数据集按 batch 组织起来，训练时更方便
BATCH_SIZE = 64
from torch.utils.data import DataLoader

# 用训练集和验证集分别创建 Dataset 对象
train_set=MyDataset(train_x, train_y)
val_set=MyDataset(val_x, val_y)

# 用 DataLoader 包装数据集，batch_size=64 表示每次取 64 个样本
# shuffle=True 表示训练集每个 epoch 之前打乱顺序，提升训练稳定性
# 验证集一般不需要打乱，shuffle=False
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

#清除不需要的变量，释放内存
import gc
del train, train_label, train_x, train_y, val_x, val_y
gc.collect()


#创建模型，准备训练
model = Classifier()
#训练
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'
#固定随机种子，以保证结果可复现
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seed = 42
device=get_device()
print(f'正在使用{device} 进行训练.')
#设置训练的轮次
num_epoch = 20
learning_rate = 0.0001
#设置保存模型的路径
model_save_path = './model.pth'
model=Classifier().to(device)
criterion=torch.nn.CrossEntropyLoss()#交叉熵损失函数
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)#Adam优化器

#开始训练
best_acc=0.0#最好的验证集准确率
for epoch in range(num_epoch):
   train_loss = 0.0
   train_acc= 0.0
   val_loss = 0.0
   val_acc = 0.0
   model.train()#设置模型为训练模式
   for i, data in enumerate(train_loader):
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    batch_loss = criterion(outputs, labels)
    train_pred = torch.max(outputs.data, 1)#使用relu函数来激活输出层的结果
    train_acc += (train_pred.cpu() == labels.cpu()).sum().item()#找到最大概率对应的类别下标
    train_loss += batch_loss.item()
    print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
         epoch + 1, num_epoch, train_acc/len(train_set), train_loss/len(train_loader), val_acc/len(val_set), val_loss/len(val_loader)
    ))
    batch_loss.backward()#反向传播计算梯度
    optimizer.step()#更新参数

    train_acc+=(train_pred.cpu() == labels.cpu()).sum().item()
    train_loss+=batch_loss.item()

if len(val_set) > 0:
    model.eval()#设置模型为评估模式
    with torch.no_grad():#在验证集上不需要计算梯度
        for i, data in enumerate(val_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            batch_loss = criterion(outputs, labels)
            val_pred = torch.max(outputs.data, 1)
            val_acc += (val_pred.cpu() == labels.cpu()).sum().item()
            val_loss += batch_loss.item()
        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
            epoch + 1, num_epoch, train_acc/len(train_set), train_loss/len(train_loader), val_acc/len(val_set), val_loss/len(val_loader)
        ))

    #如果模型有提升，就在当前轮次保存一个检查点
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), model_save_path)
        print('saving model with acc {:.3f}'.format(best_acc/len(val_set)))
    else:
        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
            epoch + 1, num_epoch, train_acc/len(train_set), train_loss/len(train_loader)
        ))

 #如果不进行验证，就保存最后一个轮次的模型
if len(val_set) == 0:
    torch.save(model.state_dict(), model_save_path)
    print('saving model at last epoch')