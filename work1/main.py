import os
import numpy as np
import pandas as pd
import torch
from My_model import Classifier
from torch.utils.data import Dataset, DataLoader
# 数据根目录，下面会从该目录读取训练集、标签和测试集数据
data_root='./Data/timit_11/'
# 加载训练特征数据，形状通常为 [样本数, 特征维度]
train=np.load(data_root+'train_11.npy')
# 加载训练标签，和训练特征一一对应
train_label=np.load(data_root+'train_label_11.npy')
# 加载测试特征数据，当前脚本中暂未直接使用
test=np.load(data_root+'test_11.npy')
# 自定义数据集类：把 numpy 数据封装成 PyTorch 可直接读取的 Dataset
class MyDataset(Dataset):
    def __init__(self, X, y=None):
        # 将 numpy 特征数组转换为 float32 张量，便于模型训练
        self.data=torch.from_numpy(X).float()
        if y is not None:
            # 标签转为整型，确保可以正确转换为分类任务所需的 LongTensor
            y=y.astype(int)
            self.label=torch.LongTensor(y)
            '''
            因为交叉熵损失函数要求标签是 LongTensor 类型，
            所以先将 y 转为整数类型，再转成 LongTensor。
            '''
        else:
            # 测试集或无标签数据时，标签置空
            self.label=None
    def __getitem__(self, index):
        # 按索引返回单个样本；有标签时返回 (特征, 标签)，无标签时只返回特征
        if self.label is not None:
            return self.data[index], self.label[index]
        else:
            return self.data[index]
    def __len__(self):
        # 返回数据集样本数量，供 DataLoader 使用
        return len(self.data)
# 将原始训练集划分为训练集和验证集
# VAL_RATIO 表示验证集占总数据的比例，这里设为 0.2，表示 20% 作为验证集
VAL_RATIO = 0.2

# 计算训练集的样本数量：取前 80% 的样本作为训练集，后 20% 作为验证集
percent = int(train.shape[0] * (1 - VAL_RATIO))

# 按比例切分训练特征与标签
train_x=train[:percent] # 前 percent 条样本作为训练集特征
train_y=train_label[:percent] # 前 percent 条样本作为训练集标签
val_x=train[percent:] # 剩余样本作为验证集特征
val_y=train_label[percent:] # 剩余样本作为验证集标签



# 设置每个 batch 的大小
BATCH_SIZE = 64
from torch.utils.data import DataLoader

# 分别构造训练集和验证集 Dataset
train_set=MyDataset(train_x, train_y)
val_set=MyDataset(val_x, val_y)

# 用 DataLoader 包装数据集：训练集打乱，验证集不打乱
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

# 清除已经切分完成后不再需要的中间变量，释放内存
import gc
del train, train_label, train_x, train_y, val_x, val_y
gc.collect()


# 创建模型对象，准备进入训练流程
# 获取当前训练设备：有 GPU 就用 GPU，否则用 CPU
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

# 固定随机种子，保证结果尽量可复现
# 这个函数不会自动生效，后面需要显式调用
# 固定随机种子，保证结果尽量可复现
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
set_seed(same_seed)
device=get_device()
print(f'正在使用{device} 进行训练.')
# 设置训练轮数与学习率
num_epoch = 20
learning_rate = 0.0001
# 设置模型保存路径
model_save_path = './model.pth'
model=Classifier().to(device)
criterion=torch.nn.CrossEntropyLoss()# 分类任务常用的交叉熵损失函数
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)# Adam 优化器

# 开始训练
best_acc=0.0# 当前记录的最佳验证集准确率
for epoch in range(num_epoch):
    # 每个 epoch 开始前重置训练统计量
    train_loss = 0.0
    train_acc = 0.0

    model.train()# 切换到训练模式，启用 Dropout/BN 等训练行为
    for inputs, labels in train_loader:
        # 从数据加载器中取出一个 batch 的输入和标签
        inputs, labels = inputs.to(device), labels.to(device)

        # 先清空梯度，避免梯度累积
        optimizer.zero_grad()

        # 前向传播，得到模型输出 logits
        outputs = model(inputs)

        # 计算当前 batch 的损失
        batch_loss = criterion(outputs, labels)

        # 反向传播计算梯度
        batch_loss.backward()

        # 更新模型参数
        optimizer.step()

        # 统计当前 batch 预测正确的样本数
        train_pred = torch.max(outputs, 1).indices
        train_acc += (train_pred.cpu() == labels.cpu()).sum().item()

        # 累积训练损失
        train_loss += batch_loss.item()

    # 计算训练集平均指标
    train_acc = train_acc / len(train_set)
    train_loss = train_loss / len(train_loader)

    # 当验证集存在时，执行验证流程
    if len(val_set) > 0:
        model.eval()# 切换到评估模式，关闭训练时特有行为
        val_loss = 0.0
        val_acc = 0.0

        with torch.no_grad():# 验证阶段不计算梯度，减少显存占用
            for inputs, labels in val_loader:
                # 读取验证 batch
                inputs, labels = inputs.to(device), labels.to(device)

                # 验证前向传播
                outputs = model(inputs)

                # 计算验证损失
                batch_loss = criterion(outputs, labels)

                # 取出预测类别
                val_pred = torch.max(outputs, 1).indices

                # 统计验证集正确样本数
                val_acc += (val_pred.cpu() == labels.cpu()).sum().item()

                # 累积验证损失
                val_loss += batch_loss.item()

        # 计算验证集平均指标
        val_acc = val_acc / len(val_set)
        val_loss = val_loss / len(val_loader)

        # 打印该 epoch 的训练与验证结果
        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} Loss: {:3.6f}'.format(
            epoch + 1, num_epoch, train_acc, train_loss, val_acc, val_loss
        ))

        # 如果验证集表现更好，就保存当前模型参数
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print('saving model with acc {:.3f}'.format(best_acc))
    else:
        # 如果没有验证集，只打印训练结果并在最后保存模型
        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
            epoch + 1, num_epoch, train_acc, train_loss
        ))


# 如果没有验证集，则在训练结束后保存最终模型
if len(val_set) == 0:
    torch.save(model.state_dict(), model_save_path)
    print('saving model at last epoch')