import torch
import torch.nn as nn
class Classifier(nn.Module):#分类器模型
    def __init__(self):
        super(Classifier, self).__init__()
        self.layer1=nn.Linear(429,1024)#线性变换
        self.layer2=nn.Linear(1024,512)
        self.layer3=nn.Linear(512,128)
        self.out=nn.Linear(128,39)
        # 你在 forward 里使用了 self.act_fn，但 __init__ 里没有定义
        self.act_fn = nn.ReLU()
    def forward(self, x):
        x = self.layer1(x)
        x = self.act_fn(x)
        x = self.layer2(x)
        x = self.act_fn(x)
        x = self.layer3(x)
        x = self.act_fn(x)
        x = self.out(x)
        return x