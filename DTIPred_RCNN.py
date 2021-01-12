import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
from sklearn.metrics import auc
#统计运行时间
starttime = datetime.datetime.now()
#加载特征矩阵
def load_data(id, drug_protein, r_sim1,r_sim2,r_sim3, p_sim1,p_sim2, BATCH_SIZE):
    x = []
    y = []
    for j in range(id.shape[0]):
        temp_save = []
        x_A = int(id[j][0])
        y_A = int(id[j][1])
        row_1 = np.concatenate((r_sim1[x_A], drug_protein[x_A]),axis=0)
        row_2 = np.concatenate((r_sim2[x_A], drug_protein[x_A]), axis=0)
        row_3 = np.concatenate((r_sim3[x_A],drug_protein[x_A]), axis=0)
        row_4 = np.concatenate((drug_protein.T[y_A],p_sim1[y_A]), axis=0)
        row_5 = np.concatenate((drug_protein.T[y_A], p_sim2[y_A]), axis=0)

        temp_save.append(row_1)
        temp_save.append(row_2)
        temp_save.append(row_3)
        temp_save.append(row_4)
        temp_save.append(row_5)
        # 寻找坐标的值
        label = drug_protein[[x_A], [y_A]]
        x.append(np.array([temp_save]))
        y.append(label)
    x = torch.FloatTensor(x)
    print(x.size())
    y = torch.LongTensor(np.array(y))
    # y = torch.from_numpy(np.array(y)).long()
    torch_dataset = Data.TensorDataset(x, y)
    data2_loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=True
    )
    return data2_loader

train_loader = load_data(train_id, new_drug_protein, Mcsbn,Mdibn ,Mesbn ,Mpsbn,Mpibn , 4)
print(train_loader)
test_loader = load_data(test_id, drug_protein,  Mcsbn,Mdibn ,Mesbn ,Mpsbn,Mpibn ,1)
print(test_loader)
#CNN实例化
class Dual_Cnn(nn.Module):
    def __init__(self):
        super(Dual_Cnn, self).__init__()
        self.conv1 = nn.Sequential(                      # input shape (1, 5,708+1512=2220)
            nn.Conv2d(
                in_channels=1,                           # input height
                out_channels=16,                         # n_filters
                kernel_size=(3,3),                      #filter size
                stride=1                                # filter movement/step（16,3,2218）
                #padding=(1,1)
                               ),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(1,2)),             #16,3,1109
        )
        self.conv2 = nn.Sequential(                      #16,3,1109
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(3,4),                     #改变为filters_shape=(3,4)
                stride=1,
                padding=1),  # 每个卷积层有二十个filters  #16，5，1111-》32,3,1108
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(1,2)),             #32，3，554
        )

        self.out = nn.Linear(32*3*554, 2)  # 全连接层(自行计算32*3*554 -> 2)
    def forward(self, x):
        #attention
            x = self.conv1(x)
            x = self.conv2(x)
            #x = self.conv3(x)
            x = x.view(x.size(0), -1)
            out = self.out(x)
            #output = F.softmax(self.out(x), dim=1)
            return out
cnn=Dual_Cnn()

#cnn=CNN()
if torch.cuda.is_available():
   cnn.cuda()
optimizer=torch.optim.Adam(cnn.parameters(),lr=0.0005)    #定义优化方式
loss_func=nn.CrossEntropyLoss()                           #定义损失函数

#开始训练
for epoch in range(120):
    train_loss=0
    train_acc=0
    for step, (x, train_label) in enumerate(train_loader):
        b_x = Variable(x).cuda()
        train_label=Variable(train_label.squeeze(1)).cuda()
        out=cnn(b_x)
        loss = loss_func(out,train_label)      # 计算损失函数
        optimizer.zero_grad()                  # 梯度清零
        loss.backward()                        # 反向传播
        optimizer.step()                       # 梯度优化
        train_loss+=loss.data[0]

        #计算准确率
        _,pred=out.max(1)
        num_correct=(pred==train_label).sum().data[0]
        num_correct=num_correct.cpu().numpy()
        acc=num_correct/b_x.shape[0]
        train_acc+=acc
        if step % 100 == 0:        #每100步显示一次
            print('epoch: ', epoch, '| train loss: %.8f' % loss.data.cpu().numpy())
    print('Epoch: {}, Train Loss: {:.8f}, Train Acc: {:.6f}'
          .format(epoch, train_loss / len(train_loader), train_acc / len(train_loader)))

#测试集
cnn.eval()
test_acc=0
num_cor=0
o=np.zeros((0,2))
for test_x,test_label in test_loader:
    test_x=Variable(test_x).cuda()
    test_label = Variable(test_label.squeeze(1)).cuda()
    right_test_out = cnn(test_x)
    right_test_out=F.softmax(right_test_out,dim=1)
    #计算准确率
    _,pred_y = right_test_out.max(1)
    num_correct = (pred_y == test_label).sum().data[0]
    num_correct = num_correct.cpu().numpy()
    acc = num_correct / test_x.shape[0]
    test_acc += acc
    num_cor += num_correct
  # print(num_correct)
    o=np.vstack((o,right_test_out.detach().cpu().numpy()))

print('cor_num:{}, Test Auc: {:.6f}'.format(num_cor, test_acc / len(test_loader)))
print(type(right_test_out))
endtime = datetime.datetime.now()
print((endtime - starttime))


