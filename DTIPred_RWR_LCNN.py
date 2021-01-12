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
#方阵对角线元素置0并保存
def ZeroMdjx(A):#矩阵对角线置0，同类型节点的矩阵需要考虑对角线是否清零
    for i in range(A.shape[0]):
        A[i][i]=0
    return A
print("与药物相关矩阵置零：")
Md1=ZeroMdjx(Mcsbn)
Md2=ZeroMdjx(Mdibn)
Md3=ZeroMdjx(Mesbn)
print("蛋白相关矩阵对角线置0：")
Mpi=ZeroMdjx(Mpibn)
Mps=ZeroMdjx(Mpsbn)
#对药物和蛋白相关的五个矩阵（对角线置0后）行归一化
def rowNorm(A):
    sum_i = []  # 每行元素的和
    for i in range(A.shape[0]):
        sum_t = 0
        for j in range(A.shape[1]):
            sum_t = sum_t + A[i][j]
        sum_i.append(sum_t)
    M_norm = np.ones((A.shape[0], A.shape[1])) * 0  # 定义新的行归一化矩阵
    print(M_norm.shape)
    for i in range(M_norm.shape[0]):
        for j in range(M_norm.shape[1]):
            if sum_i[i] != 0:
                # print(sum_i[i])
                M_norm[i][j] = A[i][j] / sum_i[i]
            else:
                M_norm[i][j] = 0
   # print(M_norm)
    return M_norm
#逐步对转移概率矩阵进行归一化（逐步细化）
#1.对每个矩阵每行求和并作为构造矩阵的一列，新构造的矩阵大小为Max*5的辅助归一化的矩阵
def Sum_Mrow(A):#对每个矩阵每行求和并作为构造矩阵的一列，新构造的矩阵大小为Max*5的辅助归一化的矩阵
    sum_i = []  # 每行元素的和
    for i in range(A.shape[0]):
        sum_t = 0
        for j in range(A.shape[1]):
            sum_t = sum_t + A[i][j]
        sum_i.append(sum_t)
    M_norm = np.ones((A.shape[0],1)) * 0  # 定义新的行归一化矩阵，每个矩阵转为一列，元素为1和0
    #print(M_norm.shape)
    for i in range(len(sum_i)):
        M_norm[i][0]=sum_i[i]
    #print(M_norm)
    return M_norm
#读取各个归一化的矩阵
#药物
Mrri=np.loadtxt("Ri_norm.txt")
Mrrc=np.loadtxt("Rc_norm.txt")
Mrre=np.loadtxt("Re_norm.txt")
#蛋白
Mtti=np.loadtxt("Ti_norm.txt")
Mttp=np.loadtxt("Tp_norm.txt")
#五倍交叉构造的五个矩阵
print("五倍交叉构造的五个矩阵:")
Mrt1=np.loadtxt("RTi11_norm.txt")
#对每个矩阵行求和后多行*一列：
vRi=Sum_Mrow(Mrri)
vRc=Sum_Mrow(Mrrc)
vRe=Sum_Mrow(Mrre)
vTi=Sum_Mrow(Mtti)
vTp=Sum_Mrow(Mttp)
#同时考虑转置
#五倍交叉构造的五个矩阵
vRTi1=Sum_Mrow(Mrt1)
#print("查看构造的和的列向量的形状及数值")
#print(vRTi1.shape)
#print(vRTi1)
vRTi_tr1=Sum_Mrow(Mrt1.T)#转置
#2.拼接形状为Max*5的矩阵
print("拼接形状为Max*5的辅助归一化矩阵：")
#构造三个零矩阵：
print("构造三个零列向量：")
def ZeroV(A):
    Zm = np.ones((A.shape[0],1)) * 0
    print(Zm.shape)
    return Zm
print("构造三个零矩阵：")
def ZeroM(A):
    Zm = np.ones((A.shape[0],A.shape[1])) * 0
    print(Zm.shape)
    return Zm

Z1=ZeroV(vRi)#708*1
Z2=ZeroV(vTi)#1512*1
print("test开始拼接药物")
#构造参数list
arfList=[1/5,1/5,1/5,4/5,1/5]
arf=0.9
R1 = np.concatenate((vRc,Z1 , Z1, vRTi1,vRTi1), axis = 1)#按列拼接
    #print(D1.shape)
R2 = np.concatenate((Z1,vRi , Z1, vRTi1,vRTi1), axis = 1)
R3 = np.concatenate((Z1,Z1 , vRe, vRTi1,vRTi1), axis = 1)
print("test开始拼接蛋白")
T1=np.concatenate((vRTi_tr1,vRTi_tr1 , vRTi_tr1, vTp,Z2), axis = 1)
T2=np.concatenate((vRTi_tr1,vRTi_tr1 , vRTi_tr1, Z2,vTi), axis = 1)
print("test开始拼接疾病")
tGlobalM = np.concatenate((R1, R2, R3, T1,T2), axis = 0)#按行拼接，Max*5,元素都为0或1或和！=0
#  print(tGlobalM.shape)
#对其中值不为1且不等于0的元素置为1，以对每个矩阵每行求和并作为构造矩阵的一列，新构造的矩阵大小为Max*5的辅助归一化的矩阵
for i in range(tGlobalM.shape[0]):
    for j in range(tGlobalM.shape[1]):
        if tGlobalM[i][j]!=0:
            tGlobalM[i][j]=1
print("此时元素为0和1辅助矩阵构造完毕：")
for i in range(tGlobalM.shape[0]):
    for j in range(tGlobalM.shape[1]):
        tGlobalM[i][j] = tGlobalM[i][j] * arfList[j]
# print(tGlobalM)
# print(tGlobalM.shape)
print("arf辅助矩阵构造完毕，元素为arf！")
#对辅助矩阵进行行归一化
Norm_arfM=rowNorm(tGlobalM)
#开始拼接转移概率矩阵
#拼接转移概率矩阵2495*6
#第一大列
print("构造三个零矩阵：")
RZ1=ZeroM(Mrri)#549*549
TZ2=ZeroM(Mtti)#424*424
print("第一列拼接完毕")
#print(Mrri.shape)
#print(RZ1.shape)
#print(Mrt.T.shape)
#print(Mrd.T.shape)
D1=np.concatenate((Mrri,RZ1 , RZ1,Mrt1.T,Mrt1.T), axis = 0)
D2=np.concatenate((RZ1,Mrrc , RZ1,Mrt1.T,Mrt1.T), axis = 0)
D3=np.concatenate((RZ1,RZ1 ,Mrre ,Mrt1.T,Mrt1.T), axis = 0)
D4=np.concatenate((Mrt1,Mrt1 ,Mrt1 ,Mtti,TZ2,), axis = 0)
D5=np.concatenate((Mrt1,Mrt1 ,Mrt1 ,TZ2,Mttp), axis = 0)
#print(D1.shape)
#print(D2.shape)
#print(D3.shape)
#print(D4.shape)
#print(D5.shape)
print("开始拼接MaxNorM矩阵...")
tDlist=[D1,D2,D3,D4,D5]
tMlist=[]
for k in range(Norm_arfM.shape[1]):  # 共六列0到4
    tM = np.ones((tDlist[k].shape[0], tDlist[k].shape[1])) * 0  # 定义新的行归一化矩阵
    tMlist.append(tM)
    for i in range(tDlist[k].shape[0]):
        tMlist[k][i] = tDlist[k][i] * Norm_arfM[i][k]
MaxNorM = np.concatenate((tMlist[0], tMlist[1], tMlist[2], tMlist[3], tMlist[4]), axis=1)
print("拼接MaxNorM矩阵完毕")
print(MaxNorM.shape)
#开始随机游走部分：
print("1.构造药物P0种子单位矩阵，但元素为1/3:")
Ip0M=np.eye(708)*(1/3)
print("2.构造蛋白P1种子节点，但元素为1/2:")
Ip1M=np.eye(1512)*(1/2)
#print(Ip0M)
MzeroIp0 = np.ones((1512*2, 708)) * 0  # 定义新的行归一化矩阵
MzeroIp1 = np.ones((708*3, 1512)) * 0
MP0=np.concatenate((Ip0M,Ip0M ,Ip0M,MzeroIp0), axis = 0)
MP1=np.concatenate((MzeroIp1,Ip1M,Ip1M), axis = 0)
print("708个种子P0所组成的矩阵MP0完毕：")
print(MP0.shape)
#构造矩阵存储结果向量
M_reslts0 = np.ones((5148, 708)) * 0 #549个药物，每个向量为2495维
M_reslts1 = np.ones((5148, 1512)) * 0 #424个蛋白，每个向量为2495维
print("执行随机游走算法：")
cycle=30#，每轮进行30次循环
for r in range(708):#共708个药物
    vPk=MP0[:,r]
    print("计算第{}个药物的结果向量：".format(r+1))
    for i in range(cycle):
     old_vPk=vPk
     vPk=(1-arf)*np.dot(MaxNorM.T,vPk)+arf*MP0[:,r]
     vd=vPk-old_vPk
     y = np.linalg.norm(vd)
     print("迭代次数：")
     print(i)
     print("第{0}个药物的第{1}次迭代l2.norm值：".format(r,i))
     print(y)
     if y<(1E-10):
      break
     M_reslts0[:,r]=vPk
print("药物随机游走结束！")

for r in range(1512):#共708个药物
    vPk=MP1[:,r]
    print("计算第{}个药物的结果向量：".format(r+1))
    for i in range(cycle):
     old_vPk=vPk
     vPk=(1-arf)*np.dot(MaxNorM.T,vPk)+arf*MP1[:,r]
     vd=vPk-old_vPk
     y = np.linalg.norm(vd)
     print("迭代次数：")
     print(i)
     print("第{0}个蛋白的第{1}次迭代l2.norm值：".format(r,i))
     print(y)
     if y<(1E-10):
      break
     M_reslts1[:,r]=vPk
print("蛋白随机游走结束！")
#LCNN
#统计运行时间
starttime = datetime.datetime.now()
#加载特征矩阵
def load_data(id, drug_protein, M_reslts0,M_reslts1, BATCH_SIZE):
    x = []
    y = []
    for j in range(id.shape[0]):
        temp_save = []
        x_A = int(id[j][0])
        y_A = int(id[j][1])
        row_1 = np.concatenate((M_reslts0.T[x_A], drug_protein[x_A]),axis=0)
        row_2 = np.concatenate((M_reslts1.T[x_A], drug_protein[x_A]), axis=0)
        temp_save.append(row_1)
        temp_save.append(row_2)
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

train_loader = load_data(train_id, new_drug_protein, M_reslts0,M_reslts1  , 4)
print(train_loader)
test_loader = load_data(test_id, drug_protein, M_reslts0,M_reslts1 ,1)
print(test_loader)
#CNN实例化
class Dual_Cnn(nn.Module):
    def __init__(self):
        super(Dual_Cnn, self).__init__()
        self.conv1 = nn.Sequential(                      # input shape (1, 2,5148)
            nn.Conv2d(
                in_channels=1,                           # input height
                out_channels=16,                         # n_filters
                kernel_size=(3,3),                      #filter size
                stride=1,                                # filter movement/step（16,2,5148）
                padding=(1,1)
                               ),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(1,2)),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(3,4),                     #改变为filters_shape=(3,4)
                stride=1,
                padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(1,2)),
        )

        self.out = nn.Linear(32*2*1286, 2)  # 全连接层
    def forward(self, x):

            x = self.conv1(x)
            x = self.conv2(x)

            x = x.view(x.size(0), -1)
            out = self.out(x)

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
    o=np.vstack((o,right_test_out.detach().cpu().numpy()))

print('cor_num:{}, Test Auc: {:.6f}'.format(num_cor, test_acc / len(test_loader)))
print(type(right_test_out))
endtime = datetime.datetime.now()
print((endtime - starttime))

