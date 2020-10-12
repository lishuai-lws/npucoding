import torchvision
import torchvision.transforms as transforms
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from sklearn import metrics
import matplotlib.pyplot as plt
import itertools
from sklearn import preprocessing

data_path='../emodb/emo-db_spectrogram'
batch_size=32
epochnum = 50
Dropoutrate=0.3
learning_rate = 0.0001
foldnum=10
is_trainsettest=False
classes=['angry','bored','neutral','happy','fearful','sad','disgusted']
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
ACClist,MAPlist,MARlist,MAFlist=[],[],[],[]
tlosslist,vlosslist=[],[]
def Dataopen(data_path):
    images=[]
    labels=[]
    spectrogrampath = data_path
    filelist = os.listdir(spectrogrampath)
    filelist.sort()
    for file in filelist:
        filepath = os.path.join(spectrogrampath, file)
        imagelist = os.listdir(filepath)
        for image in imagelist:
            images.append(os.path.join(filepath,image))
            labels.append(stof(file[-2]))
    return np.array(images),np.array(labels)

def imagenumcount(i):
    people=[[0,48],[49,106],[107,149],[150,187],[188,242],[243,277],[278,338],[339,407],[408,463],[464,534]]
    filelist=os.listdir(data_path)
    filelist.sort()
    num=[0]
    count=0
    for j in range(people[i][0],people[i][1]+1):
        filepath=os.path.join(data_path,filelist[j])
        imagelist=os.listdir(filepath)
        count+=len(imagelist)
        num.append(count)
    return num

def stof(s):
    S = ['W', 'L', 'N', 'F', 'A', 'T', 'E']
    for i in range(0,7):
        if S[i]==s:
            return i

class Dataset(Dataset):
    def __init__(self,images, labels):
        self.images=images
        self.labels=torch.LongTensor(labels)
        self.transforms=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0,0,0),(255,255,255))
            # transforms.Normalize((57.168066,165.677403,121.896779),(35.441201,31.733191,23.499222)) #标准化RGB
        ]
        )
    def __getitem__(self, item):
        image_path,label=self.images[item],self.labels[item]
        pil_image=Image.open(image_path).convert('RGB')
        data=self.transforms(pil_image)
        return data,label
    def __len__(self):
        return len(self.images)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5,stride=2),  # in_channels=3,out_channels=32,kernel_size=3,stride=2
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(Dropoutrate),
            # 32*63
            nn.Conv2d(32, 48, kernel_size=5,stride=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(Dropoutrate),
            # 64*15
            nn.Conv2d(48, 64, kernel_size=3,stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(Dropoutrate),
        )
        self.classer=nn.Sequential(
            nn.Linear(2240,7)
        )

    def forward(self,x):
        x=self.conv(x)
        x=x.view(x.size()[0],-1)
        x=self.classer(x)
        return x

def train(model,epochnum,images,labels,imagestest,labelstest):
    trainset = Dataset(images, labels)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    epochnum=epochnum
    trainloader=DataLoader(dataset=trainset,
                           batch_size=batch_size,
                           shuffle=True
                           )
    for epoch in range(epochnum):
        # print('Epoch [{}/{}]'.format(epoch+1,epochnum))
        # 训练模型，计算训练集的loss
        runing_loss=0
        for i,(inputs,labels) in enumerate(trainloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            model.zero_grad()
            outputs=model(inputs)
            loss=criterion(outputs,labels)
            runing_loss+=loss.item()
            loss.backward()
            optimizer.step()
        tlosslist.append(runing_loss/(len(trainset)//batch_size+1))
        # print('Epoch [{}/{}] Trainloss:{}'.format(epoch + 1, epochnum,runing_loss/(len(trainset)//batch_size+1)))
        # 计算测试集的损失值
        testset = Dataset(imagestest,labelstest)
        test_loader = DataLoader(dataset=testset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=2
                                 )
        testloss=0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                testnum = labels.size(0)
                output = model(images)
                loss = criterion(output, labels)
                testloss+=loss.item()
        vlosslist.append(testloss / (len(testset) // batch_size + 1))
        print('Epoch [{}/{}] Trainloss:{:.4f} Testloss:{:.4f}'.format(epoch + 1, epochnum,runing_loss/(len(trainset)//batch_size+1),testloss / (len(testset) // batch_size + 1)))

def modeltest(i,model,images,labels):
    ACC = 0
    MAP = 0
    MAR = 0
    MAF = 0
    num = imagenumcount(i)
    pre_labels=[]
    ori_labels=[]
    for j in range(len(num) - 1):
    # for j in range(405,405):
        # print(images[num[j]:num[j+1]])
        testset=Dataset(images[num[j]:num[j+1]],labels[num[j]:num[j+1]])
        test_loader = DataLoader(dataset=testset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=2
                                 )
        with torch.no_grad():
            outputs=[]
            for image,label in test_loader:
                image = image.to(device)
                testnum=label.size(0)
                output = model(image)
                output = output.cpu()
                output= preprocessing.MinMaxScaler().fit_transform(output.numpy().T)
                outputs += output.T.tolist()
            outputs=np.sum(np.array(outputs),axis=0)
            outputs=np.where(outputs==np.max(outputs))
            outputs=list(outputs[0])
            pre_labels.append(outputs[0])
        ori_labels.append(testset.labels[0].tolist())
    np.array(pre_labels)
    print('pre_labels:',pre_labels)
    print('labels:',ori_labels)
    ACC = metrics.accuracy_score(pre_labels,ori_labels,normalize=False)/len(pre_labels)
    MAP = metrics.precision_score(pre_labels,ori_labels, average='macro')
    MAR = metrics.recall_score(pre_labels,ori_labels, average='macro')
    MAF = metrics.f1_score(pre_labels,ori_labels, average='macro')
    ACClist.append(round(100 * ACC , 2))
    MAPlist.append(round(100 * MAP , 2))
    MARlist.append(round(100 * MAR , 2))
    MAFlist.append(round(100 * MAF , 2))
        # ACClist.append(round(100*ACC/Total,2)),MAPlist.append(round(100*MAP/Total,2)),MARlist.append(round(100*MAR/Total,2)),MAFlist.append(round(100*MAF/Total,2))

def modelfoldtest(i,model,x_data, y_data, begin, end):
    x_datatrain=np.delete(x_data, np.arange(begin, end + 1), axis=0)
    y_datatrain=np.delete(y_data, np.arange(begin, end + 1))
    x_datatest=x_data[begin:end + 1]
    y_datatest=y_data[begin:end + 1]
    train(model, epochnum, x_datatrain,y_datatrain,x_datatest, y_datatest)
    if is_trainsettest:
        modeltest(model, x_datatrain, y_datatrain)
    modeltest(i,model, x_datatest, y_datatest)


def outprint():
    if is_trainsettest==False:
        print('ACC:',ACClist)
        print('MAP:',MAPlist)
        print('MAR:',MARlist)
        print('MAF:',MAFlist)
        print('Average ACC: %.2f'% (sum(ACClist)/foldnum))
        print('Average MAP: %.2f' % (sum(MAPlist) / foldnum))
        print('Average MAR: %.2f' % (sum(MARlist) / foldnum))
        print('Average MAF: %.2f'% (sum(MAFlist)/foldnum))
    else:
        print('Train ACC:',ACClist[0:foldnum*2:2])
        print('Test ACC:',ACClist[1:foldnum*2:2])
        print('Train MAP:',MAPlist[0:foldnum*2:2])
        print('Test MAP:',MAPlist[1:foldnum*2:2])
        print('Train MAR:',MARlist[0:foldnum*2:2])
        print('Test MAR:',MARlist[1:foldnum*2:2])
        print('Train MAF:',MAFlist[0:foldnum*2:2])
        print('Test MAF:',MAFlist[1:foldnum*2:2])
        print('Average ACC: %.2f'% (sum(ACClist[1:foldnum*2:2])/foldnum))
        print('Average MAP: %.2f' % (sum(MAPlist[1:foldnum*2:2]) / foldnum))
        print('Average MAR: %.2f' % (sum(MARlist[1:foldnum*2:2]) / foldnum))
        print('Average MAF: %.2f'% (sum(MAFlist[1:foldnum*2:2])/foldnum))

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    '''
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    '''


    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    # # 。。。。。。。。。。。。新增代码开始处。。。。。。。。。。。。。。。。
    # # x,y轴长度一致(问题1解决办法）
    # plt.axis("equal")
    # # x轴处理一下，如果x轴或者y轴两边有空白的话(问题2解决办法）
    # ax = plt.gca()  # 获得当前axis
    # left, right = plt.xlim()  # 获得x轴最大最小值
    # ax.spines['left'].set_position(('data', left))
    # ax.spines['right'].set_position(('data', right))
    # for edge_i in ['top', 'bottom', 'right', 'left']:
    #     ax.spines[edge_i].set_edgecolor("white")
    # # 。。。。。。。。。。。。新增代码结束处。。。。。。。。。。。。。。。。

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        plt.text(j, i, num,
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="white" if num > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('../figure/emodb_dcnn2')
# 更新混淆矩阵
def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix

def finaltest(model,x_data,y_data):
    conf_matrix = torch.zeros(7, 7)
    pre_labels = []
    ori_labels = []
    for i in range(foldnum):
        begin,end=people[i][0],people[i][1]
        images,labels = x_data[begin:end + 1],y_data[begin:end + 1]
        num = imagenumcount(i)
        for j in range(len(num) - 1):
            testset = Dataset(images[num[j]:num[j + 1]], labels[num[j]:num[j + 1]])
            test_loader = DataLoader(dataset=testset,batch_size=batch_size,shuffle=False,num_workers=2 )
            with torch.no_grad():
                outputs = []
                for image, label in test_loader:
                    image = image.to(device)
                    output = model(image)
                    output = output.cpu()
                    outputs += output.tolist()
                outputs = np.sum(np.array(outputs), axis=0)
                outputs = np.where(outputs == np.max(outputs))
                outputs = list(outputs[0])
            pre_labels.append(outputs[0])
            ori_labels.append(testset.labels[0].tolist())
    np.array(pre_labels)
    conf_matrix = confusion_matrix(pre_labels, ori_labels, conf_matrix)
    plot_confusion_matrix(conf_matrix, classes=classes, normalize=False, title='Normalized confusion matrix')

if __name__=='__main__':
    images, labels = Dataopen(data_path)
    people = [
        [0, 232],
        [233, 561],
        [562, 775],
        [776, 939],
        [940, 1231],
        [1232, 1411],
        [1412, 1700],
        [1701, 2059],
        [2060, 2307],
        [2308, 2708]
    ]
    maxacc=0
    for i in range(foldnum):
        print('第',i+1,'次验证')
        model = Net().to(device)
        modelfoldtest(i,model,images, labels,people[i][0],people[i][1])
        if ACClist[i]>maxacc:
            maxacc=ACClist[i]
            model1=model
    finaltest(model1,images,labels)
    outprint()
    print('Finish!')
# nohup python dcnn.py > dcnn.out &