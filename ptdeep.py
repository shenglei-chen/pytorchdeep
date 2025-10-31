#https://zhuanlan.zhihu.com/p/613526316


import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt

# 定义ToTensor和Normalize的transform
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize((0.5,), (0.5,))

# 定义Compose的transform
transform = transforms.Compose([
    to_tensor,  # 转换为张量
    normalize  # 标准化
])

# 下载数据集
data_train = datasets.MNIST(root="..//data//",
                            transform=transform,
                            train=True,
                            download=True)

data_test = datasets.MNIST(root="..//data//",
                           transform=transform,
                           train=False,
                           download=True)
# 装载数据
data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                batch_size=64,
                                                shuffle=True)

data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               batch_size=64,
                                               shuffle=True)


class MLP(torch.nn.Module):

    def __init__(self, num_i, num_h, num_o):
        super(MLP, self).__init__()

        self.linear1 = torch.nn.Linear(num_i, num_h)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(num_h, num_h)  # 2个隐层
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(num_h, num_o)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x


def train(model):
    # 损失函数，它将网络的输出和目标标签进行比较，并计算它们之间的差异。在训练期间，我们尝试最小化损失函数，以使输出与标签更接近
    cost = torch.nn.CrossEntropyLoss()
    # 优化器的一个实例，用于调整模型参数以最小化损失函数。
    # 使用反向传播算法计算梯度并更新模型的权重。在这里，我们使用Adam优化器来优化模型。model.parameters()提供了要优化的参数。
    optimizer = torch.optim.Adam(model.parameters())
    # 设置迭代次数
    epochs = 2
    for epoch in range(epochs):
        sum_loss = 0
        train_correct = 0
        for data in data_loader_train:
            inputs, labels = data  # inputs 维度：[64,1,28,28]
            #     print(inputs.shape)
            inputs = torch.flatten(inputs, start_dim=1)  # 展平数据，转化为[64,784]
            #     print(inputs.shape)
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = cost(outputs, labels)
            loss.backward()
            optimizer.step()

            _, id = torch.max(outputs.data, 1)
            sum_loss += loss.data
            train_correct += torch.sum(id == labels.data)
        print('[%d/%d] loss:%.3f, correct:%.3f%%, time:%s' %
              (epoch + 1, epochs, sum_loss / len(data_loader_train),
               100 * train_correct / len(data_train),
               time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
    model.train()


# 测试模型
def test(model, test_loader):
    model.eval()
    test_correct = 0
    for data in test_loader:
        inputs, lables = data
        inputs, lables = Variable(inputs).cpu(), Variable(lables).cpu()
        inputs = torch.flatten(inputs, start_dim=1)  # 展并数据
        outputs = model(inputs)
        _, id = torch.max(outputs.data, 1)
        test_correct += torch.sum(id == lables.data)
    print(f'Accuracy on test set: {100 * test_correct / len(data_test):.3f}%')


# 预测模型
def predict(model, data):
    model.eval()
    with torch.no_grad():
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
    return pred


num_i = 28 * 28  # 输入层节点数
num_h = 100  # 隐含层节点数
num_o = 10  # 输出层节点数
batch_size = 64

model = MLP(num_i, num_h, num_o)
train(model)
test(model, data_loader_test)

# 预测图片，这里取测试集前10张图片
for i in range(10):
    # 获取测试数据中的第一张图片
    test_image = data_test[i][0]
    # 展平图片
    test_image = test_image.flatten()
    # 增加一维作为 batch 维度
    test_image = test_image.unsqueeze(0)
    # 显示图片
    plt.imshow(test_image.view(28, 28), cmap='gray')
    plt.show()
    pred = predict(model, test_image)
    print('Prediction:', pred.item())
