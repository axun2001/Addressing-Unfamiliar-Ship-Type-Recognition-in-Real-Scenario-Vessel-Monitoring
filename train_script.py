import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import resnet34,resnet50,resnet101
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import time
from torchsummary import summary
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 设置超参数
lr = 0.001
embedding_dim = 2048
Margin = 0.875
batch_size = 32
num_epochs = 30

import torch.nn as nn
import torchvision.models as models

import os
from PIL import Image

import csv

def train(net, dataloader, criterion, optimizer, device):
    net.train()
    running_loss = 0.0
    losses = []
    pbar = tqdm(dataloader)
    for i, (input1, input2, input3) in enumerate(pbar):
        input1 = input1.to(device)
        input2 = input2.to(device)
        input3 = input3.to(device)
        anchor, positive, negative = net(input1, input2, input3)
        loss = criterion(anchor, positive, negative)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        losses.append(loss.item())
        pbar.set_description(f"Epoch {epoch + 1}, Loss {running_loss / (i + 1):.4f}")

    return running_loss / len(dataloader)

# 加载数据集
transform = transforms.Compose([
    transforms.Resize((224, 224)),#调整图像大小为 (224, 224)
    transforms.ToTensor(),#转换为张量
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])#进行归一化，使用均值 [0.5, 0.5, 0.5] 和标准差 [0.5, 0.5, 0.5]
])

train_dataset = MyDataset(data_dir='D:/PyTorch/resnet34-ship/Ship/train',transform=transform)
print(train_dataset.get_num_triplets())  # 输出生成的三元组数量

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print(len(train_dataset))

# 定义模型、损失函数和优化器
net = Siamese(embedding_dim=embedding_dim)
criterion = TripletLoss(margin=Margin)
optimizer = torch.optim.ASGD(net.parameters(), lr=lr)
optimizer_name = str(type(optimizer)).split(".")[-1][:-2]
print(optimizer_name)
print(Margin)
# 将模型移动到GPU上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
net.to(device)
criterion.to(device)



best_acc = 0.0
# 训练模型
for epoch in range(num_epochs):
    net.train()
    start_time = time.time()
    train_loss = train(net, train_dataloader, criterion, optimizer, device)

    print(f'| Epoch: {epoch+1}')
    print(f'| Train Loss: {train_loss:.5f}')

    query_dir = 'D:/PyTorch/resnet34-ship/val1'

    net.eval()
    support_dir_5 = 'D:/PyTorch/OSL-pt/support-cl-5'
    support_dataset_5 = []
    for class_name in os.listdir(support_dir_5):
        class_dir = os.path.join(support_dir_5, class_name)
        if os.path.isdir(class_dir):
            support_images_5 = []
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                support_image_5 = Image.open(image_path).convert('RGB')
                support_tensor_5 = transform(support_image_5).unsqueeze(0)
                support_tensor_5 = support_tensor_5.to('cuda')
                support_images_5.append(support_tensor_5)
            # 计算支持集类别平均特征向量
            support_embeddings_5 = [net.forward_once(support_image_5).detach().cpu().numpy() for support_image_5 in
                                  support_images_5]
            support_embedding_5 = np.mean(support_embeddings_5, axis=0)
            support_dataset_5.append((class_name, support_embedding_5))


    # 遍历 query 目录下的所有图像
    correct_num_5 = 0  # 记录预测正确的数量
    total_num = 0  # 记录查询集下所有图片的数量
    for class_name in os.listdir(query_dir):
        class_dir = os.path.join(query_dir, class_name)
        if os.path.isdir(class_dir):
            acc_1 = 0.0

            for image_name in os.listdir(class_dir):
                total_num += 1  # 查询集下所有图片数量加一
                image_path = os.path.join(class_dir, image_name)
                query_image = Image.open(image_path).convert('RGB')
                query_tensor = transform(query_image).unsqueeze(0)
                query_tensor = query_tensor.to('cuda')

                # 提取特征向量
                query_embedding = net.forward_once(query_tensor).detach().cpu().numpy()

                # 预测标签
                similarities_5 = []

                for support_label_5, support_embedding_5 in support_dataset_5:
                    # 计算余弦相似度
                    similarity_cosine_5 = np.dot(query_embedding, support_embedding_5.T) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(support_embedding_5))
                    similarities_5.append(similarity_cosine_5)

                similarities_5 = np.array(similarities_5)
                # 综合计算相似度
                labels_5 = np.argmax(similarities_5)
                similarity_5 = np.round(similarities_5[labels_5], decimals=9)

                # 判断预测是否正确
                if support_dataset_5[labels_5][0] == class_name:
                    correct_num_5 += 1

                sim_5 = (similarities_5)

                #print(support_dataset[labels][0])
                #image = (image_name)
                #predict = (support_dataset[labels][0])
                list = [sim_5]
                data = pd.DataFrame([list])
                data.to_csv('./csv/similarity/similarity-cl-5-{}-{}.csv'.format(epoch + 1,best_acc*100), mode='a', header=False, index=False)


    # 输出准确率
    accuracy_5 = correct_num_5 / total_num

    #print(f'准确率：{accuracy:.2%}')

    if accuracy_5 > best_acc:
        best_acc = accuracy_5
        #save_path = f"./pth/Ships dataset-pre-{accuracy * 100:.4f}.pth"
        #torch.save(net.state_dict(), save_path)
        # 获取query集的特征向量
        query_embeddings = []
        query_labels = []
        for class_name in os.listdir(query_dir):
            class_dir = os.path.join(query_dir, class_name)
            if os.path.isdir(class_dir):
                for image_name in os.listdir(class_dir):
                    image_path = os.path.join(class_dir, image_name)
                    query_image = Image.open(image_path).convert('RGB')
                    query_tensor = transform(query_image).unsqueeze(0)
                    query_tensor = query_tensor.to('cuda')

                    # 提取特征向量
                    query_embedding = net.forward_once(query_tensor).detach().cpu().numpy()
                    query_embeddings.append(query_embedding)
                    query_labels.append(class_name)

        query_embeddings = np.concatenate(query_embeddings, axis=0)
        # print(len(query_embeddings))
        query_labels = np.array(query_labels)
        # print(len(query_labels))

        # 使用t-SNE降维
        pca_2 = PCA(n_components=2)
        pca_embeddings_2 = pca_2.fit_transform(query_embeddings)

        pca_3 = PCA(n_components=3)
        pca_embeddings_3 = pca_3.fit_transform(query_embeddings)

        tsne_2 = TSNE(n_components=3)
        tsne_embeddings_2 = tsne_2.fit_transform(query_embeddings)

        tsne_3 = TSNE(n_components=3)
        tsne_embeddings_3 = tsne_3.fit_transform(query_embeddings)

        # 绘制散点图
        unique_labels = np.unique(query_labels)
        colors = [(r / 255, g / 255, b / 255) for r, g, b in
                  [(2, 48, 71), (14, 91, 118), (26, 134, 163),
                   (70, 172, 202), (155, 207, 232), (243, 249, 252),
                   (255, 202, 95), (254, 168, 9), (251, 132, 2)]]
        #for i, label in enumerate(unique_labels):
            #indices = np.where(query_labels == label)[0]
            #plt.scatter(tsne_embeddings_2[indices, 0], tsne_embeddings_2[indices, 1], s=16, c=colors[i] , label=label)

        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.font_manager import FontProperties

        # 创建三维子图
        fig = plt.figure()
        ax1 = fig.add_subplot(projection='3d')
        #ax2 = fig.add_subplot( projection='3d')

        # 设置子图的背景色为白色
        ax1.set_facecolor('white')
        #ax2.set_facecolor('white')

        # 绘制三维散点图
        for i, label in enumerate(unique_labels):
            indices = np.where(query_labels == label)[0]
            ax1.scatter(tsne_embeddings_3[indices, 0], tsne_embeddings_3[indices, 1], tsne_embeddings_3[indices, 2], s=16,
                        c=colors[i], label=label)
        #plt.legend(loc="best", prop={'family': 'Times New Roman', 'size': '8'})
        title_font = FontProperties(family='Times New Roman', size=18)
        tick_font = FontProperties(family='Times New Roman', size=12)

        # 设置主图的标题
        ax1.set_title('Query Embedding-{:.2f}%'.format(accuracy_5 * 100), fontproperties=title_font)
        plt.savefig('./tsne-{:.2f}.pdf'.format(accuracy_5 * 100), format='pdf')
        plt.savefig('./tsne-{:.2f}.png'.format(accuracy_5 * 100), format='png', dpi=600)
        #plt.show()

        # 设置刻度字体和旋转角度
        ax1.xaxis.set_ticklabels(ax1.get_xticks(), fontproperties=tick_font)
        ax1.yaxis.set_ticklabels(ax1.get_yticks(), fontproperties=tick_font)
        ax1.zaxis.set_ticklabels(ax1.get_zticks(), fontproperties=tick_font)

        # 绘制子图的散点图和图例
        #for i, label in enumerate(unique_labels):
            #indices = np.where(query_labels == label)[0]
            #ax2.scatter(tsne_embeddings_3[indices, 0], tsne_embeddings_3[indices, 1], tsne_embeddings_3[indices, 2],
                        #s=16,
                        #c=colors[i], label=label)

        # 设置刻度字体和旋转角度

        #ax2.xaxis.set_ticklabels(ax2.get_xticks(), fontproperties=tick_font)
        #ax2.yaxis.set_ticklabels(ax2.get_yticks(), fontproperties=tick_font)
        #ax2.zaxis.set_ticklabels(ax2.get_zticks(), fontproperties=tick_font)

        # 显示图形
        #plt.show()

        #plt.savefig('./pdf/{}-{}.pdf'.format(embedding_dim, Margin), format='pdf')

        #plt.title('Query Embedding-t-SNE', family='Times New Roman', fontsize=18)
        #plt.xticks(fontsize=12, family='Times New Roman')  # 设置x轴刻度字号
        #plt.yticks(fontsize=12, family='Times New Roman')  # 设置y轴刻度字号
        #plt.legend(loc="best", prop={'family': 'Times New Roman', 'size': '8'})

        #plt.savefig('./pdf/{}-{}.pdf'.format(embedding_dim,Margin), format='pdf')

    end_time = time.time()
    print(f'| predict number: {correct_num_5} ')
    print(f'| total number: {total_num} ')
    print(f'| Query accuracy: {accuracy_5 * 100:.3f} %')
    print(f'| Training Time: {end_time - start_time:.2f}s')
    print('=' * 100)

    jc = (epoch + 1)
    t_loss = (train_loss)
    #num = (correct_num)
    acc_5 = (accuracy_5 * 100)

